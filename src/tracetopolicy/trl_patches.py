"""Make TRL's fused LM-head work without Triton (i.e. on CPU / Apple Silicon).

TRL 1.x patches the model with a *fused LM head* that computes per-token
log-probabilities with a Triton kernel. Triton ships with PyTorch on Linux only,
so on macOS/CPU ``trl.trainer.utils._ChunkedLogProbFunction`` is ``None`` and
every SFT/DPO training step crashes with ``'NoneType' object has no attribute
'apply'``.

This module provides a pure-PyTorch, chunked, autograd-capable drop-in and
installs it into ``trl.trainer.utils``. The forward never materializes the full
``(tokens, vocab)`` logits for backward (they are recomputed per chunk in
``backward``), so memory stays bounded. On CUDA with Triton the real kernel is
left untouched.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_CHUNK = 1024


def _scale_logits(logits, temperature, logit_scale, softcap):
    if logit_scale is not None and logit_scale != 1.0:
        logits = logits * logit_scale
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    if temperature is not None and temperature != 1.0:
        logits = logits / temperature
    return logits


class ChunkedLogProbFunction(torch.autograd.Function):
    """Drop-in for ``trl.kernels.ChunkedLogProbFunction`` (pure PyTorch)."""

    @staticmethod
    def forward(
        ctx,
        hidden_states,
        weight,
        bias,
        labels,
        temperature,
        chunk_size,
        final_logit_softcapping,
        logit_scale,
        outputs,
    ):
        hidden = hidden_states.contiguous()
        labels = labels.contiguous()
        outputs = tuple(outputs) if outputs else ("log_probs",)
        want_entropy = "entropy" in outputs
        want_top1 = "is_top1" in outputs
        want_lssp = "log_sum_sq_probs" in outputs
        want_mean_logits = "mean_logits" in outputs

        n = hidden.shape[0]
        step = max(1, min(int(chunk_size or _CHUNK), _CHUNK, n if n else 1))
        log_probs_list, entropy_list, top1_list, lssp_list, mlog_list = [], [], [], [], []
        for i in range(0, n, step):
            h = hidden[i : i + step]
            y = labels[i : i + step]
            logits = _scale_logits(
                F.linear(h, weight, bias), temperature, logit_scale, final_logit_softcapping
            )
            logp = F.log_softmax(logits.float(), dim=-1)
            p = logp.exp()
            log_probs_list.append(logp.gather(1, y.unsqueeze(1)).squeeze(1))
            if want_entropy:
                entropy_list.append(-(p * logp).sum(dim=-1))
            if want_top1:
                top1_list.append((logits.argmax(dim=-1) == y).detach())
            if want_lssp:
                lssp_list.append(torch.log((p * p).sum(dim=-1) + 1e-12).detach())
            if want_mean_logits:
                mlog_list.append(logits.mean(dim=-1).detach())

        log_probs = (
            torch.cat(log_probs_list)
            if log_probs_list
            else hidden.new_zeros(0, dtype=torch.float32)
        )
        entropy = torch.cat(entropy_list) if entropy_list else None
        top1 = torch.cat(top1_list) if top1_list else None
        lssp = torch.cat(lssp_list) if lssp_list else None
        mean_logits = torch.cat(mlog_list) if mlog_list else None

        ctx.save_for_backward(hidden, weight, bias, labels)
        ctx.mark_non_differentiable(*(x for x in (lssp, mean_logits, top1) if x is not None))
        ctx.temperature = temperature
        ctx.softcap = final_logit_softcapping
        ctx.logit_scale = logit_scale
        ctx.step = step
        ctx.has_bias = bias is not None
        # (log_probs, entropy, log_sum_sq_probs, mean_logits, is_top1)
        return log_probs, entropy, lssp, mean_logits, top1

    @staticmethod
    def backward(ctx, grad_log_probs, grad_entropy, grad_lssp, grad_ml, grad_top1):
        hidden, weight, bias, labels = ctx.saved_tensors
        n = hidden.shape[0]
        if grad_log_probs is None:
            grad_log_probs = hidden.new_zeros(n)
        g = grad_log_probs
        grad_hidden = torch.zeros_like(hidden)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias) if ctx.has_bias else None
        for i in range(0, n, ctx.step):
            h = hidden[i : i + ctx.step]
            y = labels[i : i + ctx.step]
            gi = g[i : i + ctx.step]
            logits = _scale_logits(
                F.linear(h, weight, bias), ctx.temperature, ctx.logit_scale, ctx.softcap
            )
            logp = F.log_softmax(logits.float(), dim=-1)
            p = logp.exp()
            gl = -p * gi.unsqueeze(1)
            gl.scatter_add_(1, y.unsqueeze(1), gi.unsqueeze(1))
            gl = gl.to(h.dtype)
            grad_hidden[i : i + ctx.step] = gl @ weight
            grad_weight += gl.transpose(0, 1) @ h
            if grad_bias is not None:
                grad_bias += gl.sum(dim=0)
        # grads for: hidden, weight, bias, labels, temperature, chunk_size, softcap, logit_scale, outputs
        return grad_hidden, grad_weight, grad_bias, None, None, None, None, None, None


def patch_trl_for_cpu() -> bool:
    """Install the pure-PyTorch kernel if Triton is unavailable. Returns applied?"""
    try:
        import trl.trainer.utils as utils

        if getattr(utils, "_ChunkedLogProbFunction", None) is None:
            utils._ChunkedLogProbFunction = ChunkedLogProbFunction
            return True
    except Exception:
        return False
    return False
