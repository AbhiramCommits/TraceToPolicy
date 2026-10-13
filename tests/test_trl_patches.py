import torch

from tracetopolicy.trl_patches import ChunkedLogProbFunction, patch_trl_for_cpu


def _manual(h, W, b, y, g):
    logits = h @ W.t() + b
    logp = torch.log_softmax(logits, dim=-1)
    mlp = logp.gather(1, y.unsqueeze(1)).squeeze(1)
    mlp.backward(g)
    return mlp.detach()


def test_fused_logprob_matches_manual_forward_and_backward():
    torch.manual_seed(0)
    h = torch.randn(7, 8, dtype=torch.float64, requires_grad=True)
    W = torch.randn(11, 8, dtype=torch.float64, requires_grad=True)
    b = torch.randn(11, dtype=torch.float64, requires_grad=True)
    y = torch.randint(0, 11, (7,))
    g = torch.randn(7, dtype=torch.float64)

    lp, ent, lssp, ml, top1 = ChunkedLogProbFunction.apply(
        h,
        W,
        b,
        y,
        1.0,
        3,
        None,
        1.0,
        ("log_probs", "entropy", "log_sum_sq_probs", "mean_logits", "is_top1"),
    )
    assert ent is not None and lssp is not None and ml is not None and top1 is not None
    assert top1.dtype == torch.bool

    lp.backward(g)
    gh, gW, gb = h.grad.clone(), W.grad.clone(), b.grad.clone()

    h2 = h.detach().clone().requires_grad_(True)
    W2 = W.detach().clone().requires_grad_(True)
    b2 = b.detach().clone().requires_grad_(True)
    mlp = _manual(h2, W2, b2, y, g)
    assert torch.allclose(lp.double(), mlp, atol=1e-5)
    assert torch.allclose(gh, h2.grad, atol=1e-5)
    assert torch.allclose(gW, W2.grad, atol=1e-5)
    assert torch.allclose(gb, b2.grad, atol=1e-5)


def test_patch_returns_bool():
    assert isinstance(patch_trl_for_cpu(), bool)
