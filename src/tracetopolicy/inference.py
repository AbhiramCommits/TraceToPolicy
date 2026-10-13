"""One generation interface, two backends.

``VLLMGenerator`` is used when a CUDA GPU and the ``vllm`` package are present;
otherwise ``HFGenerator`` wraps plain ``transformers.generate``. Both expose the
same API: :meth:`render` (chat template) and :meth:`generate` (batched). Eval
uses greedy decoding; trace harvesting uses temperature sampling.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Generation:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    meta: dict = field(default_factory=dict)


class Generator(ABC):
    """Batched text generator over chat messages."""

    name: str = "generator"
    backend: str = "abstract"

    @abstractmethod
    def render(self, messages: list[dict]) -> str: ...

    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 192,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> list[Generation]: ...

    def chat(
        self,
        conversations: list[list[dict]],
        max_new_tokens: int = 192,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> list[Generation]:
        prompts = [self.render(c) for c in conversations]
        return self.generate(
            prompts, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, seed=seed
        )


# --------------------------------------------------------------------------- #
# Transformers backend
# --------------------------------------------------------------------------- #
class HFGenerator(Generator):
    backend = "transformers"

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        adapter_path: str | None = None,
        load_in_4bit: bool = False,
        max_seq_length: int = 1024,
        batch_size: int = 4,
        name: str | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.name = name or model_name
        self.device = device
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.adapter_path = adapter_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        kwargs: dict = {"trust_remote_code": True, "dtype": dtype}
        if load_in_4bit and device == "cuda":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if not (load_in_4bit and device == "cuda"):
            self.model.to(device)
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def render(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 192,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> list[Generation]:
        torch = self.torch
        if seed is not None:
            torch.manual_seed(seed)
        results: list[Generation] = []
        for bi, start in enumerate(range(0, len(prompts), self.batch_size)):
            batch = prompts[start : start + self.batch_size]
            t0 = time.perf_counter()
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
            )
            input_ids = enc["input_ids"].to(self.device)
            attn = enc["attention_mask"].to(self.device)
            gen_kwargs: dict = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if temperature and temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
            else:
                gen_kwargs.update(do_sample=False)
            with torch.no_grad():
                out = self.model.generate(input_ids=input_ids, attention_mask=attn, **gen_kwargs)
            new_tokens = out[:, input_ids.shape[1] :]
            texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            elapsed = (time.perf_counter() - t0) * 1000.0 / max(len(batch), 1)
            prompt_lens = attn.sum(dim=1).tolist()
            comp_lens = (new_tokens != self.tokenizer.pad_token_id).sum(dim=1).tolist()
            for i, text in enumerate(texts):
                results.append(
                    Generation(
                        text=text.strip(),
                        prompt_tokens=int(prompt_lens[i]),
                        completion_tokens=int(comp_lens[i]),
                        latency_ms=elapsed,
                    )
                )
            # release MPS/CUDA cache so a long harvest does not grow swap
            del out, new_tokens, input_ids, attn
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()
        return results


# --------------------------------------------------------------------------- #
# vLLM backend
# --------------------------------------------------------------------------- #
class VLLMGenerator(Generator):
    backend = "vllm"

    def __init__(
        self,
        model_name: str,
        adapter_path: str | None = None,
        max_seq_length: int = 1024,
        name: str | None = None,
        gpu_memory_utilization: float = 0.85,
        dtype: str = "auto",
    ):
        from transformers import AutoTokenizer
        from vllm import LLM

        self.model_name = model_name
        self.name = name or model_name
        self.max_seq_length = max_seq_length
        self.adapter_path = adapter_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.llm = LLM(
            model=model_name,
            dtype=dtype,
            max_model_len=max_seq_length,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_lora=bool(adapter_path),
            trust_remote_code=True,
        )

    def render(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 192,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> list[Generation]:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        lora_request = None
        if self.adapter_path:
            lora_request = LoRARequest("adapter", 1, self.adapter_path)
        t0 = time.perf_counter()
        outputs = self.llm.generate(prompts, params, lora_request=lora_request)
        elapsed = (time.perf_counter() - t0) * 1000.0 / max(len(prompts), 1)
        results = []
        for out in outputs:
            text = out.outputs[0].text.strip()
            results.append(
                Generation(
                    text=text,
                    prompt_tokens=len(out.prompt_token_ids or []),
                    completion_tokens=len(out.outputs[0].token_ids or []),
                    latency_ms=elapsed,
                )
            )
        return results


# --------------------------------------------------------------------------- #
# Auto-selection
# --------------------------------------------------------------------------- #
def backend_available() -> tuple[bool, bool]:
    """Return ``(cuda, vllm)`` availability without importing heavy modules eagerly."""
    cuda = False
    try:
        import torch

        cuda = torch.cuda.is_available()
    except Exception:
        pass
    vllm = False
    try:
        import importlib.util

        vllm = importlib.util.find_spec("vllm") is not None
    except Exception:
        pass
    return cuda, vllm


def get_generator(
    model_name: str,
    device: str | None = None,
    adapter_path: str | None = None,
    backend: str = "auto",
    load_in_4bit: bool = False,
    max_seq_length: int = 1024,
    batch_size: int = 4,
    name: str | None = None,
) -> Generator:
    """Pick vLLM on CUDA when available, else the transformers backend."""
    cuda, vllm = backend_available()
    if backend == "auto":
        backend = "vllm" if (cuda and vllm) else "transformers"
    if backend == "vllm":
        if not (cuda and vllm):
            raise RuntimeError("vLLM requested but no CUDA GPU / vllm package available")
        return VLLMGenerator(
            model_name, adapter_path=adapter_path, max_seq_length=max_seq_length, name=name
        )
    if device is None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        except Exception:
            device = "cpu"
    return HFGenerator(
        model_name,
        device=device,
        adapter_path=adapter_path,
        load_in_4bit=load_in_4bit,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        name=name,
    )


def get_generator_with_fallback(
    model_name: str,
    fallbacks: list[str] | None = None,
    **kwargs,
) -> tuple[Generator, list[str]]:
    """Try ``model_name`` (with one retry), then fall back down the tier list.

    Returns the generator and a list of human-readable notes about fallbacks.
    """
    notes: list[str] = []
    candidates = [model_name] + list(fallbacks or [])
    for idx, candidate in enumerate(candidates):
        for attempt in (1, 2):
            try:
                gen = get_generator(candidate, **kwargs)
                if idx > 0:
                    notes.append(f"fell back to {candidate} after {model_name} failed")
                return gen, notes
            except Exception as exc:  # noqa: BLE001
                notes.append(f"load {candidate} attempt {attempt} failed: {str(exc)[:160]}")
    raise RuntimeError(f"could not load any model from {candidates}: {notes}")
