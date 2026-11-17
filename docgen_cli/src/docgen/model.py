"""Model loading and generation.

Generation settings match the evaluation in the capstone notebooks: 4 beams,
128 new tokens, early stopping, 512 input tokens.
"""

import json
import os
import sys
from typing import List, Optional

DEFAULT_BASE = "Salesforce/codet5p-220m"


def _load_tokenizer(path_or_id: str):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(path_or_id)
    except Exception as first_error:
        if os.path.isdir(path_or_id):
            raise
        # Hub copy of the base model: apply the same tokenizer fix used in the notebooks.
        from huggingface_hub import snapshot_download

        local = snapshot_download(path_or_id)
        added = os.path.join(local, "added_tokens.json")
        stm = os.path.join(local, "special_tokens_map.json")
        try:
            if os.path.exists(added):
                os.remove(added)
            if os.path.exists(stm):
                with open(stm, "r", encoding="utf-8") as f:
                    m = json.load(f)
                extra = m.get("additional_special_tokens")
                if extra and isinstance(extra[0], dict):
                    m["additional_special_tokens"] = [t.get("content", t) if isinstance(t, dict) else t for t in extra]
                    with open(stm, "w", encoding="utf-8") as f:
                        json.dump(m, f)
        except OSError:
            raise first_error
        return AutoTokenizer.from_pretrained(local)


def _load_seq2seq(path_or_id: str):
    import torch
    from transformers import AutoModelForSeq2SeqLM

    try:
        return AutoModelForSeq2SeqLM.from_pretrained(path_or_id, dtype=torch.float32)
    except TypeError:  # older transformers versions use torch_dtype
        return AutoModelForSeq2SeqLM.from_pretrained(path_or_id, torch_dtype=torch.float32)


class Summarizer:
    """Wraps the fine-tuned model. One instance, many calls."""

    def __init__(
        self,
        model_path: str,
        adapter: Optional[str] = None,
        base_model: str = DEFAULT_BASE,
        device: Optional[str] = None,
        num_beams: int = 4,
        max_new_tokens: int = 128,
        max_source_len: int = 512,
    ):
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.max_source_len = max_source_len

        if adapter:
            from peft import PeftModel

            self.tokenizer = _load_tokenizer(base_model)
            base = _load_seq2seq(base_model)
            model = PeftModel.from_pretrained(base, adapter).merge_and_unload()
        else:
            if not os.path.isdir(model_path):
                raise FileNotFoundError(
                    f"Model folder not found: {model_path}\n"
                    "Download the 'best' folder of your full fine-tune from Google Drive, place it there, "
                    "or pass --model PATH (or set DOCGEN_MODEL)."
                )
            self.tokenizer = _load_tokenizer(model_path)
            model = _load_seq2seq(model_path)
        self.model = model.to(self.device).eval()

    def summarize(self, texts: List[str], batch_size: int = 8, progress: bool = True) -> List[str]:
        torch = self.torch
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out: List[Optional[str]] = [None] * len(texts)
        use_amp = self.device == "cuda"
        with torch.no_grad():
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                batch = self.tokenizer(
                    [texts[i] for i in idx],
                    max_length=self.max_source_len,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    gen = self.model.generate(
                        **batch,
                        num_beams=self.num_beams,
                        max_new_tokens=self.max_new_tokens,
                        early_stopping=True,
                    )
                for i, d in zip(idx, self.tokenizer.batch_decode(gen, skip_special_tokens=True)):
                    out[i] = d.strip()
                    if os.environ.get("DOCGEN_DEBUG"):
                        print(f"[raw] {out[i]!r}", file=sys.stderr)
                if progress:
                    done = min(start + batch_size, len(order))
                    print(f"  generated {done}/{len(order)}", file=sys.stderr, flush=True)
        return [o or "" for o in out]
