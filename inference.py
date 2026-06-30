from __future__ import annotations

import argparse
import base64
import json
import pickle
import sys
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from models.LFM2.model import LFM2, LFM2_5_350M_Config


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = "~/workspace/data/models/LFM2.5-350M-Base/best.pt"
DEFAULT_TOKENIZER = ROOT / "tokenizer" / "LFM2_5_VL_tiktoken"
DEFAULT_STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")


def log(message: str) -> None:
    print(message, file=sys.stderr)


class TokenizerAdapter:
    @property
    def n_vocab(self) -> int:
        raise NotImplementedError

    @property
    def special_token_ids(self) -> set[int]:
        raise NotImplementedError

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        raise NotImplementedError

    def token_to_id(self, token: str) -> int | None:
        raise NotImplementedError


class TiktokenAdapter(TokenizerAdapter):
    def __init__(self, tokenizer_dir: Path) -> None:
        import tiktoken

        meta_path = tokenizer_dir / "tokenizer_meta.json"
        with meta_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)

        mergeable_ranks: dict[bytes, int] = {}
        with (tokenizer_dir / meta["bpe_file"]).open("rb") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                token_b64, rank = line.split()
                mergeable_ranks[base64.b64decode(token_b64)] = int(rank)

        self.encoding = tiktoken.Encoding(
            name=meta["name"],
            pat_str=meta["pat_str"],
            mergeable_ranks=mergeable_ranks,
            special_tokens=meta["special_tokens"],
            explicit_n_vocab=meta["n_vocab"],
        )
        self._special_tokens = dict(meta["special_tokens"])
        self._special_token_ids = set(self._special_tokens.values())

    @property
    def n_vocab(self) -> int:
        return int(self.encoding.n_vocab)

    @property
    def special_token_ids(self) -> set[int]:
        return self._special_token_ids

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.encoding.encode(text, allowed_special="all")
        if add_bos:
            bos_id = self.token_to_id("<|startoftext|>")
            if bos_id is None:
                raise ValueError("Tokenizer does not define <|startoftext|>")
            ids = [bos_id, *ids]
        if add_eos:
            eos_id = self.token_to_id("<|im_end|>")
            if eos_id is None:
                raise ValueError("Tokenizer does not define <|im_end|>")
            ids = [*ids, eos_id]
        return ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        if skip_special_tokens:
            ids = [token_id for token_id in ids if token_id not in self._special_token_ids]
        return self.encoding.decode(ids)

    def token_to_id(self, token: str) -> int | None:
        if token in self._special_tokens:
            return int(self._special_tokens[token])
        try:
            token_ids = self.encoding.encode(token, allowed_special="all")
        except Exception:
            return None
        return int(token_ids[0]) if len(token_ids) == 1 else None


class HFTokenizerAdapter(TokenizerAdapter):
    def __init__(self, tokenizer_dir: Path) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True)
        all_special_ids = getattr(self.tokenizer, "all_special_ids", None) or []
        self._special_token_ids = {int(token_id) for token_id in all_special_ids}

    @property
    def n_vocab(self) -> int:
        return int(len(self.tokenizer))

    @property
    def special_token_ids(self) -> set[int]:
        return self._special_token_ids

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if add_bos:
            if self.tokenizer.bos_token_id is None:
                raise ValueError("Tokenizer does not define a BOS token")
            ids = [int(self.tokenizer.bos_token_id), *ids]
        if add_eos:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer does not define an EOS token")
            ids = [*ids, int(self.tokenizer.eos_token_id)]
        return [int(token_id) for token_id in ids]

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def token_to_id(self, token: str) -> int | None:
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != self.tokenizer.unk_token_id:
            return int(token_id)
        token_ids = self.tokenizer.encode(token, add_special_tokens=False)
        return int(token_ids[0]) if len(token_ids) == 1 else None


def load_tokenizer(tokenizer_path: str | Path) -> TokenizerAdapter:
    tokenizer_dir = Path(tokenizer_path).expanduser()
    if not tokenizer_dir.is_absolute():
        tokenizer_dir = ROOT / tokenizer_dir
    if (tokenizer_dir / "tokenizer_meta.json").is_file():
        return TiktokenAdapter(tokenizer_dir)
    if (tokenizer_dir / "tokenizer.json").is_file():
        return HFTokenizerAdapter(tokenizer_dir)
    raise FileNotFoundError(
        f"Tokenizer not found at {tokenizer_dir}. Expected tokenizer_meta.json or tokenizer.json."
    )


def torch_load(path: Path, map_location: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except pickle.UnpicklingError:
        log("Falling back to torch.load(..., weights_only=False) for this checkpoint.")
        return torch.load(path, map_location=map_location, weights_only=False)


def extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("model", "state_dict", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise ValueError("Checkpoint does not contain a model state dict.")


def strip_state_dict_prefixes(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = dict(state_dict)
    prefixes = ("module.", "_orig_mod.")

    changed = True
    while changed and cleaned:
        changed = False
        for prefix in prefixes:
            if all(key.startswith(prefix) for key in cleaned):
                cleaned = {key[len(prefix) :]: value for key, value in cleaned.items()}
                changed = True
    return cleaned


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested, but it is not available.")
    return device


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32

    dtypes = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    try:
        return dtypes[dtype_arg]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype_arg}") from exc


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def load_model(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    strict: bool,
) -> tuple[LFM2, Any]:
    checkpoint_file = Path(checkpoint_path).expanduser()
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    config = LFM2_5_350M_Config()
    model = LFM2(config)
    checkpoint = torch_load(checkpoint_file, map_location="cpu")
    state_dict = strip_state_dict_prefixes(extract_state_dict(checkpoint))
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        log(f"Missing checkpoint keys: {len(missing)}")
    if unexpected:
        log(f"Unexpected checkpoint keys: {len(unexpected)}")

    model.to(device=device, dtype=dtype)
    model.eval()
    return model, checkpoint


def apply_repetition_penalty(logits: torch.Tensor, input_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    if penalty <= 0:
        raise ValueError("repetition_penalty must be positive.")

    logits = logits.clone()
    for batch_idx in range(logits.size(0)):
        token_ids = torch.unique(input_ids[batch_idx])
        token_logits = logits[batch_idx, token_ids]
        logits[batch_idx, token_ids] = torch.where(
            token_logits < 0,
            token_logits * penalty,
            token_logits / penalty,
        )
    return logits


def filter_logits(logits: torch.Tensor, *, top_k: int, top_p: float) -> torch.Tensor:
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k, dim=-1)
        logits = logits.masked_fill(logits < values[:, [-1]], -torch.inf)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = sorted_probs.cumsum(dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, -torch.inf)

        filtered_logits = torch.full_like(logits, -torch.inf)
        logits = filtered_logits.scatter(dim=-1, index=sorted_indices, src=sorted_logits)

    return logits


@torch.inference_mode()
def generate(
    model: LFM2,
    input_ids: torch.Tensor,
    *,
    tokenizer: TokenizerAdapter,
    max_new_tokens: int,
    context_length: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    stop_ids: set[int],
    device: torch.device,
    dtype: torch.dtype,
    stream: bool,
    skip_special_tokens: bool,
) -> torch.Tensor:
    generated = input_ids

    for _ in range(max_new_tokens):
        model_input = generated[:, -context_length:]
        with autocast_context(device, dtype):
            logits = model(model_input)[:, -1, :].float()

        logits = apply_repetition_penalty(logits, generated, repetition_penalty)
        if temperature <= 0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            logits = filter_logits(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat((generated, next_token), dim=1)
        token_id = int(next_token.item())

        if stream and token_id not in stop_ids:
            piece = tokenizer.decode([token_id], skip_special_tokens=skip_special_tokens)
            print(piece, end="", flush=True)

        if token_id in stop_ids:
            break

    if stream:
        print()
    return generated


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is not None:
        return Path(args.prompt_file).expanduser().read_text(encoding="utf-8")
    if args.prompt is not None:
        return args.prompt
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return input("Prompt: ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with the local LFM2.5 350M model.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to best.pt or another checkpoint.")
    parser.add_argument("--tokenizer-path", default=str(DEFAULT_TOKENIZER), help="Local tokenizer directory.")
    parser.add_argument("--prompt", help="Prompt text. If omitted, stdin is used.")
    parser.add_argument("--prompt-file", help="Read prompt text from a UTF-8 file.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.8, help="Use 0 for greedy decoding.")
    parser.add_argument("--top-k", type=int, default=50, help="0 disables top-k filtering.")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, cuda:0, etc.")
    parser.add_argument("--dtype", default="auto", help="auto, fp32, bf16, or fp16.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--add-bos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--add-eos", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stop-token", action="append", default=None, help="Special token that stops generation.")
    parser.add_argument("--no-stop", action="store_true", help="Disable default and custom stop tokens.")
    parser.add_argument("--include-stop-token", action="store_true", help="Include the final stop token in decoded output.")
    parser.add_argument("--skip-special-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--truncate-prompt", action="store_true", help="Left-truncate prompts longer than context length.")
    parser.add_argument("--print-full-text", action="store_true", help="Print prompt plus completion instead of only completion.")
    parser.add_argument("--stream", action="store_true", help="Print generated tokens as they are produced.")
    parser.add_argument("--allow-mismatch", action="store_true", help="Load checkpoints with missing/unexpected keys.")
    parser.add_argument("--compile", action="store_true", help="Run torch.compile(model) before generation.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be non-negative.")
    if args.context_length <= 0:
        raise ValueError("--context-length must be positive.")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative.")
    if args.top_k < 0:
        raise ValueError("--top-k must be non-negative.")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in the range (0, 1].")
    if args.prompt is not None and args.prompt_file is not None:
        raise ValueError("Use either --prompt or --prompt-file, not both.")


def resolve_stop_ids(tokenizer: TokenizerAdapter, args: argparse.Namespace) -> set[int]:
    if args.no_stop:
        return set()

    stop_tokens = args.stop_token if args.stop_token is not None else list(DEFAULT_STOP_TOKENS)
    stop_ids: set[int] = set()
    for token in stop_tokens:
        token_id = tokenizer.token_to_id(token)
        if token_id is None:
            log(f"Warning: stop token {token!r} is not a single tokenizer token; ignoring it.")
            continue
        stop_ids.add(token_id)
    return stop_ids


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = load_tokenizer(args.tokenizer_path)
    model_config = LFM2_5_350M_Config()
    if tokenizer.n_vocab != model_config.vocab_size:
        raise ValueError(
            f"Tokenizer vocab ({tokenizer.n_vocab}) does not match model vocab ({model_config.vocab_size})."
        )

    prompt = read_prompt(args)
    prompt_ids = tokenizer.encode(prompt, add_bos=args.add_bos, add_eos=args.add_eos)
    if not prompt_ids:
        raise ValueError("Prompt produced no tokens. Use --add-bos or pass non-empty prompt text.")
    if len(prompt_ids) > args.context_length:
        if not args.truncate_prompt:
            raise ValueError(
                f"Prompt is {len(prompt_ids)} tokens, longer than --context-length {args.context_length}. "
                "Pass --truncate-prompt to keep the final context window."
            )
        prompt_ids = prompt_ids[-args.context_length :]
        log(f"Prompt truncated to the final {args.context_length} tokens.")

    if args.verbose:
        log(f"Device: {device}")
        log(f"Dtype: {dtype}")
        log(f"Tokenizer vocab: {tokenizer.n_vocab}")
        log(f"Prompt tokens: {len(prompt_ids)}")

    log(f"Loading checkpoint: {Path(args.checkpoint).expanduser()}")
    model, checkpoint = load_model(
        args.checkpoint,
        device=device,
        dtype=dtype,
        strict=not args.allow_mismatch,
    )
    if args.compile:
        log("Compiling model...")
        model.warmup_caches(args.context_length, device, dtype)
        model = torch.compile(model)

    if isinstance(checkpoint, Mapping) and args.verbose:
        step = checkpoint.get("step")
        val_loss = checkpoint.get("val_loss")
        best_val_loss = checkpoint.get("best_val_loss")
        if step is not None:
            log(f"Checkpoint step: {step}")
        if val_loss is not None:
            log(f"Checkpoint val_loss: {val_loss}")
        if best_val_loss is not None:
            log(f"Checkpoint best_val_loss: {best_val_loss}")

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    stop_ids = resolve_stop_ids(tokenizer, args)

    output_ids = generate(
        model,
        input_ids,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        context_length=args.context_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        stop_ids=stop_ids,
        device=device,
        dtype=dtype,
        stream=args.stream,
        skip_special_tokens=args.skip_special_tokens,
    )[0].tolist()

    if args.stream:
        return

    if args.print_full_text:
        decoded_ids = output_ids
    else:
        decoded_ids = output_ids[len(prompt_ids) :]

    if decoded_ids and decoded_ids[-1] in stop_ids and not args.include_stop_token:
        decoded_ids = decoded_ids[:-1]

    text = tokenizer.decode(decoded_ids, skip_special_tokens=args.skip_special_tokens)
    print(text)


if __name__ == "__main__":
    main()
