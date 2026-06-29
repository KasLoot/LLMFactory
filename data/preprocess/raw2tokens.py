"""Encode Dolma3 JSONL/Zstandard text shards into packed token binaries.

This is the authoritative pretraining encoder: it writes contiguous `.bin`
token shards plus a manifest that the training dataloader can consume.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import shutil
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import zstandard as zstd
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase


DEFAULT_MAX_VAL_TOKENS_PER_SUBSET = 20_000_000
DEFAULT_FLUSH_BATCH_TOKENS = 8_000_000
HASH_MODULUS = 1_000_000
RESUME_STATE_VERSION = 1


@dataclass
class EncoderConfig:
    data_root: Path
    tokenizer_path: str
    output_root: Path
    target_config: Path
    context_length: int
    train_shard_tokens: int
    val_shard_tokens: int
    flush_batch_tokens: int
    val_fraction: float
    seed: int
    min_doc_tokens: int
    max_doc_tokens: int
    dtype: str
    dry_run: bool
    overwrite: bool
    resume: bool
    limit_docs_per_subset: int | None
    num_workers: int
    max_val_tokens_per_subset: int
    skip_missing_subsets: bool
    checkpoint_interval_docs: int


@dataclass
class SubsetStats:
    target_tokens: int
    actual_tokens: int = 0
    train_tokens: int = 0
    val_tokens: int = 0
    docs_seen: int = 0
    docs_written: int = 0
    docs_skipped_empty: int = 0
    docs_skipped_short: int = 0
    docs_malformed_json: int = 0
    docs_split_long: int = 0
    shards_seen: int = 0
    completed: bool = False


@dataclass
class ResumeState:
    current_subset: str | None
    current_cursor: dict[str, Any] | None
    train_writer: dict[str, Any]
    val_writer: dict[str, Any]
    subset_stats: dict[str, SubsetStats]


@dataclass
class TokenShardWriter:
    output_dir: Path
    split_name: str
    dtype: np.dtype
    shard_token_limit: int
    flush_batch_tokens: int = DEFAULT_FLUSH_BATCH_TOKENS
    dry_run: bool = False
    shard_index: int = 0
    total_tokens: int = 0
    shards: list[dict[str, Any]] = field(default_factory=list)
    _current_shard_tokens: int = 0
    _buffer_chunks: list[np.ndarray] = field(default_factory=list)
    _buffer_token_count: int = 0

    def __post_init__(self) -> None:
        if self.shard_token_limit <= 0:
            raise ValueError("shard_token_limit must be positive")
        if self.flush_batch_tokens <= 0:
            raise ValueError("flush_batch_tokens must be positive")
        if not self.dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def current_shard(self) -> str:
        return self._relative_filename_for_index(self.shard_index)

    @property
    def token_count(self) -> int:
        return self.total_tokens + self._current_shard_tokens + self._buffer_token_count

    def _filename_for_index(self, index: int) -> str:
        return f"tokens_{index:06d}.bin"

    def _relative_filename_for_index(self, index: int) -> str:
        return f"{self.split_name}/{self._filename_for_index(index)}"

    def write(self, tokens: list[int] | np.ndarray) -> None:
        if len(tokens) == 0:
            return

        values = np.asarray(tokens, dtype=self.dtype)

        pos = 0
        while pos < len(values):
            room = self.shard_token_limit - self._current_shard_tokens - self._buffer_token_count
            if room <= 0:
                self.flush_pending()
                self._finalize_current_shard()
                continue
            take = min(room, len(values) - pos)
            self._buffer_chunks.append(values[pos : pos + take])
            self._buffer_token_count += take
            pos += take
            if (
                self._buffer_token_count >= self.flush_batch_tokens
                or self._current_shard_tokens + self._buffer_token_count >= self.shard_token_limit
            ):
                self.flush_pending()

    def _write_buffer_chunks(self, fh: Any) -> None:
        batch_chunks: list[np.ndarray] = []
        batch_token_count = 0

        for chunk in self._buffer_chunks:
            offset = 0
            while offset < len(chunk):
                room = self.flush_batch_tokens - batch_token_count
                take = min(room, len(chunk) - offset)
                batch_chunks.append(chunk[offset : offset + take])
                batch_token_count += take
                offset += take

                if batch_token_count >= self.flush_batch_tokens:
                    self._write_batch(fh, batch_chunks)
                    batch_chunks = []
                    batch_token_count = 0

        if batch_token_count:
            self._write_batch(fh, batch_chunks)

    def _write_batch(self, fh: Any, chunks: list[np.ndarray]) -> None:
        if len(chunks) == 1:
            chunks[0].tofile(fh)
        else:
            np.concatenate(chunks).tofile(fh)

    def flush_pending(self) -> None:
        if self._buffer_token_count == 0:
            return

        token_count = self._buffer_token_count
        filename = self._filename_for_index(self.shard_index)

        if not self.dry_run:
            path = self.output_dir / filename
            mode = "ab" if self._current_shard_tokens else "wb"
            with path.open(mode) as fh:
                self._write_buffer_chunks(fh)

        self._current_shard_tokens += token_count
        self._buffer_chunks = []
        self._buffer_token_count = 0

        if self._current_shard_tokens >= self.shard_token_limit:
            self._finalize_current_shard()

    def _finalize_current_shard(self) -> None:
        if self._current_shard_tokens == 0:
            return

        token_count = self._current_shard_tokens
        filename = self._filename_for_index(self.shard_index)
        rel_filename = self._relative_filename_for_index(self.shard_index)

        if not self.dry_run:
            path = self.output_dir / filename
            expected_bytes = token_count * self.dtype.itemsize
            actual_bytes = path.stat().st_size
            if actual_bytes != expected_bytes:
                raise RuntimeError(
                    f"Shard size mismatch for {path}: expected {expected_bytes} bytes, got {actual_bytes}"
                )

        self.shards.append(
            {
                "filename": rel_filename,
                "tokens": token_count,
                "dtype": self.dtype.name,
                "split": self.split_name,
            }
        )
        self.total_tokens += token_count
        self.shard_index += 1
        self._current_shard_tokens = 0

    def state_dict(self) -> dict[str, Any]:
        if self._buffer_token_count != 0:
            raise RuntimeError("Cannot checkpoint writer state while tokens are still buffered")
        return {
            "split_name": self.split_name,
            "shard_index": self.shard_index,
            "total_tokens": self.total_tokens,
            "current_shard_tokens": self._current_shard_tokens,
            "shards": self.shards,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("split_name") != self.split_name:
            raise ValueError(f"Resume state split mismatch for {self.split_name}")

        self.shard_index = int(state["shard_index"])
        self.total_tokens = int(state["total_tokens"])
        self._current_shard_tokens = int(state.get("current_shard_tokens", 0))
        self.shards = [dict(meta) for meta in state.get("shards", [])]
        self._buffer_chunks = []
        self._buffer_token_count = 0

        if self.total_tokens != sum(int(meta["tokens"]) for meta in self.shards):
            raise ValueError(f"Resume state token total mismatch for {self.split_name}")
        if self.shard_index != len(self.shards):
            raise ValueError(f"Resume state shard index mismatch for {self.split_name}")

        if not self.dry_run:
            self._sync_files_to_state()

    def _sync_files_to_state(self) -> None:
        expected_names: set[str] = set()
        for meta in self.shards:
            path = self.output_dir / Path(str(meta["filename"])).name
            expected_names.add(path.name)
            expected_bytes = int(meta["tokens"]) * self.dtype.itemsize
            if not path.is_file():
                raise FileNotFoundError(f"Missing completed token shard from resume state: {path}")
            actual_bytes = path.stat().st_size
            if actual_bytes != expected_bytes:
                raise RuntimeError(
                    f"Completed shard size mismatch for {path}: expected {expected_bytes} bytes, got {actual_bytes}"
                )

        if self._current_shard_tokens:
            current_path = self.output_dir / self._filename_for_index(self.shard_index)
            expected_names.add(current_path.name)
            expected_bytes = self._current_shard_tokens * self.dtype.itemsize
            if not current_path.is_file():
                raise FileNotFoundError(f"Missing partial token shard from resume state: {current_path}")
            actual_bytes = current_path.stat().st_size
            if actual_bytes < expected_bytes:
                raise RuntimeError(
                    f"Partial shard is shorter than resume state for {current_path}: "
                    f"expected at least {expected_bytes} bytes, got {actual_bytes}"
                )
            if actual_bytes > expected_bytes:
                with current_path.open("r+b") as fh:
                    fh.truncate(expected_bytes)

        for path in self.output_dir.glob("tokens_*.bin"):
            if path.name not in expected_names:
                path.unlink()

    def close(self) -> None:
        self.flush_pending()
        self._finalize_current_shard()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--train-shard-tokens", type=int, default=100_000_000)
    parser.add_argument("--val-shard-tokens", type=int, default=20_000_000)
    parser.add_argument("--flush-batch-tokens", type=int, default=DEFAULT_FLUSH_BATCH_TOKENS)
    parser.add_argument("--val-fraction", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-doc-tokens", type=int, default=32)
    parser.add_argument("--max-doc-tokens", type=int, default=65_536)
    parser.add_argument("--dtype", choices=["auto", "uint16", "uint32"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-docs-per-subset", type=int)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-val-tokens-per-subset", type=int, default=DEFAULT_MAX_VAL_TOKENS_PER_SUBSET)
    parser.add_argument("--skip-missing-subsets", action="store_true")
    parser.add_argument("--checkpoint-interval-docs", type=int, default=10_000)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> EncoderConfig:
    return EncoderConfig(
        data_root=Path(args.data_root),
        tokenizer_path=args.tokenizer_path,
        output_root=Path(args.output_root),
        target_config=Path(args.target_config),
        context_length=args.context_length,
        train_shard_tokens=args.train_shard_tokens,
        val_shard_tokens=args.val_shard_tokens,
        flush_batch_tokens=args.flush_batch_tokens,
        val_fraction=args.val_fraction,
        seed=args.seed,
        min_doc_tokens=args.min_doc_tokens,
        max_doc_tokens=args.max_doc_tokens,
        dtype=args.dtype,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        resume=args.resume,
        limit_docs_per_subset=args.limit_docs_per_subset,
        num_workers=args.num_workers,
        max_val_tokens_per_subset=args.max_val_tokens_per_subset,
        skip_missing_subsets=args.skip_missing_subsets,
        checkpoint_interval_docs=args.checkpoint_interval_docs,
    )


def validate_config(cfg: EncoderConfig) -> None:
    if cfg.context_length <= 0:
        raise ValueError("--context-length must be positive")
    if cfg.train_shard_tokens <= 0:
        raise ValueError("--train-shard-tokens must be positive")
    if cfg.val_shard_tokens <= 0:
        raise ValueError("--val-shard-tokens must be positive")
    if cfg.flush_batch_tokens <= 0:
        raise ValueError("--flush-batch-tokens must be positive")
    if not (0.0 <= cfg.val_fraction <= 1.0):
        raise ValueError("--val-fraction must be between 0 and 1")
    if cfg.min_doc_tokens < 0:
        raise ValueError("--min-doc-tokens must be non-negative")
    if cfg.max_doc_tokens <= 0:
        raise ValueError("--max-doc-tokens must be positive")
    if cfg.num_workers != 1:
        print("[WARN] --num-workers is accepted for compatibility, but v0 streams in one process.")
    if cfg.resume and cfg.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")


def list_shards(subset_path: Path) -> list[Path]:
    return sorted(subset_path.glob("*.jsonl.zst"))


def stable_seed(seed: int, name: str) -> int:
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return seed + int.from_bytes(digest, "big")


def shuffled_shards(shards: list[Path], subset: str, seed: int) -> list[Path]:
    result = list(shards)
    random.Random(stable_seed(seed, subset)).shuffle(result)
    return result


def iter_jsonl_zst(path: Path) -> Iterable[tuple[int, dict[str, Any] | None]]:
    dctx = zstd.ZstdDecompressor()
    with path.open("rb") as fh:
        with dctx.stream_reader(fh, closefd=False) as reader:
            stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            for line_idx, line in enumerate(stream):
                try:
                    yield line_idx, json.loads(line)
                except json.JSONDecodeError:
                    yield line_idx, None


def load_target_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if "subsets" not in payload or not isinstance(payload["subsets"], dict):
        raise ValueError(f"{path} must contain a 'subsets' object")
    return payload


def max_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    vocab = tokenizer.get_vocab()
    ids = list(vocab.values())
    ids.extend(
        token_id
        for token_id in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]
        if token_id is not None
    )
    if not ids:
        raise ValueError("Tokenizer has no vocabulary ids")
    return max(ids)


def resolve_dtype(requested: str, max_id: int) -> np.dtype:
    if requested == "auto":
        return np.dtype("uint16" if max_id <= np.iinfo(np.uint16).max else "uint32")
    dtype = np.dtype(requested)
    if max_id > np.iinfo(dtype).max:
        raise ValueError(f"Tokenizer max id {max_id} does not fit in {dtype.name}")
    return dtype


def require_special_token_ids(tokenizer: PreTrainedTokenizerBase) -> tuple[int, int, int | None]:
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    if bos is None:
        raise ValueError("Tokenizer has no bos_token_id")
    if eos is None:
        raise ValueError("Tokenizer has no eos_token_id")
    return int(bos), int(eos), None if pad is None else int(pad)


def document_segments(raw_tokens: list[int], bos_id: int, eos_id: int, max_doc_tokens: int) -> list[list[int]]:
    if len(raw_tokens) <= max_doc_tokens:
        return [[bos_id, *raw_tokens, eos_id]]

    segments: list[list[int]] = []
    for start in range(0, len(raw_tokens), max_doc_tokens):
        segment = raw_tokens[start : start + max_doc_tokens]
        if start == 0:
            segment = [bos_id, *segment]
        if start + max_doc_tokens >= len(raw_tokens):
            segment = [*segment, eos_id]
        segments.append(segment)
    return segments


def validation_hash_value(subset: str, shard_name: str, line_idx: int, text: str) -> int:
    key = f"{subset}|{shard_name}|{line_idx}|{text[:128]}"
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % HASH_MODULUS


def should_send_to_val(subset: str, shard_name: str, line_idx: int, text: str, val_fraction: float) -> bool:
    threshold = int(val_fraction * HASH_MODULUS)
    return validation_hash_value(subset, shard_name, line_idx, text) < threshold


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def jsonable_config(cfg: EncoderConfig) -> dict[str, Any]:
    payload = asdict(cfg)
    for key in ["data_root", "output_root", "target_config"]:
        payload[key] = str(payload[key])
    return payload


def validate_resume_config(cfg: EncoderConfig) -> None:
    config_path = cfg.output_root / "config_used.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Cannot resume without config file: {config_path}")

    previous = json.loads(config_path.read_text(encoding="utf-8"))
    current = jsonable_config(cfg)
    ignored_keys = {"checkpoint_interval_docs", "flush_batch_tokens", "num_workers", "overwrite", "resume"}
    mismatches = []
    for key, value in current.items():
        if key in ignored_keys:
            continue
        if previous.get(key) != value:
            mismatches.append((key, previous.get(key), value))

    if mismatches:
        details = ", ".join(f"{key}: previous={old!r}, current={new!r}" for key, old, new in mismatches)
        raise ValueError(f"Cannot resume with different preprocessing config ({details})")


def subset_stats_from_dict(payload: dict[str, Any]) -> SubsetStats:
    field_names = {field.name for field in fields(SubsetStats)}
    values = {key: payload[key] for key in field_names if key in payload}
    return SubsetStats(**values)


def prepare_output_root(cfg: EncoderConfig) -> None:
    out = cfg.output_root
    manifest = out / "manifest.json"
    progress = out / "progress.json"

    if cfg.resume:
        if manifest.exists():
            print(f"[resume] Complete manifest already exists at {manifest}; nothing to resume.")
            raise SystemExit(0)
        if not progress.exists():
            raise FileNotFoundError(f"Cannot resume without progress file: {progress}")
        (out / "train").mkdir(parents=True, exist_ok=True)
        (out / "val").mkdir(parents=True, exist_ok=True)
        return

    if cfg.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        return

    if out.exists():
        if not cfg.overwrite:
            raise FileExistsError(f"{out} already exists; pass --overwrite or choose a new output root")
        shutil.rmtree(out)

    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "val").mkdir(parents=True, exist_ok=True)


def make_manifest(
    cfg: EncoderConfig,
    target_payload: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    dtype: np.dtype,
    max_id: int,
    train_writer: TokenShardWriter,
    val_writer: TokenShardWriter,
    subset_stats: dict[str, SubsetStats],
) -> dict[str, Any]:
    total_train = train_writer.token_count
    total_val = val_writer.token_count
    return {
        "format": "raw_token_binary_v1",
        "created_by": "prepare_pretrain_tokens.py",
        "dry_run": cfg.dry_run,
        "data_root": str(cfg.data_root),
        "tokenizer_path": cfg.tokenizer_path,
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": len(tokenizer),
        "max_token_id": max_id,
        "dtype": dtype.name,
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "context_length": cfg.context_length,
        "target_total_tokens": int(target_payload.get("total_target_tokens", sum(target_payload["subsets"].values()))),
        "actual_total_tokens": total_train + total_val,
        "splits": {
            "train": {"tokens": total_train, "shards": train_writer.shards},
            "val": {"tokens": total_val, "shards": val_writer.shards},
        },
        "subsets": {name: asdict(stats) for name, stats in subset_stats.items()},
    }


def make_checkpoint_payload(
    cfg: EncoderConfig,
    target_payload: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    dtype: np.dtype,
    max_id: int,
    train_writer: TokenShardWriter,
    val_writer: TokenShardWriter,
    subset_stats: dict[str, SubsetStats],
    current_subset: str | None,
    current_cursor: dict[str, Any] | None,
) -> dict[str, Any]:
    train_writer.flush_pending()
    val_writer.flush_pending()
    payload = make_manifest(cfg, target_payload, tokenizer, dtype, max_id, train_writer, val_writer, subset_stats)
    payload["updated_at_unix"] = time.time()
    payload["current_subset"] = current_subset
    payload["resume_state"] = {
        "version": RESUME_STATE_VERSION,
        "current_subset": current_subset,
        "current_cursor": current_cursor,
        "train_writer": train_writer.state_dict(),
        "val_writer": val_writer.state_dict(),
    }
    return payload


def write_progress(
    cfg: EncoderConfig,
    target_payload: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    dtype: np.dtype,
    max_id: int,
    train_writer: TokenShardWriter,
    val_writer: TokenShardWriter,
    subset_stats: dict[str, SubsetStats],
    current_subset: str | None,
    current_cursor: dict[str, Any] | None = None,
) -> None:
    payload = make_checkpoint_payload(
        cfg,
        target_payload,
        tokenizer,
        dtype,
        max_id,
        train_writer,
        val_writer,
        subset_stats,
        current_subset,
        current_cursor,
    )
    write_json_atomic(cfg.output_root / "progress.json", payload)


def load_resume_state(cfg: EncoderConfig, train_writer: TokenShardWriter, val_writer: TokenShardWriter) -> ResumeState:
    progress_path = cfg.output_root / "progress.json"
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    resume_payload = payload.get("resume_state")
    if not isinstance(resume_payload, dict):
        raise ValueError(
            f"{progress_path} does not contain resume_state. "
            "Resume is available for checkpoints created by the updated raw2tokens.py."
        )
    if int(resume_payload.get("version", 0)) != RESUME_STATE_VERSION:
        raise ValueError(f"Unsupported resume_state version in {progress_path}")

    train_writer.load_state_dict(resume_payload["train_writer"])
    val_writer.load_state_dict(resume_payload["val_writer"])
    subset_stats = {
        name: subset_stats_from_dict(stats)
        for name, stats in payload.get("subsets", {}).items()
        if isinstance(stats, dict)
    }
    return ResumeState(
        current_subset=resume_payload.get("current_subset"),
        current_cursor=resume_payload.get("current_cursor"),
        train_writer=resume_payload["train_writer"],
        val_writer=resume_payload["val_writer"],
        subset_stats=subset_stats,
    )


def encode_subset(
    subset: str,
    target_tokens: int,
    cfg: EncoderConfig,
    tokenizer: PreTrainedTokenizerBase,
    bos_id: int,
    eos_id: int,
    train_writer: TokenShardWriter,
    val_writer: TokenShardWriter,
    target_payload: dict[str, Any],
    dtype: np.dtype,
    max_id: int,
    subset_stats: dict[str, SubsetStats],
    resume_cursor: dict[str, Any] | None = None,
) -> None:
    if subset in subset_stats and subset_stats[subset].completed:
        print(f"[resume] skipping completed subset: {subset}")
        return

    subset_dir = cfg.data_root / subset
    if not subset_dir.is_dir():
        if cfg.skip_missing_subsets:
            print(f"[WARN] missing subset: {subset_dir}")
            subset_stats[subset] = SubsetStats(target_tokens=target_tokens, completed=True)
            return
        raise FileNotFoundError(f"Missing subset directory: {subset_dir}")

    stats = subset_stats.get(subset, SubsetStats(target_tokens=target_tokens))
    stats.target_tokens = target_tokens
    subset_stats[subset] = stats
    shards = shuffled_shards(list_shards(subset_dir), subset, cfg.seed)
    stats.shards_seen = len(shards)

    resume_shard = None
    resume_next_line_idx = 0
    if resume_cursor is not None:
        if resume_cursor.get("subset") != subset:
            raise ValueError(f"Resume cursor subset mismatch: expected {subset}, got {resume_cursor.get('subset')}")
        resume_shard = resume_cursor.get("shard")
        resume_next_line_idx = int(resume_cursor.get("next_line_idx", 0))
        if resume_shard not in {shard.name for shard in shards}:
            raise FileNotFoundError(f"Resume cursor shard is missing for {subset}: {resume_shard}")

    with tqdm(desc=f"encode {subset}", unit="docs", initial=stats.docs_seen) as bar:
        for shard in shards:
            if resume_shard is not None:
                if shard.name != resume_shard:
                    continue
                resume_shard = None

            if stats.actual_tokens >= target_tokens:
                break

            for line_idx, row in iter_jsonl_zst(shard):
                if line_idx < resume_next_line_idx:
                    continue
                resume_next_line_idx = 0

                if stats.actual_tokens >= target_tokens:
                    break
                if cfg.limit_docs_per_subset is not None and stats.docs_seen >= cfg.limit_docs_per_subset:
                    break

                stats.docs_seen += 1
                if row is None:
                    stats.docs_malformed_json += 1
                    bar.update(1)
                    continue

                text_value = row.get("text", "")
                text = text_value.strip() if isinstance(text_value, str) else ""
                if not text:
                    stats.docs_skipped_empty += 1
                    bar.update(1)
                    continue

                raw_tokens = tokenizer.encode(text, add_special_tokens=False)
                if len(raw_tokens) < cfg.min_doc_tokens:
                    stats.docs_skipped_short += 1
                    bar.update(1)
                    continue

                if len(raw_tokens) > cfg.max_doc_tokens:
                    stats.docs_split_long += 1
                segments = document_segments(raw_tokens, bos_id, eos_id, cfg.max_doc_tokens)
                doc_token_count = sum(len(segment) for segment in segments)

                split = "val" if should_send_to_val(subset, shard.name, line_idx, text, cfg.val_fraction) else "train"
                if split == "val" and stats.val_tokens + doc_token_count > cfg.max_val_tokens_per_subset:
                    split = "train"

                writer = val_writer if split == "val" else train_writer
                for segment in segments:
                    writer.write(segment)

                stats.actual_tokens += doc_token_count
                stats.docs_written += 1
                if split == "val":
                    stats.val_tokens += doc_token_count
                else:
                    stats.train_tokens += doc_token_count

                if stats.docs_seen % 100 == 0:
                    bar.set_postfix(
                        actual=stats.actual_tokens,
                        target=target_tokens,
                        train=stats.train_tokens,
                        val=stats.val_tokens,
                        shard=writer.current_shard,
                    )
                if cfg.checkpoint_interval_docs > 0 and stats.docs_seen % cfg.checkpoint_interval_docs == 0:
                    write_progress(
                        cfg,
                        target_payload,
                        tokenizer,
                        dtype,
                        max_id,
                        train_writer,
                        val_writer,
                        subset_stats,
                        subset,
                        {"subset": subset, "shard": shard.name, "next_line_idx": line_idx + 1},
                    )
                bar.update(1)

            if cfg.limit_docs_per_subset is not None and stats.docs_seen >= cfg.limit_docs_per_subset:
                break

    stats.completed = True
    write_progress(cfg, target_payload, tokenizer, dtype, max_id, train_writer, val_writer, subset_stats, subset, None)


def main() -> None:
    cfg = config_from_args(parse_args())
    validate_config(cfg)
    prepare_output_root(cfg)
    if cfg.resume:
        validate_resume_config(cfg)

    target_payload = load_target_config(cfg.target_config)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path, use_fast=True)
    bos_id, eos_id, _pad_id = require_special_token_ids(tokenizer)
    max_id = max_token_id(tokenizer)
    dtype = resolve_dtype(cfg.dtype, max_id)

    if not cfg.resume:
        write_json_atomic(cfg.output_root / ("dry_run_config_used.json" if cfg.dry_run else "config_used.json"), jsonable_config(cfg))

    train_writer = TokenShardWriter(
        cfg.output_root / "train",
        "train",
        dtype,
        cfg.train_shard_tokens,
        flush_batch_tokens=cfg.flush_batch_tokens,
        dry_run=cfg.dry_run,
    )
    val_writer = TokenShardWriter(
        cfg.output_root / "val",
        "val",
        dtype,
        cfg.val_shard_tokens,
        flush_batch_tokens=cfg.flush_batch_tokens,
        dry_run=cfg.dry_run,
    )
    subset_stats: dict[str, SubsetStats] = {}
    resume_state: ResumeState | None = None
    if cfg.resume:
        resume_state = load_resume_state(cfg, train_writer, val_writer)
        subset_stats = resume_state.subset_stats
        print(f"[resume] loaded checkpoint from {cfg.output_root / 'progress.json'}")

    for subset, target in target_payload["subsets"].items():
        resume_cursor = None
        if resume_state is not None and resume_state.current_subset == subset:
            resume_cursor = resume_state.current_cursor
        encode_subset(
            subset=subset,
            target_tokens=int(target),
            cfg=cfg,
            tokenizer=tokenizer,
            bos_id=bos_id,
            eos_id=eos_id,
            train_writer=train_writer,
            val_writer=val_writer,
            target_payload=target_payload,
            dtype=dtype,
            max_id=max_id,
            subset_stats=subset_stats,
            resume_cursor=resume_cursor,
        )

    train_writer.close()
    val_writer.close()

    manifest = make_manifest(
        cfg,
        target_payload,
        tokenizer,
        dtype,
        max_id,
        train_writer,
        val_writer,
        subset_stats,
    )
    manifest_name = "dry_run_manifest.json" if cfg.dry_run else "manifest.json"
    write_json_atomic(cfg.output_root / manifest_name, manifest)
    write_json_atomic(cfg.output_root / "progress.json", {**manifest, "updated_at_unix": time.time(), "current_subset": None})

    print()
    print(f"train tokens: {train_writer.total_tokens:,}")
    print(f"val tokens: {val_writer.total_tokens:,}")
    print(f"total tokens: {train_writer.total_tokens + val_writer.total_tokens:,}")
    print(f"train shards: {len(train_writer.shards):,}")
    print(f"val shards: {len(val_writer.shards):,}")
    print(f"dtype: {dtype.name}")
    print(f"output path: {cfg.output_root}")


if __name__ == "__main__":
    main()
