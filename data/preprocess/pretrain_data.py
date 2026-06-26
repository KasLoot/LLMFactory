"""Estimate token counts and sampled document-length stats for Dolma3 subsets.

Compared with the original script, this version:
- samples shards across the whole sub-dataset instead of only the first sorted shards;
- optionally counts BOS/EOS document-boundary tokens;
- avoids hard-coded special-token ids;
- writes a compact JSON + text report.

It is still an estimator. The final authoritative token count should come from the
actual pre-tokenized training shards.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import zstandard as zstd
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase


CATEGORIES: dict[str, list[str]] = {
    "wiki and prose": [
        "dolma1_7-wiki-en",
    ],
    "math and science": [
        "finemath-3plus",
        "common_crawl-science_math_and_technology-0012",
        "common_crawl-science_math_and_technology-0013",
    ],
    "software and programming": [
        "common_crawl-software-0011",
        "common_crawl-software-0012",
        "common_crawl-software_development-0012",
        "common_crawl-software_development-0013",
        "stack_edu-Python",
        "stack_edu-Markdown",
        "stack_edu-Shell",
        "stack_edu-SQL",
    ],
    "industrial and engineering": [
        "common_crawl-electronics_and_hardware-0012",
        "common_crawl-electronics_and_hardware-0013",
        "common_crawl-industrial-0015",
        "common_crawl-industrial-0016",
        "common_crawl-transportation-0016",
        "common_crawl-transportation-0017",
    ],
    "casual and lifestyle": [
        "common_crawl-home_and_hobbies-0017",
        "common_crawl-home_and_hobbies-0018",
        "common_crawl-home_and_hobbies-0019",
        "common_crawl-food_and_dining-0015",
        "common_crawl-food_and_dining-0016",
        "common_crawl-travel_and_tourism-0017",
        "common_crawl-travel_and_tourism-0018",
        "common_crawl-travel_and_tourism-0019",
    ],
}


@dataclass
class SubsetEstimate:
    mode: str
    sample_docs: int
    sample_tokens: int
    sample_compressed_bytes: int
    total_compressed_bytes: int
    tokens_per_compressed_byte: float
    tokens: int
    n_shards: int
    n_sampled_shards: int
    sampled_shard_names: list[str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/data/datasets/dolma3_mix-6T/data")
    p.add_argument("--tokenizer", default="tokenizer/LFM2_5_VL")
    p.add_argument("--out-dir", default="stats")
    p.add_argument("--sample-docs", type=int, default=12_000)
    p.add_argument("--max-sampled-shards", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--context-length", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--count-bos-eos", action="store_true")
    return p.parse_args()


def list_shards(subset_dir: Path) -> list[Path]:
    return sorted(subset_dir.glob("*.jsonl.zst"))


def stable_seed(base_seed: int, name: str) -> int:
    digest = hashlib.sha1(name.encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:4], "big")


def choose_shards(shards: list[Path], subset: str, seed: int, max_shards: int) -> list[Path]:
    if len(shards) <= max_shards:
        return shards
    rng = random.Random(stable_seed(seed, subset))
    return sorted(rng.sample(shards, max_shards))


def iter_jsonl_zst_texts(path: Path, max_docs: int | None = None) -> tuple[list[str], int]:
    """Return up to max_docs non-empty text fields and compressed bytes consumed."""
    texts: list[str] = []
    with path.open("rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh, closefd=False) as reader:
            stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            for line in stream:
                try:
                    text = json.loads(line).get("text", "")
                except json.JSONDecodeError:
                    continue
                if not text or not text.strip():
                    continue
                texts.append(text)
                if max_docs is not None and len(texts) >= max_docs:
                    break
        compressed_bytes = fh.tell()
    return texts, compressed_bytes


def batched(xs: list[str], batch_size: int) -> Iterable[list[str]]:
    for i in range(0, len(xs), batch_size):
        yield xs[i : i + batch_size]


def token_lengths(
    texts: list[str], tokenizer: PreTrainedTokenizerBase, batch_size: int, special_tokens_per_doc: int
) -> list[int]:
    lengths: list[int] = []
    for batch in batched(texts, batch_size):
        enc = tokenizer(batch, add_special_tokens=False, truncation=False)
        lengths.extend(len(ids) + special_tokens_per_doc for ids in enc["input_ids"])
    return lengths


def estimate_subset(
    subset: str,
    subset_dir: Path,
    tokenizer: PreTrainedTokenizerBase,
    args: argparse.Namespace,
    special_tokens_per_doc: int,
) -> tuple[int, SubsetEstimate, list[int]]:
    shards = list_shards(subset_dir)
    total_bytes = sum(p.stat().st_size for p in shards)
    sampled_shards = choose_shards(shards, subset, args.seed, args.max_sampled_shards)
    docs_per_shard = max(1, math.ceil(args.sample_docs / len(sampled_shards)))

    sample_docs = 0
    sample_tokens = 0
    sample_bytes = 0
    sampled_lengths: list[int] = []

    with tqdm(total=args.sample_docs, desc=f"sample {subset}", leave=False) as bar:
        for shard in sampled_shards:
            remaining = args.sample_docs - sample_docs
            if remaining <= 0:
                break
            texts, used_bytes = iter_jsonl_zst_texts(shard, max_docs=min(docs_per_shard, remaining))
            lens = token_lengths(texts, tokenizer, args.batch_size, special_tokens_per_doc)

            sample_docs += len(lens)
            sample_tokens += int(sum(lens))
            sample_bytes += int(used_bytes)
            sampled_lengths.extend(lens)
            bar.update(len(lens))

    tokens_per_byte = sample_tokens / sample_bytes if sample_bytes else 0.0
    est_tokens = int(round(tokens_per_byte * total_bytes))
    info = SubsetEstimate(
        mode="estimate",
        sample_docs=sample_docs,
        sample_tokens=sample_tokens,
        sample_compressed_bytes=sample_bytes,
        total_compressed_bytes=total_bytes,
        tokens_per_compressed_byte=tokens_per_byte,
        tokens=est_tokens,
        n_shards=len(shards),
        n_sampled_shards=len(sampled_shards),
        sampled_shard_names=[p.name for p in sampled_shards],
    )
    return est_tokens, info, sampled_lengths


def summarize_lengths(lengths: list[int], context_length: int) -> dict:
    arr = np.asarray(lengths, dtype=np.int64)
    percentiles = [50, 75, 90, 95, 99, 99.9]
    pct_values = np.percentile(arr, percentiles)
    over = int(np.sum(arr > context_length))
    return {
        "count": int(arr.size),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "percentiles": {f"p{p}": float(v) for p, v in zip(percentiles, pct_values)},
        "n_over_context": over,
        "frac_over_context": over / arr.size,
        "context_length": context_length,
        "tokens_total_in_sample": int(arr.sum()),
    }


def human(n: float) -> str:
    for unit in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}"
        n /= 1000
    return f"{n:.1f}P"


def make_report(per_subset: dict[str, int], per_category: dict[str, int], total: int) -> str:
    lines: list[str] = []
    for category, subsets in CATEGORIES.items():
        lines.append(f"{category}:\n")
        for subset in subsets:
            lines.append(f"  data/{subset}: {per_subset.get(subset, 0):,} tokens\n")
        lines.append(f"  -> subtotal: {per_category[category]:,} tokens ({human(per_category[category])})\n\n")
    lines.append(f"TOTAL: {total:,} tokens ({human(total)})\n")
    return "".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    special_tokens_per_doc = 2 if args.count_bos_eos else 0

    per_subset: dict[str, int] = {}
    per_category: dict[str, int] = {}
    details: dict[str, dict] = {}
    all_lengths: list[int] = []

    for category, subsets in CATEGORIES.items():
        cat_total = 0
        for subset in subsets:
            subset_dir = Path(args.data_root) / subset
            if not subset_dir.is_dir():
                print(f"[WARN] missing subset: {subset_dir}")
                per_subset[subset] = 0
                continue

            tokens, info, lengths = estimate_subset(
                subset, subset_dir, tokenizer, args, special_tokens_per_doc
            )
            per_subset[subset] = tokens
            details[subset] = asdict(info)
            all_lengths.extend(lengths)
            cat_total += tokens
        per_category[category] = cat_total

    total = sum(per_category.values())
    length_stats = summarize_lengths(all_lengths, args.context_length) if all_lengths else None
    report = make_report(per_subset, per_category, total)

    print(report)
    if length_stats:
        print(json.dumps(length_stats, indent=2))

    payload = {
        "mode": "estimate",
        "tokenizer": args.tokenizer,
        "count_bos_eos": args.count_bos_eos,
        "sample_docs_per_subset": args.sample_docs,
        "max_sampled_shards": args.max_sampled_shards,
        "per_subset": per_subset,
        "per_category": per_category,
        "total": total,
        "details": details,
        "length_stats": length_stats,
    }
    (out_dir / "token_count_report.txt").write_text(report, encoding="utf-8")
    (out_dir / "token_counts.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'token_count_report.txt'} and {out_dir / 'token_counts.json'}")


if __name__ == "__main__":
    main()
