"""Examples for the exported LFM 2.5 VL tiktoken tokenizer.

This folder is a self-contained, portable tiktoken bundle:
    lfm2_5_vl.tiktoken    base64 mergeable BPE ranks  (63,893 entries)
    tokenizer_meta.json   name, split regex, 507 special tokens, n_vocab

Loading needs only `tiktoken` (no transformers / tokenizers / tokenizer.json).

Run:
    python tokenizer/LFM2_5_VL_tiktoken/example.py
"""

import base64
import json
import os

import tiktoken

HERE = os.path.dirname(os.path.abspath(__file__))


def load_tokenizer(export_dir: str = HERE) -> tiktoken.Encoding:
    """Rebuild the tiktoken.Encoding from the files in this folder."""
    with open(os.path.join(export_dir, "tokenizer_meta.json")) as f:
        meta = json.load(f)

    mergeable_ranks = {}
    with open(os.path.join(export_dir, meta["bpe_file"]), "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            token_b64, rank = line.split()
            mergeable_ranks[base64.b64decode(token_b64)] = int(rank)

    return tiktoken.Encoding(
        name=meta["name"],
        pat_str=meta["pat_str"],
        mergeable_ranks=mergeable_ranks,
        special_tokens=meta["special_tokens"],
        explicit_n_vocab=meta["n_vocab"],
    )


def main() -> None:
    enc = load_tokenizer()
    print(f"loaded '{enc.name}' | n_vocab={enc.n_vocab} | {len(enc.special_tokens_set)} special tokens\n")

    text = "Artificial Intelligence is a branch of computer science. héllo 你好 🚀"

    # 1) Plain text encode/decode. Use encode_ordinary for normal data: it never
    #    interprets <|...|> markers, so untrusted input can never inject specials.
    ids = enc.encode_ordinary(text)
    print("encode_ordinary ->", len(ids), "tokens")
    print("first 12 ids     ->", ids[:12])
    print("decode roundtrip ->", enc.decode(ids) == text, "\n")

    # 2) Encode text that contains special-token markers.
    #    By default encode() RAISES on special markers (a safety guard); you must
    #    opt in. allowed_special="all" matches HF's add_special_tokens=False output.
    framed = "<|startoftext|>Hello world<|im_end|>"
    ids_special = enc.encode(framed, allowed_special="all")
    print("encode(allowed_special='all') ->", ids_special)
    print("  '<|startoftext|>' is one token:", enc.encode_single_token("<|startoftext|>"))
    print("  '<|im_end|>'      is one token:", enc.encode_single_token("<|im_end|>"))

    #    Same string via encode_ordinary: the markers become regular byte tokens.
    print("encode_ordinary(framed)       ->", enc.encode_ordinary(framed), "\n")

    # 3) A few useful special token ids.
    for tok in ["<|pad|>", "<|startoftext|>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"]:
        print(f"  {tok:<18} id = {enc.encode_single_token(tok)}")
    print()

    # 4) Fast batch encoding (releases the GIL, multithreaded under the hood).
    docs = ["first document", "second, slightly longer document", "third 🚀"]
    batch = enc.encode_ordinary_batch(docs, num_threads=4)
    print("batch token counts ->", [len(x) for x in batch])

    # 5) Token counting (e.g. for packing / truncation in a data pipeline).
    print("token count        ->", len(enc.encode_ordinary(text)))

    # 6) Decoding a single token back to bytes (handy for debugging).
    print("token 0 bytes      ->", enc.decode_single_token_bytes(ids[0]))


if __name__ == "__main__":
    main()
