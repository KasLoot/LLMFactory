from transformers import AutoTokenizer

qwen_tok = AutoTokenizer.from_pretrained("tokenizer/qwen3_6", use_fast=True)
gemma_tok = AutoTokenizer.from_pretrained("tokenizer/gemma4", use_fast=True)
lfm2_5_tok = AutoTokenizer.from_pretrained("tokenizer/LFM2_5", use_fast=True)
lfm2_5_vl_tok = AutoTokenizer.from_pretrained("tokenizer/LFM2_5_VL", use_fast=True)

samples = [
    "<|startoftext|>Artificial Intelligence (AI) is a branch of computer science focused on building systems capable of performing tasks that typically require human intelligence.<|im_end|><|startoftext|>These systems can learn from data, recognize patterns, solve problems, make decisions, and understand or generate natural language.<|im_end|>",
]

for name, tok in [("lfm 2.5 VL", lfm2_5_vl_tok)]:
    print("=" * 40)
    print(name, "vocab:", len(tok))

    for s in samples:
        ids = tok.encode(s, add_special_tokens=False)
        decoded = tok.decode(ids, skip_special_tokens=False)

        print(repr(s))
        print("tokens:", len(ids))
        print(f"encoded: {ids}")
        print("chars/token:", len(s) / max(len(ids), 1))
        print("roundtrip:", decoded == s)
        print()