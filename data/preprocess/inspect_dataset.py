import json
from pathlib import Path

import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset_dir = Path("/home/yuxin/workspace/data/datasets/llmfactory_pretrain_v0")
manifest = json.loads((dataset_dir / "manifest.json").read_text())

split = "train"
shard = manifest["splits"][split]["shards"][0]

path = dataset_dir / shard["filename"]
dtype = np.dtype(shard["dtype"])
token_count = int(shard["tokens"])

data_shard = np.memmap(path, dtype=dtype, mode="r", shape=(token_count,))

print(f"path: {path}")
print(f"dtype: {data_shard.dtype}")
print(f"shape: {data_shard.shape}")
print(f"first 32 tokens: {data_shard[:32].tolist()}")

# If you need a torch tensor for a slice:
tokens = torch.from_numpy(np.asarray(data_shard[:2048], dtype=np.int64)).to(device)
print(tokens.shape, tokens.dtype, tokens.device)