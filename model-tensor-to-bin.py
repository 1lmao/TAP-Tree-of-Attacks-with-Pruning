#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gc
import json
import torch
import psutil
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights

# ------------------- CONFIG -------------------
INPUT_DIR  = "/home/server/Desktop/TAP/models/gpt-oss-20b"
OUTPUT_DIR = "/home/server/Desktop/TAP/models/gpt-oss-20b-bin"
SHARD_SIZE = "5GB"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
# ------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

def mem():
    ram = psutil.virtual_memory()
    vram = torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else 0
    print(f"  RAM {ram.percent:5.1f}% ({ram.used/1e9:.2f} GB) | VRAM {vram:.2f} GB")

print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    torch.cuda.empty_cache()

# ------------------------------------------------
print("\n1. Load config")
config = AutoConfig.from_pretrained(INPUT_DIR, trust_remote_code=True)

# ------------------------------------------------
print("\n2. Initialise **empty meta model** (0 RAM, 0 VRAM)")
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

# DO NOT call .to() or .half() here!
model.eval()
print("   Model created with meta tensors")

# ------------------------------------------------
print("\n3. Load weight-map")
index_path = os.path.join(INPUT_DIR, "model.safetensors.index.json")
with open(index_path) as f:
    index = json.load(f)
weight_map = index["weight_map"]
print(f"   → {len(weight_map)} tensors to load")

# ------------------------------------------------
print("\n4. Stream shards → assign directly to meta model (in-place)")
torch.set_grad_enabled(False)

for i, (name, file) in enumerate(weight_map.items(), 1):
    path = os.path.join(INPUT_DIR, file)
    print(f"   [{i:3d}/{len(weight_map)}] {file} → {name}", end="")

    with safe_open(path, framework="pt", device=DEVICE) as f:
        tensor = f.get_tensor(name).to(torch.float16)  # FP16 + on GPU

    # Build state_dict with **exact name**
    state_dict = {name: tensor}

    # Assign **in-place** — replaces meta tensor, no copy!
    model.load_state_dict(state_dict, strict=False, assign=True)

    del state_dict, tensor
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    if i % 20 == 0:
        print()
        mem()
    else:
        print(".", end="", flush=True)

print("\nAll weights loaded into model!")
mem()


# ------------------------------------------------
print("\n5. Move to CPU & save sharded .bin")
model = model.cpu()  # Now it's real FP16 → FP32 temp (~14 GB RAM)
torch.cuda.empty_cache()

model.save_pretrained(
    OUTPUT_DIR,
    safe_serialization=False,
    max_shard_size=SHARD_SIZE
)

print("\n6. Save tokenizer")
tokenizer = AutoTokenizer.from_pretrained(INPUT_DIR, trust_remote_code=True)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nSUCCESS! Files:")
os.system(f"ls -lh {OUTPUT_DIR}/pytorch_model*.bin")
