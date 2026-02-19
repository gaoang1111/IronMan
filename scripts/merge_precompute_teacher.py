from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _load_shard(path: str) -> dict:
    data = torch.load(path, map_location="cpu")
    required = {"targets", "valid_lens", "chunk_size", "max_seq_len", "layers"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Missing keys in shard {path}: {sorted(missing)}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--shard_paths", type=str, nargs="+", required=True)
    args = parser.parse_args()

    shards = []
    meta = None
    for shard_path in args.shard_paths:
        data = _load_shard(shard_path)
        if meta is None:
            meta = {
                "chunk_size": data["chunk_size"],
                "max_seq_len": data["max_seq_len"],
                "layers": data["layers"],
            }
        else:
            if data["chunk_size"] != meta["chunk_size"]:
                raise ValueError(f"chunk_size mismatch in {shard_path}")
            if data["max_seq_len"] != meta["max_seq_len"]:
                raise ValueError(f"max_seq_len mismatch in {shard_path}")
            if data["layers"] != meta["layers"]:
                raise ValueError(f"layers mismatch in {shard_path}")
        shards.append(data)

    targets = torch.cat([s["targets"] for s in shards], dim=0)
    valid_lens = torch.cat([s["valid_lens"] for s in shards], dim=0)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "targets": targets,
            "valid_lens": valid_lens,
            "chunk_size": meta["chunk_size"],
            "max_seq_len": meta["max_seq_len"],
            "layers": meta["layers"],
        },
        out_path,
    )


if __name__ == "__main__":
    main()
