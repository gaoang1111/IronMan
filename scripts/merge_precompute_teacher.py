from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _load_shard(path: str) -> dict:
    data = torch.load(path, map_location="cpu")
    # 【核心修改】对齐最新的 distill_hidden.py 保存的字典键值
    required = {"targets", "valid_v_lens", "chunk_size", "target_layer", "q_num"}
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
            # 【核心修改】记录新的全局元数据
            meta = {
                "chunk_size": data["chunk_size"],
                "target_layer": data["target_layer"],
                "q_num": data["q_num"],
            }
        else:
            # 【核心修改】严格校验各碎片的元数据是否一致
            if data["chunk_size"] != meta["chunk_size"]:
                raise ValueError(f"chunk_size mismatch in {shard_path}")
            if data["target_layer"] != meta["target_layer"]:
                raise ValueError(f"target_layer mismatch in {shard_path}")
            if data["q_num"] != meta["q_num"]:
                raise ValueError(f"q_num mismatch in {shard_path}")
        shards.append(data)

    # 【核心修改】拼接 targets 和新的 valid_v_lens
    targets = torch.cat([s["targets"] for s in shards], dim=0)
    valid_v_lens = torch.cat([s["valid_v_lens"] for s in shards], dim=0)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存最终合并的大文件
    torch.save(
        {
            "targets": targets,
            "valid_v_lens": valid_v_lens,
            "chunk_size": meta["chunk_size"],
            "target_layer": meta["target_layer"],
            "q_num": meta["q_num"],
        },
        out_path,
    )
    print(f"Successfully merged {len(shards)} shards.")
    print(f"Final targets shape: {targets.shape}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()