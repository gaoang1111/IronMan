import os
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

def prepare_wikitext(output_dir: str = "data", min_length: int = 100):
    """
    下载 Wikitext-103，清洗并保存为 JSONL。
    """
    print(f"Downloading Wikitext-103...")
    # split="train[:20000]" 表示我们只取前 2万条作为 Phase 1 的热身数据
    # 对于 Phase 1，这足够了。如果要全量，去掉 [:20000]
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train[:20000]")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / "phase1_train.jsonl"
    
    print(f"Processing and saving to {file_path}...")
    
    count = 0
    with open(file_path, "w", encoding="utf-8") as f:
        for item in tqdm(dataset):
            text = item["text"].strip()
            
            # 1. 过滤掉空行和太短的标题
            if len(text) < min_length:
                continue
            
            # 2. 过滤掉无意义的表头 (Wikitext 特有)
            if text.startswith(" =") and text.endswith("= "):
                continue
                
            # 3. 写入文件
            json_line = json.dumps({"text": text}, ensure_ascii=False)
            f.write(json_line + "\n")
            count += 1
            
    print(f"Done! Saved {count} valid samples to {file_path}")

if __name__ == "__main__":
    prepare_wikitext()