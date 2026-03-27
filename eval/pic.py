import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def plot_all_results(res_dir="data/eval_results"):
    res_path = Path(res_dir)
    # 查找目录下所有的 loss 文件
    files = list(res_path.glob("*_losses.json"))
    
    if not files:
        print(f"❌ 在 {res_dir} 没找到任何 JSON 文件！")
        return

    plt.figure(figsize=(14, 7))
    
    # 颜色配置
    colors = {"oracle": "gray", "amnesiac": "red", "amnesiac_buf2": "purple", "mark42": "blue", "mark42-buffer1": "green", "mark42-buffer2": "orange"}
    styles = {"oracle": "--", "amnesiac": ":", "amnesiac_buf2": ":", "mark42": "-", "mark42-buffer1": "-.", "mark42-buffer2": "-."}

    print("📊 统计结果如下：")
    print("-" * 50)
    print(f"{'模型模式':<15} | {'平均 Loss':<10} | {'PPL (越低越好)':<10}")
    print("-" * 50)

    for f in files:
        mode = f.name.replace("_losses.json", "")
        data = load_json(f)
        
        # 计算统计信息
        avg_loss = np.mean(data)
        ppl = np.exp(avg_loss) # $PPL = e^{Loss}$
        
        print(f"{mode:<15} | {avg_loss:.4f}    | {ppl:.4f}")

        # 绘制平滑曲线 (滑动窗口平均)
        window_size = 10
        if len(data) > window_size:
            smooth_data = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
            x_axis = np.arange(len(smooth_data))
            
            plt.plot(x_axis, smooth_data, 
                     label=f"{mode} (PPL: {ppl:.2f})", 
                     color=colors.get(mode, None),
                     linestyle=styles.get(mode, "-"),
                     linewidth=2 if mode == "mark42" else 1.5)

    plt.title("Long-Context Performance Comparison (Wikitext-103 8K)")
    plt.xlabel("Chunk Index (Sequence Progress)")
    plt.ylabel("Cross Entropy Loss (Smoothed)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_fig = res_path / "comparison_curve.png"
    plt.savefig(output_fig, dpi=300)
    print("-" * 50)
    print(f"✅ 对比图已保存至: {output_fig.absolute()}")

if __name__ == "__main__":
    plot_all_results()