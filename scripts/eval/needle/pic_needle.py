import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_needle_results(csv_path: str, out_img_path: str):
    # 1. 检查文件是否存在
    if not Path(csv_path).exists():
        print(f"找不到文件: {csv_path}。请确认你的测试已经跑完并生成了该文件。")
        return

    # 2. 读取数据
    df = pd.read_csv(csv_path)
    
    # 将 position_idx 转换成百分比深度 (0 到 100%)
    # 假设你的 num_positions 是 20
    num_positions = len(df)
    df['Depth (%)'] = (df['position_idx'] / num_positions) * 100
    
    # 3. 设置画图风格
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    
    # 4. 绘制折线图，并填充下方区域
    sns.lineplot(
        data=df, 
        x='Depth (%)', 
        y='hit_rate', 
        marker='o',      # 打上圆点
        markersize=8,
        color='#2ca02c', # 经典科技绿
        linewidth=2.5
    )
    plt.fill_between(df['Depth (%)'], df['hit_rate'], color='#2ca02c', alpha=0.15)
    
    # 5. 美化图表
    plt.title('Needle In A Haystack: Retrieval Accuracy by Depth', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Document Depth (%)', fontsize=12)
    plt.ylabel('Hit Rate (Accuracy)', fontsize=12)
    
    # 设定 X 轴和 Y 轴的范围
    plt.xlim(0, max(df['Depth (%)']))
    plt.ylim(-0.05, 1.05) # 上下稍微留白，避免 1.0 的点贴在最顶端
    
    # 将 Y 轴显示为百分比格式
    plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
    
    plt.tight_layout()
    
    # 6. 保存图片
    plt.savefig(out_img_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已成功保存到: {out_img_path}")
    
    # 如果你在有界面的系统里，可以取消下面这行的注释来直接弹窗显示
    # plt.show()

if __name__ == "__main__":
    # INPUT_CSV = "data/needle_scale/dynamicq/needle_scale_summary_by_position.csv"
    # OUTPUT_IMG = "data/needle_scale/dynamicq/niah_result_chart.png"

    # INPUT_CSV = "data/needle_scale/dynamicq-0/needle_scale_summary_by_position.csv"
    # OUTPUT_IMG = "data/needle_scale/dynamicq-0/niah_result_chart.png"
    
    INPUT_CSV = "data/needle_scale/base/needle_scale_summary_by_position.csv"
    OUTPUT_IMG = "data/needle_scale/base/niah_result_chart.png"
    
    plot_needle_results(INPUT_CSV, OUTPUT_IMG)