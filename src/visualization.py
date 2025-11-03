import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class ResultVisualizer:
    def __init__(self):
        self.figures_dir = os.path.join("data", "figures")
        os.makedirs(self.figures_dir, exist_ok=True)
        self._set_chinese_font()

    def _set_chinese_font(self):
        """自动适配系统中文字体"""
        try:
            # Windows系统
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            # 验证字体
            test_fig, test_ax = plt.subplots(figsize=(1, 1))
            test_ax.text(0.5, 0.5, "测试中文显示", ha='center')
            plt.close(test_fig)
        except Exception:
            try:
                # macOS系统
                plt.rcParams['font.sans-serif'] = ['PingFang SC', 'DejaVu Sans']
                test_fig, test_ax = plt.subplots(figsize=(1, 1))
                test_ax.text(0.5, 0.5, "测试中文显示", ha='center')
                plt.close(test_fig)
            except Exception:
                # Linux系统
                plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
                print("提示：Linux系统若中文显示异常，请执行 'sudo apt install fonts-wqy-zenhei'")
        
        # 关闭字体警告
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, message="Glyph .* missing from font")

    def plot_preference_distribution(self, evaluation_df, model_a_name, model_b_name):
        """绘制偏好概率分布"""
        prob_col = f"prefer_{model_a_name}_prob"
        
        plt.figure(figsize=(10, 6))
        sns.histplot(evaluation_df[prob_col], bins=20, kde=True, color='#1f77b4')
        plt.axvline(50, color='r', linestyle='--', label='中立点（50%）')
        plt.title(f"{model_a_name} vs {model_b_name} 模型偏好分布", fontsize=14)
        plt.xlabel(f"偏好{model_a_name}的概率（%）", fontsize=12)
        plt.ylabel("题目数量（频率）", fontsize=12)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        save_path = os.path.join(self.figures_dir, f"{model_a_name}_vs_{model_b_name}_distribution.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path

    def plot_preference_summary(self, summary_data):
        """绘制汇总比较图"""
        labels = [f"{a} vs {b}" for (a, b) in summary_data.keys()]
        values = [v for v in summary_data.values()]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(labels, values, color=['#ff7f0e', '#2ca02c', '#d62728'])
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.axhline(50, color='r', linestyle='--', label='中立点（50%）')
        plt.title("模型偏好概率平均值对比", fontsize=14)
        plt.ylabel("偏好第一个模型的平均概率（%）", fontsize=12)
        plt.ylim(0, 100)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        save_path = os.path.join(self.figures_dir, "model_preference_summary.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path

    def generate_summary_stats(self, evaluation_df, model_a_name, model_b_name):
        """生成统计摘要"""
        prob_col = f"prefer_{model_a_name}_prob"
        return {
            "平均偏好概率": evaluation_df[prob_col].mean(),
            "偏好A的比例(>50%)": (evaluation_df[prob_col] > 50).mean() * 100,
            "偏好B的比例(<50%)": (evaluation_df[prob_col] < 50).mean() * 100,
            "中立比例(=50%)": (evaluation_df[prob_col] == 50).mean() * 100,
            "标准差": evaluation_df[prob_col].std()
        }