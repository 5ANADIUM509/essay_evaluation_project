import os
import pandas as pd
import time
from src.model_client import ModelClient
from src.evaluator import ResponseEvaluator
from src.visualization import ResultVisualizer

def load_prompts():
    """加载已有的作文题目文件，不生成新题目"""
    prompts_path = os.path.join("data", "prompts.csv")
    if os.path.exists(prompts_path) and os.path.getsize(prompts_path) > 0:
        print("加载作文题目...")
        prompts_df = pd.read_csv(prompts_path)
        # 确保包含必要的index列
        if "index" not in prompts_df.columns:
            prompts_df = prompts_df.reset_index()
            prompts_df = prompts_df.rename(columns={"index": "original_index"})
            prompts_df["index"] = prompts_df.index
        print(f"成功加载已生成的{len(prompts_df)}个作文题")
        return prompts_df
    else:
        raise FileNotFoundError("未找到作文题文件，请确保data/prompts.csv存在且不为空")

def auto_fix_invalid_scores(model_size, prompts_df, evaluator, max_retries=3):
    """自动检测并修复指定模型的无效评分"""
    scores_path = os.path.join("data", "evaluations", f"{model_size}_single_scores.csv")
    responses_path = os.path.join("data", "model_responses", f"{model_size}_responses.csv")
    
    # 加载现有数据
    responses_df = pd.read_csv(responses_path) if os.path.exists(responses_path) else pd.DataFrame(columns=["index", "prompt", "response"])
    scores_df = pd.read_csv(scores_path) if os.path.exists(scores_path) else pd.DataFrame(columns=["index", "prompt", "response", "final_score", "all_reasons"])
    
    client = ModelClient(model_size)
    retry_count = 0
    total_invalid = None
    
    while retry_count < max_retries and (total_invalid is None or total_invalid > 0):
        # 检测无效评分
        invalid_mask = scores_df["final_score"].isna() | scores_df["final_score"].isnull()
        total_invalid = invalid_mask.sum()
        
        if total_invalid == 0:
            print(f"{model_size}模型所有评分均有效，无需修复")
            break
            
        print(f"\n{model_size}模型第{retry_count+1}轮修复：检测到{total_invalid}条无效评分")
        invalid_indices = scores_df[invalid_mask]["index"].tolist()
        
        # 逐个修复无效记录
        fixed_count = 0
        for idx in invalid_indices:
            try:
                # 获取对应题目
                prompt_row = prompts_df[prompts_df["index"] == idx].iloc[0]
                prompt = prompt_row["prompt"]
                
                # 1. 修复模型回答（若为空）
                response_row = responses_df[responses_df["index"] == idx]
                if len(response_row) == 0 or pd.isna(response_row["response"].iloc[0]):
                    print(f"修复{model_size}模型题目{idx}的回答...")
                    new_response = client.generate_single_response(prompt)
                    if new_response:
                        # 更新回答数据
                        if len(response_row) == 0:
                            responses_df = pd.concat([responses_df, pd.DataFrame({
                                "index": [idx], "prompt": [prompt], "response": [new_response]
                            })], ignore_index=True)
                        else:
                            responses_df.loc[responses_df["index"] == idx, "response"] = new_response
                        responses_df.to_csv(responses_path, index=False, encoding="utf-8-sig")
                
                # 2. 获取最新回答
                current_response = responses_df[responses_df["index"] == idx]["response"].iloc[0]
                if not current_response or pd.isna(current_response):
                    print(f"题目{idx}回答仍为空，跳过评估")
                    continue
                
                # 3. 修复评分
                print(f"修复{model_size}模型题目{idx}的评分...")
                new_score, reason = evaluator.score_single_response(prompt, current_response, model_size)
                
                # 更新评分数据
                if len(scores_df[scores_df["index"] == idx]) == 0:
                    scores_df = pd.concat([scores_df, pd.DataFrame({
                        "index": [idx], "prompt": [prompt], "response": [current_response],
                        "final_score": [new_score], "all_reasons": [reason]
                    })], ignore_index=True)
                else:
                    scores_df.loc[scores_df["index"] == idx, "final_score"] = new_score
                    scores_df.loc[scores_df["index"] == idx, "all_reasons"] = reason
                
                scores_df.to_csv(scores_path, index=False, encoding="utf-8-sig")
                fixed_count += 1 if new_score is not None else 0
                
            except Exception as e:
                print(f"修复题目{idx}失败：{str(e)}")
                continue
        
        print(f"第{retry_count+1}轮修复完成：成功修复{fixed_count}/{total_invalid}条")
        retry_count += 1
        # 每轮修复后等待10秒，避免API请求过于密集
        time.sleep(10)
    
    # 最终验证
    final_scores = pd.read_csv(scores_path)
    final_invalid = final_scores["final_score"].isna().sum()
    print(f"\n{model_size}模型修复结束，剩余无效评分：{final_invalid}")
    return final_scores

def main():
    # 1. 加载作文题目（仅加载已存在的文件，不生成新题目）
    prompts_df = load_prompts()
    print(f"已加载{len(prompts_df)}个作文题目")

    # 2. 初始化客户端和评估器
    model_sizes = ["7B", "14B", "70B"]
    model_responses = {}
    evaluator = ResponseEvaluator()

    # 3. 处理模型回答
    for size in model_sizes:
        print(f"\n处理{size}模型的回答...")
        client = ModelClient(size)
        responses_df = client.process_prompts(prompts_df)
        model_responses[size] = responses_df
        # 检查回答完整性
        missing_responses = responses_df["response"].isna().sum()
        if missing_responses > 0:
            print(f"警告：{size}模型存在{missing_responses}条缺失回答，将在评分阶段修复")

    # 4. 评估单模型并自动修复无效评分
    for size in model_sizes:
        print(f"\n=== 开始{size}模型评估与自动修复 ===")
        # 先进行常规评估
        evaluator.evaluate_single_model(model_responses[size], size)
        # 自动修复无效评分
        auto_fix_invalid_scores(size, prompts_df, evaluator, max_retries=3)
        # 输出最终有效数量
        final_scores = pd.read_csv(os.path.join("data", "evaluations", f"{size}_single_scores.csv"))
        valid_count = final_scores["final_score"].notna().sum()
        print(f"{size}模型最终有效评分数量：{valid_count}/{len(final_scores)}")

    # 5. 两模型对比评估
    print("\n=== 开始两模型对比评估 ===")
    pairs = [("7B", "14B"), ("7B", "70B"), ("14B", "70B")]
    for a, b in pairs:
        print(f"\n评估{a} vs {b}...")
        # 强制重新生成对比结果，确保列名正确
        evaluator.evaluate_pairwise(model_responses[a], model_responses[b], a, b, force_regenerate=False)
        stats = evaluator.get_comparison_stats(a, b)
        print(f"{a} vs {b} 对比统计：")
        print(f"  有效对比组数：{stats['有效对比组数']}")
        print(f"  {a}获胜次数：{stats[f'{a}获胜次数']}（{stats[f'{a}胜率']:.2f}%）")
        print(f"  {b}获胜次数：{stats[f'{b}获胜次数']}（{stats[f'{b}胜率']:.2f}%）")
        print(f"  平均分差：{stats['平均分差']}（正值表示{a}更优）")
        print(f"{a} vs {b}评估完成，共{len(model_responses[a])}对有效对比")

    # 6. 生成可视化结果
    print("\n=== 生成评估可视化结果 ===")
    visualizer = ResultVisualizer()
    summary_data = {}
    for a, b in pairs:
        comparison_df = pd.read_csv(os.path.join("data", "evaluations", f"{a}_vs_{b}_comparison.csv"))
        visualizer.plot_preference_distribution(comparison_df, a, b)
        stats = evaluator.get_comparison_stats(a, b)
        summary_data[(a, b)] = stats[f"偏好{a}的概率"]
        print(f"\n{a} vs {b} 评估统计:")
        print(f"  有效对比组数：{stats['有效对比组数']}")
        print(f"  {a}获胜次数/胜率：{stats[f'{a}获胜次数']}/{stats[f'{a}胜率']:.2f}%")
        print(f"  {b}获胜次数/胜率：{stats[f'{b}获胜次数']}/{stats[f'{b}胜率']:.2f}%")
        print(f"  平均分差（{a}-{b}）：{stats['平均分差']}")
        print(f"  偏好{a}的概率：{stats[f'偏好{a}的概率']:.2f}%")
    visualizer.plot_preference_summary(summary_data)

    print("\n所有评估完成，结果已保存至以下路径：")
    print("  - 单模型评分：data/evaluations/7B_single_scores.csv, 14B_single_scores.csv, 70B_single_scores.csv")
    print("  - 两模型对比结果：data/evaluations/（包含comparison.csv文件）")
    print("  - 可视化图表：data/figures/")

if __name__ == "__main__":
    main()