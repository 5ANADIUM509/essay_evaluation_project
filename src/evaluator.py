import os
import pandas as pd
import time
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests.exceptions
from config.api_keys import DEEPSEEK_API_KEY

class ResponseEvaluator:
    def __init__(self, judge_model="deepseek-chat"):
        self.judge_model = judge_model
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"
        )
        self.evaluation_dir = os.path.join("data", "evaluations")
        os.makedirs(self.evaluation_dir, exist_ok=True)
        
        # 评估提示词
        self.system_prompt = """你是一位专业的作文评分专家，需要对给定的作文进行评分。
评分标准（1-10分）：
1. 内容相关性（30%）：是否紧扣题目要求，主题明确
2. 结构合理性（25%）：逻辑清晰，层次分明，有开头、主体和结尾
3. 语言表达（30%）：用词准确，语句通顺，有文采
4. 创新性（15%）：观点新颖，表达方式独特

请先给出整体评价（50字以内），然后给出具体分数（仅输出数字，如"8.5"），格式如下：
评价：[你的评价]
分数：[1-10的数字]
"""
        
        self.pairwise_prompt = """你是一位专业的作文评分专家，需要对比两篇同题作文的优劣。
对比维度：
1. 内容相关性：是否紧扣题目
2. 结构合理性：逻辑与层次
3. 语言表达：准确性与流畅度
4. 创新性：观点与表达

请先说明哪篇更优及原因（50字以内），然后给出偏好概率（0-100%，表示更偏好A的概率），格式如下：
评价：[你的评价]
偏好A的概率：[0-100的数字]%
"""

    def _extract_score(self, judgment):
        """从评估结果中提取分数"""
        try:
            if "分数：" in judgment:
                score_str = judgment.split("分数：")[-1].strip().split("\n")[0]
                return float(score_str)
            return None
        except Exception as e:
            print(f"提取分数失败: {str(e)}，评估结果：{judgment[:100]}")
            return None

    def _extract_preference(self, judgment):
        """从对比结果中提取偏好概率"""
        try:
            if "偏好A的概率：" in judgment:
                prob_str = judgment.split("偏好A的概率：")[-1].strip().replace("%", "")
                return float(prob_str)
            return None
        except Exception as e:
            print(f"提取偏好概率失败: {str(e)}，评估结果：{judgment[:100]}")
            return None

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=5, max=60),
        retry=retry_if_exception_type((requests.exceptions.RequestException, Exception))
    )
    def score_single_response(self, prompt, model_response, model_name):
        """单独评估一条模型回答（用于定向修复）"""
        if not model_response or pd.isna(model_response):
            return None, "模型回答为空，无法评估"
        
        try:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"题目：{prompt}\n回答：{model_response}"}
                ],
                temperature=0.1,
                timeout=120.0
            )
            judgment = response.choices[0].message.content.strip()
            score = self._extract_score(judgment)
            if score is None:
                return None, f"无法从评估结果中提取分数：{judgment[:100]}"
            return score, f"评估成功：{judgment[:50]}"
        except Exception as e:
            return None, f"评估失败：{str(e)}"

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=5, max=60),
        retry=retry_if_exception_type((requests.exceptions.RequestException, Exception))
    )
    def _evaluate_single(self, prompt, response):
        """内部单条评估方法"""
        response = self.client.chat.completions.create(
            model=self.judge_model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"题目：{prompt}\n回答：{response}"}
            ],
            temperature=0.1,
            timeout=120.0
        )
        return response.choices[0].message.content.strip()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=5, max=60),
        retry=retry_if_exception_type((requests.exceptions.RequestException, Exception))
    )
    def _evaluate_pair(self, prompt, response_a, response_b):
        """内部对比评估方法"""
        response = self.client.chat.completions.create(
            model=self.judge_model,
            messages=[
                {"role": "system", "content": self.pairwise_prompt},
                {"role": "user", "content": f"题目：{prompt}\nA的回答：{response_a}\nB的回答：{response_b}"}
            ],
            temperature=0.1,
            timeout=120.0
        )
        return response.choices[0].message.content.strip()

    def evaluate_single_model(self, responses_df, model_name, force_regenerate=False):
        """评估单个模型的所有回答"""
        output_file = os.path.join(self.evaluation_dir, f"{model_name}_single_scores.csv")
        
        if not force_regenerate and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            try:
                existing_df = pd.read_csv(output_file)
                if len(existing_df) == len(responses_df) and "final_score" in existing_df.columns:
                    print(f"检测到{model_name}已有完整评分，直接加载（{len(existing_df)}条）")
                    return existing_df
                print(f"{model_name}评分不完整，补充评估...")
            except Exception as e:
                print(f"{model_name}评分文件损坏: {str(e)}，重新评估...")
        
        existing_df = pd.DataFrame(columns=["index", "prompt", "response", "final_score", "all_reasons"]) if not os.path.exists(output_file) else pd.read_csv(output_file)
        processed_indices = set(existing_df["index"]) if not existing_df.empty else set()
        results = []
        total = len(responses_df)
        unprocessed = [idx for idx in responses_df["index"] if idx not in processed_indices]
        total_unprocessed = len(unprocessed)
        
        if total_unprocessed == 0:
            print(f"{model_name}所有回答已完成评分")
            return existing_df
            
        print(f"正在评估{model_name}模型...")
        print(f"{model_name}需要评估{total_unprocessed}个回答")
        
        for i, idx in enumerate(unprocessed, 1):
            row = responses_df[responses_df["index"] == idx].iloc[0]
            prompt = row["prompt"]
            response = row["response"]
            all_reasons = []
            final_score = None
            
            if not response or pd.isna(response):
                all_reasons.append("回答为空，无法评分")
            else:
                for attempt in range(3):
                    try:
                        judgment = self._evaluate_single(prompt, response)
                        all_reasons.append(f"第{attempt+1}次评估：{judgment[:100]}")
                        final_score = self._extract_score(judgment)
                        if final_score is not None:
                            break
                    except Exception as e:
                        all_reasons.append(f"第{attempt+1}次评估失败：{str(e)}")
            
            results.append({
                "index": idx,
                "prompt": prompt,
                "response": response,
                "final_score": final_score,
                "all_reasons": " | ".join(all_reasons)
            })
            
            if i % 10 == 0 or i == total_unprocessed:
                print(f"{model_name}评估进度：{i}/{total_unprocessed}")
        
        if results:
            temp_df = pd.DataFrame(results)
            combined_df = pd.concat([existing_df, temp_df], ignore_index=True)
            combined_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        
        final_df = pd.read_csv(output_file)
        valid_count = final_df["final_score"].notna().sum()
        print(f"{model_name}模型评分完成，有效评分数量：{valid_count}")
        return final_df

    def evaluate_pairwise(self, responses_a, responses_b, name_a, name_b, force_regenerate=False):
        """对比评估两个模型（修复列名问题）"""
        output_file = os.path.join(self.evaluation_dir, f"{name_a}_vs_{name_b}_comparison.csv")
        
        # 定义必须包含的列名（核心修复点1）
        required_columns = [
            "index", 
            "prompt", 
            f"response_{name_a}", 
            f"response_{name_b}", 
            f"prefer_{name_a}_prob",  # 确保偏好概率列存在
            "all_reasons"
        ]
        
        if not force_regenerate and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            try:
                existing_df = pd.read_csv(output_file)
                # 检查是否包含所有必要列且数据完整（核心修复点2）
                if (all(col in existing_df.columns for col in required_columns) and 
                    len(existing_df) == len(responses_a)):
                    print(f"检测到{name_a} vs {name_b}已有完整对比结果，直接加载")
                    return existing_df
                print(f"{name_a} vs {name_b}对比结果不完整或列名错误，重新评估...")
            except Exception as e:
                print(f"对比文件损坏: {str(e)}，重新评估...")
        
        # 合并两个模型的回答
        merged_df = pd.merge(
            responses_a[["index", "prompt", "response"]],
            responses_b[["index", "response"]],
            on="index",
            suffixes=(f"_{name_a}", f"_{name_b}")
        )
        
        # 初始化结果DataFrame，确保列名正确（核心修复点3）
        existing_df = pd.DataFrame(columns=required_columns)
        if os.path.exists(output_file):
            try:
                temp_df = pd.read_csv(output_file)
                # 只保留有效列，过滤脏数据
                existing_df = temp_df[required_columns] if all(col in temp_df.columns for col in required_columns) else existing_df
            except Exception:
                pass
        
        processed_indices = set(existing_df["index"]) if not existing_df.empty else set()
        results = []
        total = len(merged_df)
        unprocessed = [idx for idx in merged_df["index"] if idx not in processed_indices]
        total_unprocessed = len(unprocessed)
        
        if total_unprocessed == 0:
            print(f"{name_a} vs {name_b}所有对比已完成")
            return existing_df
            
        print(f"{name_a} vs {name_b}需要对比评估{total_unprocessed}对回答")
        
        for i, idx in enumerate(unprocessed, 1):
            row = merged_df[merged_df["index"] == idx].iloc[0]
            prompt = row["prompt"]
            response_a = row[f"response_{name_a}"]
            response_b = row[f"response_{name_b}"]
            all_reasons = []
            preference = None
            
            if not response_a or pd.isna(response_a) or not response_b or pd.isna(response_b):
                all_reasons.append("至少一个模型回答为空，无法对比")
            else:
                for attempt in range(3):
                    try:
                        judgment = self._evaluate_pair(prompt, response_a, response_b)
                        all_reasons.append(f"第{attempt+1}次对比：{judgment[:100]}")
                        preference = self._extract_preference(judgment)
                        if preference is not None:
                            break
                    except Exception as e:
                        all_reasons.append(f"第{attempt+1}次对比失败：{str(e)}")
            
            # 确保结果包含所有必要字段（核心修复点4）
            results.append({
                "index": idx,
                "prompt": prompt,
                f"response_{name_a}": response_a,
                f"response_{name_b}": response_b,
                f"prefer_{name_a}_prob": preference,  # 显式指定偏好概率列
                "all_reasons": " | ".join(all_reasons)
            })
            
            if i % 10 == 0 or i == total_unprocessed:
                print(f"{name_a} vs {name_b}对比进度：{i}/{total_unprocessed}")
        
        if results:
            temp_df = pd.DataFrame(results)
            # 合并时只保留必要列（核心修复点5）
            combined_df = pd.concat([existing_df, temp_df[required_columns]], ignore_index=True)
            combined_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        
        print(f"两模型对比完成，结果保存至{output_file}")
        return pd.read_csv(output_file)

    def get_comparison_stats(self, name_a, name_b):
        """获取对比统计数据"""
        comparison_df = pd.read_csv(os.path.join(self.evaluation_dir, f"{name_a}_vs_{name_b}_comparison.csv"))
        # 验证必要列是否存在（核心修复点6）
        required_col = f"prefer_{name_a}_prob"
        if required_col not in comparison_df.columns:
            raise ValueError(f"对比结果文件缺少必要列：{required_col}，请删除文件后重新运行")
        
        valid_df = comparison_df.dropna(subset=[required_col])
        valid_count = len(valid_df)
        
        a_wins = (valid_df[required_col] > 50).sum()
        b_wins = (valid_df[required_col] < 50).sum()
        ties = (valid_df[required_col] == 50).sum()
        
        # 加载单模型评分计算平均分差
        a_scores = pd.read_csv(os.path.join(self.evaluation_dir, f"{name_a}_single_scores.csv"))
        b_scores = pd.read_csv(os.path.join(self.evaluation_dir, f"{name_b}_single_scores.csv"))
        merged_scores = pd.merge(
            a_scores[["index", "final_score"]],
            b_scores[["index", "final_score"]],
            on="index",
            suffixes=(f"_{name_a}", f"_{name_b}")
        ).dropna()
        
        score_diff = merged_scores[f"final_score_{name_a}"].mean() - merged_scores[f"final_score_{name_b}"].mean() if not merged_scores.empty else 0
        
        return {
            "有效对比组数": valid_count,
            f"{name_a}获胜次数": a_wins,
            f"{name_a}胜率": (a_wins / valid_count) * 100 if valid_count > 0 else 0,
            f"{name_b}获胜次数": b_wins,
            f"{name_b}胜率": (b_wins / valid_count) * 100 if valid_count > 0 else 0,
            "平局次数": ties,
            "平局率": (ties / valid_count) * 100 if valid_count > 0 else 0,
            "平均分差": round(score_diff, 2),
            f"偏好{name_a}的概率": valid_df[required_col].mean() if valid_count > 0 else 0
        }