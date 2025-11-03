import os
import time
import pandas as pd
from openai import OpenAI
from openai import RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests.exceptions
from config.api_keys import HF_TOKENS, MODEL_ENDPOINTS, OPENROUTER_CONFIG

class ModelClient:
    def __init__(self, model_size):
        self.model_size = model_size
        self.model_name = MODEL_ENDPOINTS[model_size]
        self.response_dir = os.path.join("data", "model_responses")
        os.makedirs(self.response_dir, exist_ok=True)
        
        # 初始化客户端
        if model_size == "70B":
            self.client = OpenAI(
                base_url=OPENROUTER_CONFIG["base_url"],
                api_key=OPENROUTER_CONFIG["api_key"]
            )
            self.extra_headers = OPENROUTER_CONFIG["extra_headers"]
            # 速率限制配置
            self.rate_limit_config = {
                "per_minute": 15,
                "per_day": 45,
                "request_count": 0,
                "minute_start": time.time(),
                "day_start": time.time() - (time.time() % 86400)
            }
        else:
            self.client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=HF_TOKENS[model_size]
            )
            self.extra_headers = {}
            self.rate_limit_config = None

    def _check_rate_limit(self):
        """检查70B模型的速率限制，超限则等待"""
        if self.model_size != "70B":
            return
        
        current_time = time.time()
        # 检查每日额度
        if current_time - self.rate_limit_config["day_start"] >= 86400:
            self.rate_limit_config["request_count"] = 0
            self.rate_limit_config["day_start"] = current_time
        
        if self.rate_limit_config["request_count"] >= self.rate_limit_config["per_day"]:
            next_day = self.rate_limit_config["day_start"] + 86400
            wait_time = next_day - current_time + 60
            print(f"70B模型每日额度用尽，需等待{wait_time//3600:.0f}小时{wait_time%3600//60:.0f}分钟")
            time.sleep(wait_time)
            self.rate_limit_config["request_count"] = 0
            self.rate_limit_config["day_start"] = time.time()
        
        # 检查每分钟额度
        if current_time - self.rate_limit_config["minute_start"] >= 60:
            self.rate_limit_config["minute_start"] = current_time
        
        minute_requests = self.rate_limit_config["request_count"] - \
            int((self.rate_limit_config["minute_start"] - self.rate_limit_config["day_start"]) // 60 * self.rate_limit_config["per_minute"])
        
        if minute_requests >= self.rate_limit_config["per_minute"]:
            wait_time = 60 - (current_time - self.rate_limit_config["minute_start"]) + 2
            print(f"70B模型每分钟额度用尽，等待{wait_time:.0f}秒")
            time.sleep(wait_time)
            self.rate_limit_config["minute_start"] = time.time()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=3, max=30),
        retry=retry_if_exception_type((
            requests.exceptions.RequestException, 
            Exception,
            RateLimitError
        ))
    )
    def get_response(self, prompt):
        """获取模型回答，包含速率控制"""
        self._check_rate_limit()
        
        try:
            kwargs = {"timeout": 120.0}
            if self.model_size == "70B":
                kwargs["extra_headers"] = self.extra_headers
                
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": f"请根据以下题目写一篇作文：{prompt}"}
                ],
                temperature=0.7,
                max_tokens=1000,
                **kwargs
            )
            
            if self.model_size == "70B":
                self.rate_limit_config["request_count"] += 1
            
            return response.choices[0].message.content.strip()
        
        except RateLimitError as e:
            print(f"70B模型速率限制：{str(e)}，将重试...")
            if "X-RateLimit-Reset" in str(e):
                reset_time = int(str(e).split("X-RateLimit-Reset': '")[1].split("'")[0]) / 1000
                wait_time = reset_time - time.time() + 5
                if wait_time > 0:
                    print(f"按服务器提示等待{wait_time:.0f}秒")
                    time.sleep(wait_time)
            raise
        
        except Exception as e:
            print(f"模型{self.model_size}调用失败: {str(e)}")
            raise

    def generate_single_response(self, prompt):
        """单独生成一条回答（用于定向修复）"""
        self._check_rate_limit()
        try:
            kwargs = {"timeout": 120.0}
            if self.model_size == "70B":
                kwargs["extra_headers"] = self.extra_headers
                
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": f"请根据以下题目写一篇作文：{prompt}"}],
                temperature=0.7,
                max_tokens=1000,** kwargs
            )
            
            if self.model_size == "70B":
                self.rate_limit_config["request_count"] += 1
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"单条生成失败：{str(e)}")
            return None

    def process_prompts(self, prompts_df, force_regenerate=False):
        output_file = os.path.join(self.response_dir, f"{self.model_size}_responses.csv")
        
        if not force_regenerate and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            try:
                existing_df = pd.read_csv(output_file)
                if len(existing_df) == len(prompts_df) and "response" in existing_df.columns:
                    print(f"检测到{self.model_size}已有完整结果，直接加载（{len(existing_df)}条）")
                    return existing_df
                print(f"{self.model_size}结果不完整，补充生成...")
            except Exception as e:
                print(f"{self.model_size}文件损坏: {str(e)}，重新生成...")
        
        existing_df = pd.DataFrame(columns=["index", "prompt", "response"]) if not os.path.exists(output_file) else pd.read_csv(output_file)
        processed_indices = set(existing_df["index"]) if not existing_df.empty else set()
        results = []
        total = len(prompts_df)
        unprocessed = [idx for idx in prompts_df.index if idx not in processed_indices]
        total_unprocessed = len(unprocessed)
        
        if total_unprocessed == 0:
            print(f"{self.model_size}所有题目已处理完成")
            return existing_df
            
        print(f"{self.model_size}需要处理{total_unprocessed}个题目")
        
        for i, idx in enumerate(unprocessed, 1):
            row = prompts_df.loc[idx]
            prompt = row["prompt"]
            try:
                response = self.get_response(prompt)
                results.append({
                    "index": idx,
                    "prompt": prompt,
                    "response": response
                })
                if self.model_size == "70B":
                    avg_time_per_request = 10
                    remaining_time = (total_unprocessed - i) * avg_time_per_request
                    print(f"70B模型：已完成{i}/{total_unprocessed}，剩余约{remaining_time//60:.0f}分钟")
                else:
                    if i % 5 == 0 or i == total_unprocessed:
                        print(f"{self.model_size}：已完成{i}/{total_unprocessed}")
            except Exception as e:
                print(f"处理题目{idx}失败，跳过: {str(e)}")
                continue
        
        if results:
            temp_df = pd.DataFrame(results)
            combined_df = pd.concat([existing_df, temp_df], ignore_index=True)
            combined_df.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"{self.model_size}处理完成，共{len(combined_df)}个题目")
        
        return pd.read_csv(output_file)