import os
import pandas as pd
from openai import OpenAI
from config.api_keys import DEEPSEEK_API_KEY

class EssayPromptGenerator:
    def __init__(self):
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"  # DeepSeek API基础URL
        )
        self.output_path = os.path.join("data", "essay_prompts.csv")
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
    def generate_prompts(self, num_prompts=200):
        """生成指定数量的作文题，增强生成逻辑确保数量达标"""
        # 首次生成时多要20%作为冗余
        target_first_pass = int(num_prompts * 1.2)
        
        system_prompt = f"""
        你是一位专业的教育工作者，需要生成多样化的作文题目。
        要求:
        1. 题目类型包括记叙文、议论文、说明文、应用文等
        2. 主题涵盖成长感悟、社会热点、文化传承、科技发展、环境保护、
           人际交往、理想信念、读书感悟、家乡变化等多个领域
        3. 难度适中，适合中学生至大学生水平
        4. 每个题目需简洁明了（10-30字），具有明确的写作方向
        5. 避免重复或相似度过高的题目
        请生成{target_first_pass}个作文题目，每个题目单独一行，不要编号，不要添加额外说明。
        """
        
        try:
            # 首次生成
            prompts = self._generate_batch(target_first_pass, system_prompt)
            
            # 多次补充生成直到满足数量
            max_attempts = 5  # 最大补充尝试次数
            attempt = 0
            while len(prompts) < num_prompts and attempt < max_attempts:
                need_more = num_prompts - len(prompts)
                print(f"需要补充生成{need_more}个作文题（尝试{attempt+1}/{max_attempts}）")
                additional = self._generate_additional_prompts(need_more, prompts)
                prompts += additional
                # 去重（保持顺序）
                prompts = list(dict.fromkeys(prompts))
                attempt += 1
            
            # 最终确保数量，如果仍不足则放宽条件
            if len(prompts) < num_prompts:
                print(f"严格模式下仍不足，放宽条件补充生成...")
                final_need = num_prompts - len(prompts)
                final_additional = self._generate_additional_prompts(
                    final_need, 
                    prompts, 
                    strict=False
                )
                prompts += final_additional
                prompts = list(dict.fromkeys(prompts))[:num_prompts]
            
            # 最终检查
            if len(prompts) < num_prompts:
                # 即使有重复也先满足数量要求
                shortage = num_prompts - len(prompts)
                prompts += prompts[:shortage]  # 用已有题目补充
                print(f"警告：使用部分重复题目满足数量要求，实际独特题目{len(set(prompts))}个")
            
            # 保存结果
            df = pd.DataFrame(prompts[:num_prompts], columns=["prompt"])
            df.to_csv(self.output_path, index=False)
            print(f"成功生成{len(df)}个作文题（独特题目{len(df['prompt'].unique())}个），已保存至{self.output_path}")
            return df
        
        except Exception as e:
            print(f"生成作文题失败: {str(e)}")
            if os.path.exists(self.output_path):
                os.remove(self.output_path)
            raise
    
    def _generate_batch(self, num, system_prompt):
        """单次生成一批作文题"""
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "请生成上述要求的作文题目"}
                ],
                temperature=0.8,
                max_tokens=min(4000, num * 50)  # 按每个题目约50token估算
            )
            
            # 处理生成结果
            prompts = response.choices[0].message.content.strip().split("\n")
            # 过滤空行和过短内容（放宽条件，至少5个字符）
            prompts = [p.strip() for p in prompts if p.strip() and len(p) >= 5]
            return prompts
        except Exception as e:
            print(f"批量生成失败: {str(e)}")
            return []
    
    def _generate_additional_prompts(self, num, existing_prompts, strict=True):
        """生成补充题目，避免与已有题目重复"""
        # 提取已有题目的关键词，帮助模型避免重复
        keywords = []
        for p in existing_prompts[-20:]:  # 只取最近的20个避免过长
            keywords.extend(p.split()[:5])  # 每个题目取前5个词
        unique_keywords = list(set(keywords))[:10]  # 最多10个关键词
        
        system_prompt = f"""
        请补充生成{num}个作文题目，要求：
        1. 与已有题目不重复、不相似
        2. 已有题目关键词：{', '.join(unique_keywords)}
        3. 每个题目单独一行，不要编号
        {'4. 严格避免与上述关键词重复的主题' if strict else ''}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "请生成新的作文题目，确保多样性"}
                ],
                temperature=0.9,  # 提高多样性
                max_tokens=min(2000, num * 50)
            )
            
            prompts = response.choices[0].message.content.strip().split("\n")
            return [p.strip() for p in prompts if p.strip() and len(p) >= 5]
        except Exception as e:
            print(f"补充生成失败: {str(e)}")
            return []
    
    def load_prompts(self):
        """加载已生成的作文题，处理空文件或无效文件"""
        if os.path.exists(self.output_path):
            if os.path.getsize(self.output_path) > 0:
                try:
                    df = pd.read_csv(self.output_path)
                    if not df.empty and "prompt" in df.columns and len(df) > 0:
                        print(f"成功加载已生成的{len(df)}个作文题")
                        return df
                    else:
                        print("作文题文件内容无效，重新生成...")
                except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                    print(f"解析作文题文件失败: {str(e)}，重新生成...")
        
        # 生成新的作文题
        return self.generate_prompts()