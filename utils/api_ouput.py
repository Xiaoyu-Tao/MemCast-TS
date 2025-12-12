from openai import OpenAI
import torch.nn as nn
import base64
import os
from typing import Optional, Union, List


class deepseek_api_output(nn.Module):
    def __init__(self,api_key='',temperature=0.6,top_p=0.7):
        super(deepseek_api_output, self).__init__()
        self.client = OpenAI(base_url="https://api2.aigcbest.top/v1",api_key=api_key)
        self.temperature = temperature
        self.top_p = top_p
    
    def forward(self, content):
        response = self.client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-0528",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": content},],stream=False,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=16384,)
        # name=response.model_name
        reasoning=response.choices[0].message.model_extra['reasoning_content']
        answer=response.choices[0].message.content        
        return reasoning,answer

class gpt4_api_output(nn.Module):
    def __init__(self, api_key='', temperature=0.6, top_p=0.7):
        super(gpt4_api_output, self).__init__()
        # 改为官方或自建代理的 GPT-4-mini 端点
        self.client = OpenAI(
            base_url="https://api2.aigcbest.top/v1",  # 若用官方 OpenAI 端点
            api_key=api_key
        )
        self.temperature = temperature
        self.top_p = top_p

    def forward(self, content):
        response = self.client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3.1",  # ✅ 改这里，官方 mini 模型名
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": content},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=16384,  # GPT-4-mini 支持 4096–8192 token
        )
        reasoning = None
        answer = response.choices[0].message.content
        return reasoning, answer



class deepseek_multimodal_api_output(nn.Module):
    """多模态API接口，支持文本+图像输入"""
    def __init__(self, api_key='', temperature=0.6, top_p=0.7):
        super(deepseek_multimodal_api_output, self).__init__()
        self.client = OpenAI(base_url="https://api2.aigcbest.top/v1", api_key=api_key)
        self.temperature = temperature
        self.top_p = top_p
    
    def forward(self, content: str, image_path: Optional[Union[str, List[str]]] = None):
        """
        多模态前向传播，支持文本和图像输入
        
        Args:
            content: 文本提示内容
            image_path: 图像路径，可以是单个路径(str)或多个路径(List[str])
        
        Returns:
            tuple: (reasoning, answer)
        """
        user_message = {"role": "user"}
        
        if image_path:
            # 构建消息内容列表
            message_content = [{"type": "text", "text": content}]
            
            # 处理单个或多个图像路径
            if isinstance(image_path, str):
                image_paths = [image_path]
            else:
                image_paths = image_path
            
            # 为每个图像添加base64编码
            for img_path in image_paths:
                try:
                    with open(img_path, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    
                    # 检测图像格式
                    image_format = "png"
                    if img_path.lower().endswith('.jpg') or img_path.lower().endswith('.jpeg'):
                        image_format = "jpeg"
                    elif img_path.lower().endswith('.gif'):
                        image_format = "gif"
                    elif img_path.lower().endswith('.webp'):
                        image_format = "webp"
                    
                    message_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format};base64,{base64_image}"
                        }
                    })
                except Exception as e:
                    print(f"Warning: Failed to load image {img_path}: {e}")
                    continue
            
            user_message["content"] = message_content
        else:
            # 仅文本模式
            user_message["content"] = content
        
        response = self.client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                user_message
            ],
            stream=False,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=16384,
        )
        
        # reasoning = response.choices[0].message.model_extra['reasoning_content']

        answer = response.choices[0].message.content
        return answer
    

# if __name__ == "__main__":
#     model=deepseek_api_output(api_key='sk-PxM40luD13UVKLhp6k3zenHC2XPASEi5uazXuXsCfTrQ3hUQ',temperature=0.6,top_p=0.7)
#     reasoning,answer=model("""Here is the High UseFul Load data of the transformer.I will now give you data for the past 96 recorded dates, and please help me forecast the data for next 96 recorded dates.But please note that these data will have missing values, so be aware of that..Please give me the complete data for the next 96 recorded dates, remember to give me the complete data.You must provide the complete data.You mustn't omit any content.The data is as follows:               date       HUFL
# 2017-10-20 00:00:00  10.449000
# 2017-10-20 01:00:00  11.119000
# 2017-10-20 02:00:00   9.511000
# 2017-10-20 03:00:00   9.645000
# 2017-10-20 04:00:00   8.908000
# 2017-10-20 05:00:00  10.784000
# 2017-10-20 06:00:00  11.922000
# 2017-10-20 07:00:00  10.918000
# 2017-10-20 08:00:00   3.617000
# 2017-10-20 09:00:00  -4.890000
# 2017-10-20 10:00:00  -6.162000
# 2017-10-20 11:00:00  -0.268000
# 2017-10-20 12:00:00  -3.215000
# 2017-10-20 13:00:00  -4.019000
# 2017-10-20 14:00:00  -0.804000
# 2017-10-20 15:00:00   2.947000
# 2017-10-20 16:00:00   5.425000
# 2017-10-20 17:00:00   7.502000
# 2017-10-20 18:00:00   7.368000
# 2017-10-20 19:00:00   8.172000
# 2017-10-20 20:00:00   7.368000
# 2017-10-20 21:00:00   6.497000
# 2017-10-20 22:00:00   7.502000
# 2017-10-20 23:00:00  10.382000
# 2017-10-21 00:00:00   8.975000
# 2017-10-21 01:00:00   9.511000
# 2017-10-21 02:00:00   7.569000
# 2017-10-21 03:00:00   8.172000
# 2017-10-21 04:00:00   9.310000
# 2017-10-21 05:00:00   9.645000
# 2017-10-21 06:00:00   9.779000
# 2017-10-21 07:00:00   9.243000
# 2017-10-21 08:00:00   1.808000
# 2017-10-21 09:00:00  -6.162000
# 2017-10-21 10:00:00 -10.181000
# 2017-10-21 11:00:00 -11.989000
# 2017-10-21 12:00:00 -14.334000
# 2017-10-21 13:00:00 -13.865000
# 2017-10-21 14:00:00 -10.315000
# 2017-10-21 15:00:00  -2.746000
# 2017-10-21 16:00:00   2.679000
# 2017-10-21 17:00:00   7.301000
# 2017-10-21 18:00:00   7.770000
# 2017-10-21 19:00:00   7.904000
# 2017-10-21 20:00:00   7.301000
# 2017-10-21 21:00:00   8.573000
# 2017-10-21 22:00:00   9.645000
# 2017-10-21 23:00:00  11.922000
# 2017-10-22 00:00:00  12.659000
# 2017-10-22 01:00:00  11.989000
# 2017-10-22 02:00:00  10.985000
# 2017-10-22 03:00:00  11.788000
# 2017-10-22 04:00:00  11.922000
# 2017-10-22 05:00:00  12.056000
# 2017-10-22 06:00:00  11.721000
# 2017-10-22 07:00:00  11.521000
# 2017-10-22 08:00:00   2.947000
# 2017-10-22 09:00:00  -5.224000
# 2017-10-22 10:00:00  -9.712000
# 2017-10-22 11:00:00 -12.324000
# 2017-10-22 12:00:00 -16.275999
# 2017-10-22 13:00:00 -14.334000
# 2017-10-22 14:00:00  -9.042000
# 2017-10-22 15:00:00  -5.023000
# 2017-10-22 16:00:00   1.407000
# 2017-10-22 17:00:00   6.765000
# 2017-10-22 18:00:00   7.435000
# 2017-10-22 19:00:00   7.904000
# 2017-10-22 20:00:00   6.966000
# 2017-10-22 21:00:00   9.377000
# 2017-10-22 22:00:00   9.913000
# 2017-10-22 23:00:00  12.860000
# 2017-10-23 00:00:00  14.066000
# 2017-10-23 01:00:00  12.659000
# 2017-10-23 02:00:00  10.784000
# 2017-10-23 03:00:00  12.257000
# 2017-10-23 04:00:00  10.851000
# 2017-10-23 05:00:00  11.454000
# 2017-10-23 06:00:00  11.320000
# 2017-10-23 07:00:00  11.387000
# 2017-10-23 08:00:00   7.301000
# 2017-10-23 09:00:00   3.751000
# 2017-10-23 10:00:00   0.134000
# 2017-10-23 11:00:00  -3.416000
# 2017-10-23 12:00:00  -6.296000
# 2017-10-23 13:00:00  -5.961000
# 2017-10-23 14:00:00  -5.358000
# 2017-10-23 15:00:00   1.206000
# 2017-10-23 16:00:00   0.536000
# 2017-10-23 17:00:00   6.966000
# 2017-10-23 18:00:00   6.899000
# 2017-10-23 19:00:00   8.707000
# 2017-10-23 20:00:00   8.105000
# 2017-10-23 21:00:00   7.167000
# 2017-10-23 22:00:00   7.100000
# 2017-10-23 23:00:00   9.176000And your final answer must follow the format
#     <answer>
        
# ```

#         ...
        
# ```

#         </answer>
#     Please obey the format strictly. And you must give me the complete answer.
    
#     """)
#     print(reasoning)
#     print(answer)


if __name__ == "__main__":
    # 测试 deepseek_multimodal_api_output
    # 注意：请替换为您的实际 API key
    api_key = 'sk-PxM40luD13UVKLhp6k3zenHC2XPASEi5uazXuXsCfTrQ3hUQ'
    
    # 创建多模态模型实例
    multimodal_model = deepseek_multimodal_api_output(
        api_key=api_key,
        temperature=0.6,
        top_p=0.7
    )
    
    # print("=" * 80)
    # print("测试 1: 纯文本输入")
    # print("=" * 80)
    # try:
    #     reasoning, answer = multimodal_model.forward(
    #         content="请简单介绍一下人工智能的发展历程。"
    #     )
    #     print("\n推理过程 (reasoning):")
    #     print(reasoning)
    #     print("\n最终答案 (answer):")
    #     print(answer)
    # except Exception as e:
    #     print(f"错误: {e}")
    
    print("\n" + "=" * 80)
    print("测试 2: 文本 + 单个图像输入")
    print("=" * 80)
    # 使用项目中现有的图像文件
    image_path = "/data/songliv/TS/TimeReasoner/output/EPF/NP/NP_visualization_000.png"
    if not os.path.exists(image_path):
        print(f"警告: 图像文件不存在: {image_path}")
        print("请检查图像路径是否正确，或使用其他可用的图像文件。")
    else:
        try:
            reasoning, answer = multimodal_model.forward(
                content="请描述这张图片的内容。这是一张时间序列可视化图，请分析其中的趋势和模式。",
                image_path=image_path
            )
            print("\n推理过程 (reasoning):")
            print(reasoning)
            print("\n最终答案 (answer):")
            print(answer)
        except Exception as e:
            print(f"错误: {e}")
    
    print("\n" + "=" * 80)
    print("测试 3: 文本 + 多个图像输入")
    print("=" * 80)
    # 注意：请替换为实际的图像路径列表
    # image_paths = ["path/to/image1.png", "path/to/image2.jpg"]  # 取消注释并替换路径
    # try:
    #     reasoning, answer = multimodal_model.forward(
    #         content="请比较这两张图片的异同。",
    #         image_path=image_paths
    #     )
    #     print("\n推理过程 (reasoning):")
    #     print(reasoning)
    #     print("\n最终答案 (answer):")
    #     print(answer)
    # except Exception as e:
    #     print(f"错误: {e}")