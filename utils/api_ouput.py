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
        reasoning=response.choices[0].message.model_extra['reasoning_content']
        answer=response.choices[0].message.content        
        return reasoning,answer

class gpt4_api_output(nn.Module):
    def __init__(self, api_key='', temperature=0.6, top_p=0.7):
        super(gpt4_api_output, self).__init__()
        self.client = OpenAI(
            base_url="https://api2.aigcbest.top/v1",
            api_key=api_key
        )
        self.temperature = temperature
        self.top_p = top_p

    def forward(self, content):
        response = self.client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3.1",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": content},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=16384,
        )
        reasoning = None
        answer = response.choices[0].message.content
        return reasoning, answer



class deepseek_multimodal_api_output(nn.Module):
    def __init__(self, api_key='', temperature=0.6, top_p=0.7):
        super(deepseek_multimodal_api_output, self).__init__()
        self.client = OpenAI(base_url="https://api2.aigcbest.top/v1", api_key=api_key)
        self.temperature = temperature
        self.top_p = top_p
    
    def forward(self, content: str, image_path: Optional[Union[str, List[str]]] = None):
        user_message = {"role": "user"}
        
        if image_path:
            message_content = [{"type": "text", "text": content}]
            
            if isinstance(image_path, str):
                image_paths = [image_path]
            else:
                image_paths = image_path
            
            for img_path in image_paths:
                try:
                    with open(img_path, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    
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

        answer = response.choices[0].message.content
        return answer