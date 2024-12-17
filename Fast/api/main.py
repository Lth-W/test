from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import openai
import base64
import os

# 初始化 FastAPI
app = FastAPI()

# 配置 OpenAI API
base_url = 'http://182.92.156.78:8889/v1/'
vl_base_url = "http://182.92.156.78:8000/v1/"
model = 'Qwen2.5-32B-Instruct-GPTQ-Int4'
image_dir = "D:\\Work\\H2V\\H2V\\src\\images"

# 初始化 LLM
llm = openai.OpenAI(api_key='sk-xx', base_url=base_url)

# 定义请求数据模型
class CompletionRequest(BaseModel):
    message: str

@app.post("/completion")
async def completion(request: CompletionRequest):
    """处理文本生成请求"""
    try:
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': request.message}
        ]
        kwargs = {'temperature': 0, 'top_p': 1e-5, 'n': 1}
        completion = llm.chat.completions.create(model=model, messages=messages, stream=False, **kwargs)
        result = completion.choices[0].message.content
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in completion: {str(e)}")

@app.post("/image-caption")
async def image_caption(file: UploadFile = File(...)):
    """处理图像描述请求"""
    try:
        # 保存上传的图像
        file_path = os.path.join(image_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # 初始化 OpenAI 客户端
        client = openai.OpenAI(api_key="sk-xx", base_url=vl_base_url)

        # 将图像编码为 Base64
        with open(file_path, "rb") as f:
            encoded_image = base64.b64encode(f.read())
        encoded_image_text = encoded_image.decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"

        kwargs = {'temperature': 1e-5, 'top_p': 1e-5, 'n': 1}
        chat_response = client.chat.completions.create(
            model="Qwen2-VL-7B-Instruct",
            messages=[
                {"role": "system", "content": "你是专业的昆虫学家"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": base64_qwen},
                        },
                        {"type": "text", "text": "描述图片包含的主要内容"},
                    ],
                },
            ],
            **kwargs
        )

        result = chat_response.choices[0].message.content
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in image caption: {str(e)}")
