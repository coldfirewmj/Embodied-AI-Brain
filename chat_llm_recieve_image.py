import os
import sys
original_cwd = os.getcwd()
import time
from textwrap import dedent
import requests
from urllib.parse import urljoin
import json
from prompt_build import get_inst_plan,get_inst_chat,get_inst_find

import threading
import uvicorn
from fastapi import FastAPI, Request
import queue
import cv2
# 线程安全队列（推荐）
stt_queue = queue.Queue(maxsize=5)
# 接收STT服务器的消息
app = FastAPI(title="Main STT Receiver")

@app.post("/v1")
async def receive_stt_text(request: Request):
    try:
        body = await request.json()  # ← 必须 await！
        text = body.get("text", "")
        if text and text.strip():
            stt_queue.put(text.strip())
            print(f"📨 主进程收到STT: {text}")
        return {"status": "ok"}
    except Exception as e:
        print(f"❌ 解析JSON失败: {e}")
        return {"status": "error", "message": str(e)}

def start_stt_receiver():
    """在后台线程启动HTTP接收服务"""
    uvicorn.run(app, host="127.0.0.1", port=28184, log_level="warning")
    print("STT receiver started."," host:","127.0.0.1"," port:",28184)

# 启动接收服务（daemon=True 确保随主进程退出）
threading.Thread(target=start_stt_receiver, daemon=True).start()

# TTS发送给服务器
class TTSAgent:
    def __init__(self, host_url):
        print(f"TTSAgent: host_url {host_url}")
        self.host_url = host_url

    def run(self, input_dict: dict) -> str:
        # 直接传入字典，不要在外面转字符串
        try:
            # 这里的路径直接指向 /v1，不使用 'exec'
            resp = requests.post(self.host_url, json=input_dict, timeout=5)
            if resp.status_code == 200:
                return "ok"
        except Exception as e:
            print(f"TTSAgent Error: {e}")
        return ""

tts = TTSAgent("http://127.0.0.1:28185/v1")

def tts_sound(tts_agent, text, lang, interrupt):
    # 构造字典
    input_dict = {"task": "text_to_speech", "lang": lang, "text": text, "interrupt": interrupt}
    print(f"📤 发送中: {text}")

    # 直接发送
    tts_agent.run(input_dict)
    return 0

# 切换到sam2工作目录，加载sam
print('开始加载sam2模型')
start_time = time.time()
sam_path = original_cwd+'/camera/Grounded-SAM-2'
sys.path.append(sam_path)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
CHECKPOINT_SAM = os.path.join(sam_path,"sam2_checkpoints", "sam2.1_hiera_small.pt")
# CONFIG_SAM前面必须加'/'
CONFIG_SAM = '/'+sam_path+"/sam2/configs/sam2.1/sam2.1_hiera_s.yaml"
# Load the model
sam2_model = build_sam2(CONFIG_SAM, CHECKPOINT_SAM, device="cuda")
sam_predictor = SAM2ImagePredictor(sam2_model)
print(f"🎉 加载SAM模型耗时: {time.time() - start_time:.2f} 秒")


# 启动大模型客户端
print('开始连接大模型服务')
start_time= time.time()
import base64
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
print(f"🎉 加载大模型服务耗时: {time.time() - start_time:.2f} 秒")
start_time= time.time()

import io
import numpy as np
# 合成messages
def input_messages_build(prompt, image=None):
    print(prompt)
    # 构建文本信息
    text_content = {
        "type": "text",
        "text": prompt
    }
    if image is None:
        return [
        {
            "role": "user",
            "content": [text_content],
        }]
    else:
        # 1. 创建一个字节流容器
        buffer = io.BytesIO()
        # 2. 将 PIL 图像保存到容器中（指定格式）
        image.save(buffer, format="PNG")
        # 转换为 base64
        image_b64 = base64.b64encode(buffer).decode("utf-8")
        # 构建图像信息
        image_contents = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"
                    },
                }
        return [
            {
                "role": "user",
                "content":   [text_content, image_contents],
            }]

import re
sentence_endings = r'[。！？!?；;…\n]'  # 中英文句尾符号
# 得到回答并解析出包围框
def get_respose_(model,text,image):
    start_time = time.time()
    # 此处做一个判断，如果是C则聊天，如果是G则抓取，由于接下来如果是C则流式输出的，所以需要先判断
    
    prompt = get_inst_find(text)
    messages = input_messages_build(prompt, image)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        extra_body={"mm_processor_kwargs":{"fps": [1]}},
        stream=False,
    )
    chunk = response.choices[0].message.content
    print("回复为：",chunk)
    print("回复为：",chunk)
    if chunk is None or chunk == '[]':
        tts_sound(tts,"未找到物体","zh",True)
        entity = []
        return entity
    entity = json.loads(chunk)
    print('包围框为：',entity)
    tts_sound(tts,"已找到物体","zh",True)

    # entity = stream_message.strip('```json\n').strip('```')
    # print("entity：", entity)
    return entity


# 图像线程安全队列（推荐）
img_queue = queue.Queue(maxsize=1)
image_app = FastAPI(title="Image Receiver")
@image_app.post("/api/process_image")
async def receive_image(request: Request):
    # 1. 获取 JSON 数据 (注意这里必须使用 await)
    data = await request.json()
    image_data = data.get('image')

    # 2. 处理 Base64 数据头
    if image_data.startswith('data:image'):
        image_data = image_data.split(',')[1]
    
    # 3. 解码 Base64 图像
    image_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_queue.put(img)
    
    # 在这里进行图像处理
    # mask = your_processing_function(img)
    
    # 4. 返回响应
    return {
        "success": True, 
        "message": "Image received successfully"
    }

def send_mask_to_gui(mask_image, gui_url="http://localhost:50052"):
    """
    将 mask 发送到 GUI
    
    Args:
        mask_image: PIL Image 对象（灰度模式，L mode）
        gui_url: GUI 服务器地址
    """
    # 转换为 Base64
    buffered = io.BytesIO()
    mask_image.save(buffered, format="PNG")
    mask_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    mask_data = f"data:image/png;base64,{mask_base64}"
    
    # 发送请求
    response = requests.post(
        f"{gui_url}/external/receive_mask",
        headers={'Content-Type': 'application/json'},
        json={'mask_data': mask_data},
        timeout=10
    )
    
    if response.status_code == 200:
        result = response.json()
        if result.get('success'):
            print("Mask sent successfully")
        else:
            print(f"Failed: {result.get('message')}")
    else:
        print(f"HTTP Error: {response.status_code}")


def main():
    cv2.namedWindow("output image", cv2.WINDOW_AUTOSIZE)
    out_image = None
    while True:

        try:
            image = img_queue.get_nowait()
            out_image = image.copy()
            text = stt_queue.get_nowait()
            # 获取包围框
            input_box = get_respose_('Qwen2.5-VL-7B-Instruct',text,image)
            if input_box!=[]:
                start_time = time.time()
                x1, y1, x2, y2 = input_box
                # -------------------------- SAM3推理 --------------------------
                inference_state = sam_predictor.set_image(image)
                masks, scores, _ = sam_predictor.predict(
                    box=input_box,
                    multimask_output=False  # 单mask更高效
                )
                print("框掩码预测时间为",time.time()-start_time)
                print("可视化")
                mask = masks[np.argmax(scores)]
                # 掩码转二值图
                mask_np = mask.astype(np.uint8) * 255
                # 找掩码轮廓
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                print("轮廓数量：", len(contours))
                # 绘制轮廓（红色，线宽2）
                cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
                # 绘制包围框（绿色，线宽2）
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.namedWindow("detection window", cv2.WINDOW_AUTOSIZE)
                # 强制窗口置顶（某些Linux桌面环境支持）
                cv2.setWindowProperty("detection window", cv2.WND_PROP_TOPMOST, 1)
                # -------------------------- 关键步骤：显示并强制刷新 --------------------------
                cv2.imshow("detection window", image)
                # 这里的 waitKey 非常关键，它能让系统有时间渲染出这个新窗口
                cv2.waitKey(1)
                # 获得该物体的xyz坐标（包围框内掩码的像素点对应的xyz求平均）
                # for u in range(x1,x2+1):
                #     for v in range(y1,y2+1):
                #         depth_val = depth_image[v,u]
                #         if(depth_val > 0 and mask[v,u]):
                #             [x,y,z] = rs.rs2_deproject_pixel_to_point(
                #                 depth_intrin, [u, v], depth_val / 1000.0)
                #             # print('物体坐标是：x:',x,'y:',y,'z:',z)
                #             break

        except queue.Empty:
            # 当队列为空时，继续循环而不是抛出异常
            pass
        if out_image:
            # 显示结果
            cv2.imshow("output image", out_image)
            # 给系统 1ms 时间刷新窗口
            cv2.waitKey(1)

            key = cv2.waitKey(1)& 0xFF
            # 按q或者ESC退出
            if key == ord('q') or key == 27:
                break

if __name__ == "__main__":
        main()
