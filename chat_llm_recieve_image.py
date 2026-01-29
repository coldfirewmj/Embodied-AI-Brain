import os
import sys
original_cwd = os.getcwd()
import time
import requests
import json
from prompt_build import get_inst_find,get_inst_figure,get_inst_plan,get_inst_chat,get_inst_see

import threading
import uvicorn
from fastapi import FastAPI, Request
import queue
import cv2
import torch
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
sam2_model.half()
sam_predictor = SAM2ImagePredictor(sam2_model)
print(f"🎉 加载SAM模型耗时: {time.time() - start_time:.2f} 秒")


# 启动大模型客户端
print('开始连接大模型服务')
start_time= time.time()
import base64
from openai import OpenAI
client = OpenAI(base_url="http://192.168.20.49:8000/v1", api_key="EMPTY")
print(f"🎉 加载大模型服务耗时: {time.time() - start_time:.2f} 秒")

import io
import numpy as np
# 合成messages
def input_messages_build(prompt, image=None):
    # prompt='图片中有什么内容'
    print(prompt)
    # 构建文本信息
    text_content = {
        "type": "text",
        "text": prompt
    }
    # print('image is:',image)
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
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        # 构建图像信息
        image_contents = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                        "min_pixels": 256 * 256,
                        "max_pixels": 2000 * 2000
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
def get_bounding_box(model,text,image):
    width, height = image.size
    prompt = get_inst_find(text)
    messages = input_messages_build(prompt, image)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        extra_body={"mm_processor_kwargs":{"fps": [1]}},
        stream=False,
    )
    # print(response)
    chunk = response.choices[0].message.content
    print("回复为：",chunk)
    if chunk is None or chunk == '[]':
        # tts_sound(tts,"未找到物体","zh",True)
        entity = []
        return entity
    qwen3_entity = json.loads(chunk)
    [x1,y1,x2,y2]=qwen3_entity
    entity=[round(x1*width/1000), round(y1*height/1000),
            round(x2*width/1000), round(y2*height/1000)]
    print('包围框为：',entity)
    # tts_sound(tts,"已找到物体","zh",True)

    # entity = stream_message.strip('```json\n').strip('```')
    # print("entity：", entity)
    return entity

# 根据用户的话提取出被拿物体与放置容器
def extract_objects(model,text):
    prompt = get_inst_figure(text)
    messages = input_messages_build(prompt)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        extra_body={"mm_processor_kwargs":{"fps": [1]}},
        stream=False,
    )
    """完整解析函数，处理各种可能的输出格式"""
    raw_content = response.choices[0].message.content

    # 方法1: 尝试直接解析JSON
    try:
        clean_content = raw_content.strip()
        parsed = json.loads(clean_content)
        return {
            "被拿物体": parsed["被拿物体"],
            "放置容器": parsed["放置容器"]
        }
    except:
        pass

    # 方法2: 尝试提取JSON片段
    try:
        json_match = re.search(r'\{[^{}]*\}', raw_content)
        if json_match:
            parsed = json.loads(json_match.group(0))
            return {
                "被拿物体": parsed["被拿物体"],
                "放置容器": parsed["放置容器"]
            }
    except:
        pass

    # 方法3: 正则匹配关键字段
    main_obj = re.search(r'被拿物体\s*[:：]\s*([^\s,]+)', raw_content)
    target_obj = re.search(r'放置容器\s*[:：]\s*([^\s,]+)', raw_content)

    return {
        "被拿物体": main_obj.group(1) if main_obj else "",
        "放置容器": target_obj.group(1) if target_obj else ""
    }
# 根据用户的话进行对话
def chat_with_llm(model,text,image=None):
    start_time = time.time()
    first = True
    if image is None:
        prompt = get_inst_chat(text)
        messages = input_messages_build(prompt)
    else:
        prompt = get_inst_see(text)
        messages = input_messages_build(prompt, image)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        extra_body={"mm_processor_kwargs":{"fps": [1]}},
        stream=True,
    )
    buffer = ""
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            buffer += token
            # 检查是否遇到完整句子结尾
            if re.search(sentence_endings, token):
                # 去掉末尾空白
                sentence = buffer.strip()
                if sentence:
                    tts_sound(tts,sentence,"zh", first)
                    if first:
                        print(f"再次判断耗时: {time.time()-start_time}")
                        first = False
                    buffer = ""  # 清空缓冲区

    # 处理最后一句（如果没以句号结尾）
    if buffer.strip():
        sentence = buffer.strip()
        tts_sound(tts,sentence,"zh",False)

# 图像线程安全队列（推荐）
img_queue = queue.Queue(maxsize=1)
image_app = FastAPI(title="Image Receiver")
@image_app.post("/api/process_image")
async def receive_image(request: Request):
    # 1. 解析JSON请求体（替代结构体，直接读取）
    req_body = await request.json()

    # 2. 校验必填字段是否存在
    required_fields = ["image", "width", "height"]

    # 3. 提取字段并校验类型（width/height需为数字）
    image_data = req_body["image"]
    width = req_body["width"]
    height = req_body["height"]

    # 5. 解码base64（去除前缀后）
    image_base64 = image_data.split(',')[1]  # 去掉前缀部分
    image_bytes = base64.b64decode(image_base64)

    # 6. 验证PNG图像有效性
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # img_queue.put(img)
    # 可选：保存接收到的图像
    img.save('received_image.png', format="PNG")
    # 校验传入的宽高和实际图像宽高是否一致（可选，根据GUI需求）
    actual_width, actual_height = img.size
    if int(width) != actual_width or int(height) != actual_height:
        print(f"⚠️ 传入宽高({width}x{height})与实际图像宽高({actual_width}x{actual_height})不一致")

    # 7. 打印调试信息（确认接收成功）
    client_ip = request.client.host
    print(f"\n=== 成功接收GUI发来的图像 ===")
    print(f"客户端IP：{client_ip}")
    print(f"传入宽高：{width}x{height}")
    print(f"实际宽高：{actual_width}x{actual_height}")

def start_img_receiver():
    """在后台线程启动HTTP接收服务"""
    uvicorn.run(image_app, host="0.0.0.0", port=50056, log_level="warning")
    print("Image receiver started."," host:","0.0.0.0"," port:",50056)
# 启动接收服务（daemon=True 确保随主进程退出）
threading.Thread(target=start_img_receiver, daemon=True).start()

def send_mask_to_gui(mask_image, gui):
    """
    将 mask 发送到 GUI

    Args:
        mask_image: PIL Image 对象（灰度模式，L mode）
        gui_url: GUI 服务器地址
    """
    gui_url="http://192.168.20.29:8000"
    # 转换为 Base64
    buffered = io.BytesIO()
    mask_image.save(buffered, format="PNG")
    mask_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    mask_data = f"data:image/png;base64,{mask_base64}"

    # 发送请求
    response = requests.post(
        f"{gui_url}{gui}",
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
    # cv2.namedWindow("output image", cv2.WINDOW_AUTOSIZE)
    out_image = None
    while True:

        try:
            text = stt_queue.get_nowait()
            start_time = time.time()
            prompt = get_inst_plan(text)
            messages = input_messages_build(prompt)
            response = client.chat.completions.create(
                model='Qwen3-VL-8B-Instruct',
                messages=messages,
                temperature=0.2,
                extra_body={"mm_processor_kwargs":{"fps": [1]}},
                stream=False,
            )
            choices = response.choices[0].message.content
            print('选择操作耗时：',time.time()-start_time,'秒')
            print("选择的操作类型：", choices)
            image = Image.open('received_image.png').convert('RGB')
            if choices=='C':
                chat_with_llm('Qwen3-VL-8B-Instruct',text)
                continue
            elif choices=='S':
                chat_with_llm('Qwen3-VL-8B-Instruct',text,image)
                continue
            # image = send_signal_get_image()
            # image = img_queue.get_nowait()
            # image.save('received_image.png', format="PNG")
            result = extract_objects('Qwen3-VL-8B-Instruct',text)
            print('输入语句分割结果为',result)
            # out_image = image.copy()
            text1 = result['被拿物体']
            start_time = time.time()
            # 获取包围框
            input_box = get_bounding_box('Qwen3-VL-8B-Instruct',text1,image)
            print("包围框时间为",time.time()-start_time)
            if input_box!=[]:
                start_time = time.time()
                x1, y1, x2, y2 = input_box
                # -------------------------- SAM3推理 --------------------------
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    inference_state = sam_predictor.set_image(image)
                    masks, scores, _ = sam_predictor.predict(
                        box=input_box,
                        multimask_output=False  # 单mask更高效
                    )
                print("框掩码预测时间为",time.time()-start_time)
                mask = masks[np.argmax(scores)]
                # 掩码转二值图
                mask_np = mask.astype(np.uint8) * 255

                print(f"1. 形状（shape）：{mask_np.shape} → (高度, 宽度)（二值掩码无通道）")
                img_np = np.array(image)
                # 2. 转换通道顺序：RGB → BGR（OpenCV显示的核心要求）
                img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                print(f"1. 形状（shape）：{img_cv2.shape} → (高度, 宽度, 通道数)（BGR通道）")
                cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
                out_image = img_cv2
                mask_pil = Image.fromarray(mask_np)
                try:
                    send_mask_to_gui(mask_pil,'/external/receive_mask')
                except:
                    pass
            else:
                sentence = '未找到'+text1
                tts_sound(tts,sentence,"zh", False)
            # print("可视化")
            # cv2.namedWindow("detection window", cv2.WINDOW_AUTOSIZE)
                # 强制窗口置顶（某些Linux桌面环境支持）
            #     cv2.setWindowProperty("detection window", cv2.WND_PROP_TOPMOST, 1)
            #     cv2.imshow("detection window", mask_np)
            #     cv2.waitKey(100)

            # if out_image is not None:
            #     cv2.imshow("output image", out_image)
            #     cv2.waitKey(100)
            # continue
            text2 = result['放置容器']
            start_time = time.time()
            # 获取包围框
            input_box = get_bounding_box('Qwen3-VL-8B-Instruct',text2,image)
            print("包围框时间为",time.time()-start_time)
            if input_box!=[]:
                start_time = time.time()
                x1, y1, x2, y2 = input_box
                # -------------------------- SAM3推理 --------------------------
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    inference_state = sam_predictor.set_image(image)
                    masks, scores, _ = sam_predictor.predict(
                        box=input_box,
                        multimask_output=False  # 单mask更高效
                    )
                print("框掩码预测时间为",time.time()-start_time)
                mask = masks[np.argmax(scores)]
                # 掩码转二值图
                mask_np = mask.astype(np.uint8) * 255
                mask_pil = Image.fromarray(mask_np)
                try:
                    send_mask_to_gui(mask_pil,'/external/receive_place_target')
                except:
                    pass
            else:
                sentence = '未找到'+text2
                tts_sound(tts,sentence,"zh", False)
                # -------------------------- 关键步骤：显示并强制刷新 --------------------------

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

        #     key = cv2.waitKey(1)& 0xFF
        #     # 按q或者ESC退出
        #     if key == ord('q') or key == 27:
        #         break

if __name__ == "__main__":
        main()
