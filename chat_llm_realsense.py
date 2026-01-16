import os
import sys
original_cwd = os.getcwd()
import time
from textwrap import dedent
import threading
import requests
from urllib.parse import urljoin
# 用于控制录音线程
recording_lock = threading.Lock()
# 标记当前是否允许录音
# 创建录音+识别器（关键配置）
import sounddevice as sd
# 获取音频设备的编号
def get_in_sounddevice_index(target_name='PnP'):
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        if target_name in device['name'] and device['max_input_channels'] > 0:
            print("麦克风:",device['name'],' index:',idx)
            return idx
    return None
input_sound_index = get_in_sounddevice_index()
def get_out_ounddevice_index(target_name='default'):
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        if target_name in device['name'] and device['max_output_channels'] > 0:
            print("扬声器:",device['name'],' index:',idx)
            return idx
    return None
output_device_index = get_out_ounddevice_index()

class TTSAgent:

    def __init__(self, host_url):
        print(f"TTSAgent: host_url {host_url}")
        self.host_url = host_url

    def run(self, input_dict_str: str) -> str:
        data = {"task": input_dict_str}
        try:
            start_time = time.time()
            resp = requests.post(urljoin(self.host_url, 'exec'), data=data, timeout=10)
            duration = time.time() - start_time
            #print(f"TTSAgent: post_duration {duration:.3f}")
            try:
                resp_dict = json.loads(resp.text)
                out_text = resp_dict['out_text']
                #print(f"Recv: out_text {out_text}")
            except:
                out_text = ""
        except requests.exceptions.Timeout as e:
            print('TTSAgent: Timeout')
            out_text = ""
        return out_text

def create_tts_agent(host_url: str = None):
    host_url = host_url or os.getenv('RABBITBOT_TTS_AGENT_URL', 'http://127.0.0.1:8001')
    return TTSAgent(host_url)

tts = create_tts_agent("http://localhost:28185/v1")

# 加载whisper模型
sys.path.append(original_cwd+'/audio/RealTimeSTT')
from RealtimeSTT import AudioToTextRecorder
stt_path = original_cwd+"/Models/faster-whisper-large-v3-turbo"
print('开始whisper加载模型')
start_time = time.time()
WAKE_WORD = "你好小域"
SENSITIVITY = 0.5
recorder = AudioToTextRecorder(
    # 模型大小：/base
    model=stt_path,
    silero_vad_path= original_cwd+'/Models/snakers4_silero-vad_master',
    # 强制中文（提高准确率）
    language="zh",
    compute_type="float16",
    device="cuda",
    # 是否使用麦克风输入，False表示使用回调函数输入
    use_microphone=True,
    # 添加唤醒词
    # wake_words=WAKE_WORD,
    # wake_words_sensitivity=SENSITIVITY,
    initial_prompt="以下是普通话的句子。",
    # 如果你知道麦克风设备索引，取消注释下一行：
    input_device_index=input_sound_index,  # 替换为你的麦克风索引（通过 sounddevice_devices.py 获取）
)
print(f"🎉 加载whisper模型耗时: {time.time() - start_time:.2f} 秒")
# 过滤文本
def preprocess_voice_text(text):
    if not text:
        return None
    text = text.strip()
    if not text or text.isspace():
        return None
    text = re.sub(r"^(嗯|啊|哦|呃|哎)\s*", "", text)
    text = text[:2000]
    print(text)
    return text
start_time= time.time()

# 切换到sam2工作目录，加载sam
print('开始加载sam2模型')
sam_path = original_cwd+'/vision/sam2'
sys.path.append(sam_path)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import re
from PIL import Image
CHECKPOINT_SAM = os.path.join(sam_path,"sam2_checkpoints", "sam2.1_hiera_small.pt")
# CONFIG_SAM前面必须加'/'
CONFIG_SAM = '/'+sam_path+"/sam2/configs/sam2.1/sam2.1_hiera_s.yaml"
# Load the model
sam2_model = build_sam2(CONFIG_SAM, CHECKPOINT_SAM, device="cuda")
sam_predictor = SAM2ImagePredictor(sam2_model)
print(f"🎉 加载SAM模型耗时: {time.time() - start_time:.2f} 秒")

# 加载realsense工作目录
print('开始加载realsense工作目录')
import pyrealsense2 as rs
import cv2
import numpy as np
# D455相机初始化
pipeline = rs.pipeline()
config = rs.config()
# 配置彩色流（SAM3处理RGB图像）：分辨率640x480，帧率30
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
align = rs.align(rs.stream.color)
pipeline.start(config)

# 启动大模型客户端
print('开始连接大模型服务')
start_time= time.time()
import base64
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
print(f"🎉 加载大模型服务耗时: {time.time() - start_time:.2f} 秒")
start_time= time.time()

# 合成messages
def box_messages_build(prompt, image):
    # --- 修改部分：先编码为 png 格式 ---
    # image 是从 realsense 获取的 BGR 数组
    success, buffer = cv2.imencode('.png', image)
    if not success:
        raise ValueError("无法编码图像")

    # 转换为 base64
    image_b64 = base64.b64encode(buffer).decode("utf-8")
    # 构建图像信息
    image_contents = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_b64}"
                },
            }
    # 构建文本信息
    text_content = {
        "type": "text",
        "text": prompt
    }
    return [
    {"role": "system", "content":
    "You must output ONLY a JSON array of four integers: [x1, y1, x2, y2]. No explanation. No markdown. Just the array."},
        {
            "role": "user",
            "content":   [text_content, image_contents],
        }]
# 得到回答并解析出包围框
def get_respose_box(model,messages):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        extra_body={"mm_processor_kwargs":{"fps": [1]}},
        stream=False,
    )

    entity = response.choices[0].message.content.strip('```json\n').strip('```')
    # stream_message = ""
    # for chunk in response:
    #     try:
    #         stream_message += chunk.choices[0].delta.content
    #         print(chunk.choices[0].delta.content, end='')
    #     except AttributeError as e:
    #         if "'str' object has no attribute 'choices'" in str(e):
    #             pass
    #         else:
    #             raise
    # entity = stream_message.strip('```json\n').strip('```')
    print("entity：", entity)
    entity_bbox = entity
    print("test is exec!!!!!!!", entity_bbox)
    return entity_bbox
import json
def main():
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame :
            continue
        image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        text = preprocess_voice_text(recorder.text())
        prompt = dedent(f"""\
            你是一名识别专家。你的任务是从机器人视角，仅根据用户**指定物体**的位置。
            请首先判断物体是否在画面内，然后输出其包围盒：

            **输出要求：**
            请直接输出识别到的用户要求的物体信息“{text}”的包围框，请仔细分辨物体的形状和颜色，不要包含任何文字信息，包括plaintext。
            """)
        # prompt = dedent(""" 请回答你是谁""")
        # 后续可安全地用此 转换为点云
        depth_intrin = depth_frame.profile.as_video_stream_profile().get_intrinsics()
        # 获取包围框
        entity_bbox = get_respose_box('Qwen2.5-VL-7B-Instruct',box_messages_build(prompt,image))
        if entity_bbox!=[]:
            # input_box = list(map(int, entity_bbox[0].split(',')))
            input_box = json.loads(entity_bbox)
            x1, y1, x2, y2 = input_box
            input_box = np.array(input_box)
            print(input_box)
            # 转换为RGB格式
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # SAM3需要RGB格式
            # -------------------------- SAM3推理 --------------------------
            # Load an image
            img_pil = Image.fromarray(img_rgb)
            inference_state = sam_predictor.set_image(img_pil)
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
            # 获得该物体的xyz坐标（包围框内掩码的像素点对应的xyz求平均）
            # for u in range(x1,x2+1):
            #     for v in range(y1,y2+1):
            #         depth_val = depth_image[v,u]
            #         if(depth_val > 0 and mask[v,u]):
            #             [x,y,z] = rs.rs2_deproject_pixel_to_point(
            #                 depth_intrin, [u, v], depth_val / 1000.0)
            #             # print('物体坐标是：x:',x,'y:',y,'z:',z)
            #             break

        out_image = image.copy()
        # 显示结果
        sentence = '物体已显示'
        cv2.imshow("SAM3 mask in box", out_image)

        key = cv2.waitKey(1)& 0xFF
        # 按q或者ESC退出
        if key == ord('q') or key == 27:
            break

if __name__ == "__main__":
        main()
