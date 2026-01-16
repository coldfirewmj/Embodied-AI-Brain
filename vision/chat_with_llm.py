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

from textwrap import dedent
import base64
import requests
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

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
    answer = response.choices[0].message.content
    return answer

def main():
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    # if not color_frame or not depth_frame :
    #     break
    image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    text = "请说出来你看到了什么"
    prompt = dedent(f"""\
        你是一名识别专家。你的任务是从机器人视角，仅根据**摄像头镜头**的图片。
        对于镜头中的物体进行分析，并根据用户的问题进行相应的回答：

        **输出要求：**
        请根据用户的问题：“{text}”的包围框，请仔细分辨物体的形状和颜色，回答输出需要简略。
        """)
    answer = get_respose_box('Qwen2.5-VL-7B-Instruct',box_messages_build(prompt,image))
    print(f"回答内容: {answer}")
