import os
import sys
original_cwd = os.getcwd()
sys.path.append(original_cwd+'/RealTimeSTT')
from RealtimeSTT import AudioToTextRecorder
import time
from textwrap import dedent
os.environ["JACK_NO_START_SERVER"] = "1"
os.environ["PYAUDIO_ALSA_ERROR_QUIET"] = "1"
# -------------------------- 改步骤为了使create_vlm_openai正常加载json --------------------------
# 1. 保存当前工作目录
LOCAL_MODEL_PATH = original_cwd+"/faster-whisper-large-v3-turbo"
# 创建录音+识别器（关键配置）
start_time = time.time()
recorder = AudioToTextRecorder(
    # 模型大小：/base
    model=LOCAL_MODEL_PATH,    
    # 强制中文（提高准确率）       
    language="zh",         
    compute_type="float16",   
    device="cuda",          
    # 可自定义回调 
    # on_recording_start=lambda: None,   
    # on_recording_stop=lambda: None,
    # on_transcription_start=lambda: None,
    initial_prompt="以下是普通话的句子。",
    # 如果你知道麦克风设备索引，取消注释下一行：
    input_device_index=38,  # 替换为你的麦克风索引（通过 sounddevice_devices.py 获取）
)
print(f"🎉 加载模型耗时: {time.time() - start_time:.2f} 秒")
# 2. 切换到 rabbitbot 项目根目录
project_root = "/home/aiot/fuchengjia/Projects/rabbitbot-dev-ros2"
os.chdir(project_root)
sys.path.insert(0, project_root)
from rabbitbot.provider import create_vlm_openai
print("加载大模型成功")
# 3. 切换回来原来的环境
# os.chdir(original_cwd)

def preprocess_voice_text(text):
    if not text:
        return None
    text = text.strip()
    if not text or text.isspace():
        return None
    import re
    text = re.sub(r"^(嗯|啊|哦|呃|哎)\s*", "", text)
    text = text[:2000]
    return text

def main():
    print("🎙️ 初始化 RealtimeSTT（使用本地 faster-whisper）...")
    vlm_openai = create_vlm_openai()
    vlm_openai.prompt = dedent(f"""\
            你是一名接线员，仅根据用户**输入文本**的位置。
            请进行对应的回复：

            **输出要求：**
            请直接针对用户的文字回复，不要包含任何其他文字。
            """)


    print("✅ 准备就绪！请对着麦克风说话（中文）...")
    print("-" * 60)

    try:
        while True:
            # 获取当前最新转录文本（非阻塞）
            text = preprocess_voice_text(recorder.text())
            if text:
                print(f"🗣️ 识别结果: {text}")
                messages = [{"role": "user", "content": text}]
                for _ in range(1):
                    json_content = vlm_openai.get_chat_response(
                        messages=messages,
                        extra_body={})
                # 清空已读文本，避免重复输出
                print(f"🤖 回复内容: {json_content}")
                recorder.text("")

    except KeyboardInterrupt:
        print("\n👋 退出程序")
    finally:
        recorder.shutdown()  # 释放资源
        
if __name__ == "__main__":
    main()
