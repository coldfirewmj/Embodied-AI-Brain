import os
import sys
import sherpa_onnx
model_dir = 
sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens=model_dir + "/tokens.txt",
    encoder=model_dir + "/encoder.onnx",
    decoder=model_dir + "/decoder.onnx",
    joiner=model_dir + "/joiner.onnx",
    num_threads=4,
    sample_rate=16000,
    feature_dim=80,
    enable_endpoint_detection=True,
    rule1_min_trailing_silence=2.4,
    rule2_min_trailing_silence=1.2,
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
