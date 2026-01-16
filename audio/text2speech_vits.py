import sherpa_onnx
print(sherpa_onnx.__file__)
import sounddevice as sd
import numpy as np
import os
import wave
import time

# 请确保此路径下是模型位置
MODEL_BASE_DIR = "/home/aiot/mingjuwang/Models/vits-zh-aishell3"
def get_sounddevice_index(target_name='USB2.0 Device'):
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        if target_name in device['name'] and device['max_input_channels'] > 0:
            return idx
    return None
output_device_index = get_sounddevice_index()
# print("使用的输出设备索引:", output_device_index)
# 缓冲区
sentence_endings = r'[。！？!?；;…\n]'  # 中英文句尾符号
def play_audio(audio, gain=5.0):
    samples = np.array(audio.samples, dtype=np.float32) * gain
    samples = np.clip(samples, -1.0, 1.0)
    from scipy.signal import resample
    # 重采样到 48000 Hz
    if audio.sample_rate != 48000:
        num_samples = int(len(samples) * 48000 / audio.sample_rate)
        samples = resample(samples, num_samples)
    sd.play(samples, samplerate=48000, device=output_device_index)
    sd.wait()
import re
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
messages = [{"role": "user", "content": "请介绍一下人工智能的发展历史。"}]
def run_sherpa_tts():
    start_time = time.time()
    # 1. 配置
    vits_model_config = sherpa_onnx.OfflineTtsVitsModelConfig(
        model=os.path.join(MODEL_BASE_DIR, "vits-aishell3.onnx"),
        lexicon=os.path.join(MODEL_BASE_DIR, "lexicon.txt"),
        tokens=os.path.join(MODEL_BASE_DIR, "tokens.txt"),
        data_dir="", # 使用官方下载的路径
        noise_scale=0.667,
        noise_scale_w=0.8,
        length_scale=1.0  # 稍微调慢一点，便于听清
    )

    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=vits_model_config,
            # num_threads=14,
            debug=False,
            provider="cuda"
        )
    )
    
    tts = sherpa_onnx.OfflineTts(config)
    print('配置模型耗时: {:.2f} 秒'.format(time.time() - start_time))
    start_time = time.time()
    # 2. 生成
    text = "床前明月光，疑是地上霜。"
    # print(f"正在生成: {text}")
    response = client.chat.completions.create(
        model='Qwen2.5-VL-7B-Instruct',
        messages=messages,
        temperature=0.2,
        extra_body={"mm_processor_kwargs":{"fps": [1]}},
        stream=True,
    )
    # 尝试改变 sid 看看是否有变化
    # audio = tts.generate(text, sid=1, speed=1.0)
    buffer = ""
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            buffer += token
            print(token, end="", flush=True)  # 终端也显示

            # 检查是否遇到完整句子结尾
            if re.search(sentence_endings, token):
                # 去掉末尾空白
                sentence = buffer.strip()
                if sentence:
                    print("\n🔊 正在朗读此句...")
                    try:
                        audio = tts.generate(sentence, sid=94, speed=1.0)
                        print("=====================================")
                        if len(audio.samples) > 0:
                            print("+++++++++++++++++++++++++++++++++++++++")
                            play_audio(audio)
                    except Exception as e:
                        print(f"\n⚠️ TTS 错误: {e}")
                    buffer = ""  # 清空缓冲区

    # 处理最后一句（如果没以句号结尾）
    if buffer.strip():
        sentence = buffer.strip()
        print(f"\n🔊 朗读最后一句: {sentence}")
        try:
            audio = tts.generate(sentence, sid=94, speed=1.0)
            if len(audio.samples) > 0:
                play_audio(audio)
        except Exception as e:
            print(f"\n⚠️ TTS 错误: {e}")
    print('语音生成耗时: {:.2f} 秒'.format(time.time() - start_time))
    print("Audio type:", type(audio))
    print("Audio :", audio)

    if audio and len(audio.samples) > 0:
        samples = np.array(audio.samples).flatten()
        actual_duration = len(samples) / audio.sample_rate
        print(f"实际生成音频长度: {actual_duration:.2f} 秒")
        
        # 3. 播放
        # print("播放中...")
        # sd.play(np.array(audio.samples)*8.0, samplerate=audio.sample_rate, device=output_device_index)
        # sd.wait() # 核心：必须等待播放完成

        # 4. 保存
        # with wave.open("debug_output.wav", "wb") as wf:
        #     wf.setnchannels(1)
        #     wf.setsampwidth(2)
        #     wf.setframerate(audio.sample_rate)
        #     data = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
        #     wf.writeframes(data.tobytes())
    else:
        print("生成失败。")

if __name__ == "__main__":
    run_sherpa_tts()