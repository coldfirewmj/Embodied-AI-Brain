import sherpa_onnx
print(sherpa_onnx.__file__)
import sounddevice as sd
import numpy as np
import os
import wave
import time
import soundfile as sf

model_dir = "/home/aiot/mingjuwang/Models/kokoro-multi-lang-v1_1"
def get_sounddevice_index(target_name='USB2.0 Device'):
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        if target_name in device['name'] and device['max_output_channels'] > 0:
            return idx
    return None
output_device_index = get_sounddevice_index()
text = "Hello，欢迎使用 kokoro-multi-lang-v1_1 模型，当前温度 25℃，英文测试：This is a test."
rule_fsts = [
    os.path.join(model_dir, "phone-zh.fst"),
    os.path.join(model_dir, "date-zh.fst"),
    os.path.join(model_dir, "number-zh.fst"),
]
config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=os.path.join(model_dir,"model.onnx"),
            voices=os.path.join(model_dir,"voices.bin"),
            tokens=os.path.join(model_dir,"tokens.txt"),
            data_dir=os.path.join(model_dir,"espeak-ng-data"),
            dict_dir=os.path.join(model_dir,"dict"),
            lexicon=model_dir+"/lexicon-zh.txt,"+model_dir+"/lexicon-us-en.txt",
        ),
        num_threads=2,
        provider="cuda",  # 有GPU则改为"cuda"（需安装onnxruntime-gpu）
    ),
    rule_fsts=model_dir+"/phone-zh.fst,"+
            model_dir+"/date-zh.fst,"+
            model_dir+"/number-zh.fst",
)

# 2. 创建TTS引擎
tts = sherpa_onnx.OfflineTts(config)
import re
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
messages = [{"role": "user", "content": "请介绍一下人工智能的发展历史。"}]
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
def main():
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
                        audio = tts.generate(sentence, sid=1, speed=1.0)
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
            audio = tts.generate(sentence, sid=1, speed=1.0)
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
    main()