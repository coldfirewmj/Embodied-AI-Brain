import os
import sys
import time
import numpy as np
from textwrap import dedent
import sounddevice as sd
import sherpa_onnx
print(sherpa_onnx.__file__)


model_dir = "/home/aiot/mingjuwang/Models/kokoro-multi-lang-v1_1"
start_time = time.time()
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
print(f"🎉 加载kokoro模型耗时: {time.time() - start_time:.2f} 秒")

original_cwd = os.getcwd()
sys.path.append(original_cwd+'/RealTimeSTT')
from RealtimeSTT import AudioToTextRecorder
def get_in_sounddevice_index(target_name='ReSpeaker'):
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        if target_name in device['name'] and device['max_input_channels'] > 0:
            print("麦克风:",device['name'],' index:',idx)
            return idx
    return None
input_sound_index = get_in_sounddevice_index()
# 1. 保存当前工作目录
LOCAL_MODEL_PATH = os.path.dirname(original_cwd)+"/Models/faster-whisper-large-v3-turbo"
# 确保路径存在
# 创建录音+识别器（关键配置）
start_time = time.time()
recorder = AudioToTextRecorder(
    # 模型大小：/base
    model=LOCAL_MODEL_PATH,    
    silero_vad_path= os.path.dirname(original_cwd)+'/Models/snakers4_silero-vad_master',
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
recorder.stop()
start_time= time.time()

import re
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

def get_out_ounddevice_index(target_name='USB2.0 Device'):
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        if target_name in device['name'] and device['max_input_channels'] > 0:
            print("扬声器:",device['name'],' index:',idx)
            return idx
    return None
output_device_index = get_out_ounddevice_index()

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

from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
sentence_endings = r'[。！？!?；;…\n]'  # 中英文句尾符号


def main():
    while True:
        recorder.start()
        text = preprocess_voice_text(recorder.text())
        recorder.stop()
        messages = [{"role": "user", "content": text}]
        response = client.chat.completions.create(
            model='Qwen2.5-VL-7B-Instruct',
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
                print(token, end="", flush=True)  # 终端也显示

                # 检查是否遇到完整句子结尾
                if re.search(sentence_endings, token):
                    # 去掉末尾空白
                    sentence = buffer.strip()
                    if sentence:
                        print("\n🔊 正在朗读此句...")
                        try:
                            audio = tts.generate(sentence, sid=10, speed=1.0)
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
                audio = tts.generate(sentence, sid=10, speed=1.0)
                if len(audio.samples) > 0:
                    play_audio(audio)
            except Exception as e:
                print(f"\n⚠️ TTS 错误: {e}")

if __name__ == "__main__": 
    voice_control = True
    main()