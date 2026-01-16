import os
import sys
import time
import threading
import sounddevice as sd
import soundfile as sf
import uvicorn
from fastapi import FastAPI, Request

# 强制清理端口，防止 Errno 98
# os.system("sudo fuser -k 28185/tcp > /dev/null 2>&1")

# ===== 扬声器查找 =====
def get_output_device_index(target_name='default'):
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if target_name in dev['name'] and dev['max_output_channels'] > 0:
            return idx
    return None

output_sound_index = get_output_device_index()
print('output_sound_index is:',output_sound_index)

# ===== Kokoro 加载 =====
original_cwd = os.getcwd()
sys.path.append(os.path.join(original_cwd, 'audio', 'kokoro'))
from kokoro import KPipeline,KModel
model = KModel(config=original_cwd + "/models/Kokoro-82M/config.json",
                model=original_cwd + "/models/Kokoro-82M/kokoro-v1_0.pth",
                disable_complex=True)
pipeline = KPipeline(lang_code='z', device="cuda", model=model)
VOICE_NAME = "zm_yunxi"
KOKORO_OFFICIAL_SR = 24000.0

# 全局打断信号
interrupt_event = threading.Event()
import numpy as np
from scipy.signal import resample_poly
import torch
def play_audio_task(text):
    global interrupt_event
    interrupt_event.clear()
    
    # try:
    sd.stop() # 停止当前物理发声
    generator = pipeline(text, voice=VOICE_NAME, speed=1.0)
    print(generator)
    for _, _, audio in generator:
        if interrupt_event.is_set(): return
        wav = audio.numpy()
        if isinstance(wav, torch.Tensor):
            wav_data = wav.view(-1).cpu().numpy()
        else:
            wav_data =wav
        data_resampled = resample_poly(wav_data, up=44100, down=KOKORO_OFFICIAL_SR)
        # print("▶️ 播放音频...")
        sd.play(data_resampled, samplerate=44100)
        # sf.write('111.wav', audio, 24000)

        # 播放中检查打断
        duration = len(audio) / KOKORO_OFFICIAL_SR
        start_t = time.time()
        while time.time() - start_t < duration:
            if interrupt_event.is_set():
                sd.stop()
                return
            time.sleep(0.02)
    # except Exception as e:
    #     print(f"播放失败: {e}")

# ===== FastAPI =====
app = FastAPI()

@app.post("/v1")
async def receive_tts_text(request: Request):
    try:
        body = await request.json()
        # 兼容你的发送格式
        text = body.get("text", "")
        if text:
            print(f"📖 正在朗读: {text}")
            interrupt_event.set() # 触发打断
            time.sleep(0.05) # 给旧线程一点退出时间
            threading.Thread(target=play_audio_task, args=(text,), daemon=True).start()
            return {"status": "ok", "out_text": "playing"}
        return {"status": "empty"}
    except Exception as e:
        print(f"JSON 解析错误: {e}")
        return {"status": "error"}

if __name__ == "__main__":
    print("✅ TTS 服务已就绪 (端口 28185)")
    play_audio_task('发音服务已启动')
    uvicorn.run(app, host="0.0.0.0", port=28185, log_level="warning")