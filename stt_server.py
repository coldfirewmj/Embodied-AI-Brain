import os
import sys
import time
import re
import threading
import requests
import sounddevice as sd

# ===== 路径与导入 =====
original_cwd = os.getcwd()
sys.path.append(os.path.join(original_cwd, 'audio', 'RealTimeSTT'))
from RealtimeSTT import AudioToTextRecorder

import psutil
# ===== Jetson 专属优化：系统调度 =====
def optimize_jetson_process():
    """针对 Orin AGX 的调度优化，确保不被其他进程挤占"""
    p = psutil.Process(os.getpid())
    # 1. 提升 CPU 优先级 (Linux 最高为 -20)
    try:
        p.nice(-20)
    except: pass
    # 2. 核心绑定 (Affinity)
    # Orin AGX 有 12 核，0-3 通常处理系统中断，我们将 STT 绑定到 4-11 核
    try:
        p.cpu_affinity(list(range(4, 12)))
    except: pass
    # 3. 设置实时调度策略 (需要 sudo 权限)
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(99))
        print("🚀 已开启 SCHED_FIFO 实时调度优先级")
    except:
        print("⚠️ 提示: 请使用 'sudo' 运行以获得最高调度权限")

# ===== 麦克风设备查找 =====
def get_input_device_index(target_name='PnP'):
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if target_name in dev['name'] and dev['max_input_channels'] > 0:
            print(f"🎤 Using mic: {dev['name']} (index={idx})")
            return idx
    return None
input_sound_index = get_input_device_index()

# ===== 文本预处理 =====
def preprocess_voice_text(text: str):
    if not text or not text.strip():
        return None
    text = text.strip()
    text = re.sub(r"^(嗯|啊|哦|呃|哎|哈|嘿|喂)\s*", "", text)
    return text[:2000] if text else None

# ===== 发送到本地 /v1 =====
def send_to_local_api(text: str):
    try:
        requests.post("http://0.0.0.0:28184/v1", json={"text": text}, timeout=1)
    except Exception as e:
        print(f"❌ 发送到主进程失败: {e}")
        pass

# ===== STT 监听线程 =====
def stt_worker(recorder):
    recorder.start()
    while True:
        try:
            raw = recorder.text()
            cleaned = preprocess_voice_text(raw)
            if cleaned:
                print(  f"🗣️ Recognized: {cleaned}")
                threading.Thread(target=send_to_local_api, args=(cleaned,), daemon=True).start()
        except Exception as e:
            print(f"💥 STT error: {e}")
            time.sleep(0.5)

# ===== 主程序 =====
if __name__ == "__main__":
    optimize_jetson_process()

    # 初始化模型
    model_path = os.path.join(original_cwd, "Models", "faster-whisper-large-v3-turbo")
    vad_path = os.path.join(original_cwd, "audio", "snakers4_silero-vad_master")

    print("⏳ Loading Whisper model...")
    start = time.time()
    recorder = AudioToTextRecorder(
        model=model_path,
        silero_vad_path=vad_path,
        language="zh",
        compute_type="float16",
        device="cuda",
        use_microphone=True,
        sample_rate=16000,
        initial_prompt="以下是普通话的句子。",
        input_device_index=input_sound_index,
    )
    print(f"✅ Model loaded in {time.time() - start:.2f}s")

    stt_worker(recorder)