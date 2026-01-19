import os
import sys
import time
import threading
import heapq
import numpy as np
import torch
import sounddevice as sd
from scipy.signal import resample_poly
import uvicorn
from fastapi import FastAPI, Request

# ===== 配置 =====
KOKORO_OFFICIAL_SR = 24000.0
TARGET_SR = 44100
VOICE_NAME = "zm_yunxi"

# 加载模型 (代码保持不变)
original_cwd = os.getcwd()
sys.path.append(os.path.join(original_cwd, 'audio', 'kokoro'))
from kokoro import KPipeline, KModel

model = KModel(config=original_cwd + "/Models/Kokoro-82M/config.json",
              model=original_cwd + "/Models/Kokoro-82M/kokoro-v1_0.pth",
              disable_complex=True)
pipeline = KPipeline(lang_code='z', device="cuda", model=model)

# ===== 变量控制 =====
raw_audio_heap = [] 
interrupt_event = threading.Event()
# last_request_time记录上次请求时间
last_request_time = 0
# current_session_id表示每一轮对话的轮数
current_session_id = 0 
# global_seq_id用于为每个请求分配唯一的序号
global_seq_id = 0    
expected_seq_id = 0  
LOCK = threading.Lock()

# 查找设备
def get_output_device_index(target_name='default'):
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if target_name in dev['name'] and dev['max_output_channels'] > 0:
            return idx
    return None
output_sound_index = get_output_device_index()

# ===== 1. 播放消费者线程 (修正逻辑) =====
def tts_playback_worker():
    global expected_seq_id, raw_audio_heap, current_session_id
    print("🚀 播放线程已就绪，监听 28185...")
    
    while True:
        audio_to_play = None
        
        with LOCK:
            if raw_audio_heap:
                # 堆顶数据: (sid, seq_id, audio_data)
                # heapq 只比较前两个元素，直到确定唯一性
                top_sid, top_seq_id, _ = raw_audio_heap[0]
                
                # 情况 A: 旧 Session 的垃圾数据，直接清除
                if top_sid < current_session_id:
                    heapq.heappop(raw_audio_heap)
                    continue
                
                # 情况 B: 轮到当前序号播放
                if top_sid == current_session_id and top_seq_id == expected_seq_id:
                    _, _, audio_to_play = heapq.heappop(raw_audio_heap)
                    # print(f"▶️ 提取成功: Session {top_sid}, Seq {top_seq_id}")
        
        if audio_to_play is not None:
            print(f"🔊 正在播放 Seq: {expected_seq_id}")
            sd.play(audio_to_play, samplerate=TARGET_SR, device=output_sound_index)
            
            duration = len(audio_to_play) / TARGET_SR
            start_t = time.time()
            while time.time() - start_t < duration:
                if interrupt_event.is_set():
                    sd.stop()
                    break
                time.sleep(0.01)
            
            with LOCK:
                expected_seq_id += 1
        else:
            # 没轮到或没数据，短休眠
            time.sleep(0.02)

threading.Thread(target=tts_playback_worker, daemon=True).start()

# ===== 2. 推理生产者任务 =====
def inference_task(text, sid, seq_id):
    try:
        # print(f"⚙️ 推理开始: [ID {seq_id}] {text[:10]}...")
        generator = pipeline(text, voice=VOICE_NAME, speed=1.0)
        
        for _, _, audio in generator:
            if sid != current_session_id or interrupt_event.is_set():
                return
            
            # 预处理数据
            wav = audio.numpy() if hasattr(audio, 'numpy') else audio
            if isinstance(wav, torch.Tensor):
                wav_data = wav.view(-1).cpu().numpy()
            else:
                wav_data = wav
            
            resampled = resample_poly(wav_data, up=int(TARGET_SR), down=int(KOKORO_OFFICIAL_SR)).astype(np.float32)
            
            # 【关键修复】: 存入堆。
            # 为了防止 heapq 比较 NumPy 数组，我们将数据放在列表的第三位
            # Python 的比较规则是：先比 sid, 再比 seq_id, 只要 seq_id 不同就不再往后比。
            with LOCK:
                if sid == current_session_id:
                    heapq.heappush(raw_audio_heap, (sid, seq_id, resampled))
                    # print(f"✅ 推理完成入堆: Seq {seq_id}")
                    
    except Exception as e:
        print(f"❌ 推理异常: {e}")

# ===== 3. FastAPI 接口 =====
app = FastAPI()

@app.post("/v1")
async def receive_tts_text(request: Request):
    global last_request_time, interrupt_event, current_session_id, global_seq_id, expected_seq_id, raw_audio_heap
    
    try:
        body = await request.json()
        text = body.get("text", "").strip()
        if not text: return {"status": "empty"}

        current_time = time.time()
        
        with LOCK:
            time_diff = current_time - last_request_time
            
            # 5秒打断逻辑
            if time_diff > 5.0:
                print(f"\n⚡ 打断并重置对话 (间隔 {time_diff:.1f}s)")
                interrupt_event.set()
                sd.stop()
                
                current_session_id += 1 
                global_seq_id = 0      
                expected_seq_id = 0    
                raw_audio_heap = []    
                
                time.sleep(0.05) 
                interrupt_event.clear()
            
            target_sid = current_session_id
            target_seq = global_seq_id
            global_seq_id += 1
            last_request_time = current_time

            # 启动推理
            threading.Thread(target=inference_task, args=(text, target_sid, target_seq), daemon=True).start()
            
        return {"status": "ok", "seq": target_seq}

    except Exception as e:
        return {"status": "error", "msg": str(e)}

if __name__ == "__main__":
    generator = pipeline("发音服务已启动", voice=VOICE_NAME, speed=1.0)
    for _, _, audio in generator:
        wav = audio.numpy()
        if isinstance(wav, torch.Tensor):
            wav_data = wav.view(-1).cpu().numpy()
        else:
            wav_data =wav
        data_resampled = resample_poly(wav_data, up=44100, down=KOKORO_OFFICIAL_SR)
        # print("▶️ 播放音频...")
        sd.play(data_resampled, samplerate=44100)
    uvicorn.run(app, host="0.0.0.0", port=28185, log_level="warning")