# 本程序用来检测所有的输入channel，查看录音设备的info
import pyaudio
import sounddevice as sd

def list_pyaudio():
    p = pyaudio.PyAudio()

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if 'ReSpeaker' in info['name']:
            print('===========================================================================')
            print(info)
            print('===========================================================================')
            if (info['maxInputChannels']>0):
                print(f"{i}: {info['name']} (host API: {info['hostApi']}, max input channels: {info['maxInputChannels']})")
            # elif (info['maxOutputChannels']>0):
                # print(f"{i}: {info['name']} (host API: {info['hostApi']}, max output channels: {info['maxOutputChannels']})")

def get_sounddevice_index(target_name='ReSpeaker'):
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        if target_name in device['name'] and device['max_input_channels'] > 0:
            return idx
    return None

if __name__ == "__main__":
    print('device index',get_sounddevice_index())
