import sounddevice as sd
# print(sd.query_devices())
# 2. 筛选仅音频输入设备
print("=== 可用音频输入设备（麦克风）===")
input_devices = []
for idx, device in enumerate(sd.query_devices()):
    if device['max_input_channels'] > 0 and 'NVIDIA' not in device['name']:
        print(device)
        if 'PnP' in device['name']:  # 输入通道数大于0即为输入设备
            input_devices.append((idx, device))
            print(f"\n输入设备索引：{idx}")
            print(f"设备名称：{device['name']}")
            print(f"最大输入通道数：{device['max_input_channels']}")
            print(f"默认采样率：{device['default_samplerate']} Hz")

# 若没有可用输入设备，给出提示
if not input_devices:
    print("未检测到可用音频输入设备，请检查麦克风是否连接或驱动是否正常！")

print("=== 可用音频输出设备（扬声器）===")
output_devices = []
for idx, device in enumerate(sd.query_devices()):
    if device['max_output_channels'] > 0 and 'NVIDIA' not in device['name']:
        print(device)
        if 'default' in device['name']:  # 输出通道数大于0即为输出设备
            output_devices.append((idx, device))
            print(f"\n输出设备索引：{idx}")
            print(f"设备名称：{device['name']}")
            print(f"最大输出通道数：{device['max_output_channels']}")
            print(f"默认采样率：{device['default_samplerate']} Hz")
