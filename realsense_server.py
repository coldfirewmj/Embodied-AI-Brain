import os
# 加载realsense工作目录
print('开始加载realsense工作目录')
import pyrealsense2 as rs
import cv2
import numpy as np
# D455相机初始化
pipeline = rs.pipeline()
config = rs.config()
# 配置彩色流（SAM3处理RGB图像）：分辨率640x480，帧率30
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
align = rs.align(rs.stream.color)
pipeline.start(config)

def get_realsense_image():
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    # if not color_frame or not depth_frame :
    #     continue
    image = np.asanyarray(color_frame.get_data())
    # out_image = image.copy()
    depth_image = np.asanyarray(depth_frame.get_data())
    # 后续可安全地用此 转换为点云
    # depth_intrin = depth_frame.profile.as_video_stream_profile().get_intrinsics()
    # point = rs.rs2_deproject_pixel_to_point(depth_intrin, [u, v], depth_image[u,v] / 1000.0)
    return image,depth_image