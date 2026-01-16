import pyrealsense2 as rs
import numpy as np
import cv2

# 初始化RealSense管道
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


# 创建对齐对象 (将深度对齐到彩色，为了生成彩色点云)
align_to = rs.stream.color
align = rs.align(align_to)
# ⭐️ 新增：创建颜色映射器 (用于将深度值转为彩色图像)
colorizer = rs.colorizer()
try:
    profile = pipeline.start(config)
    # 获取内参
    depth_intrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

    while True:
        # 等待一对颜色和深度帧
        frames = pipeline.wait_for_frames()
        # 对齐帧 (关键步骤：让深度图的像素和彩色图的像素坐标对应)
        aligned_frames = align.process(frames)
        # 获取对齐后的深度帧和彩色帧
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # 将图像转换为numpy数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 假设这里我们有一个目标检测的结果 - 例如一个包围框
         # x_min, y_min, x_max, y_max 示例值
        bounding_box = [100, 100, 200, 200] 
        # 创建一个与深度图相同大小的mask
        mask = np.zeros_like(depth_image, dtype=np.uint8)  

        # 提取感兴趣区域的深度信息
        masked_depth = depth_image * mask

        # ⭐️ 新增：应用颜色映射器到对齐后的深度帧
        colorized_depth = colorizer.process(depth_frame)
        # ⭐️ 深度图现在也是 BGR 格式的 8 位图像，可以直接用于 imshow
        depth_colormap = np.asanyarray(colorized_depth.get_data())
        # 根据包围框框起来深度图
        bounding_depth = depth_image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]  
        # 根据包围框框起来mask
        
        # 后续可安全地用此 bounding_depth 转换为点云
        depth_intrin = depth_frame.profile.as_video_stream_profile().get_intrinsics()
        points = []
        
        count = 0
        total_x, total_y, total_z=[0.0,0.0,0.0]
        for v in range(bounding_depth.shape[0]):
            for u in range(bounding_depth.shape[1]):
                depth_val = bounding_depth[v, u]
                if depth_val > 0:
                    # 注意：u, v 是相对于 ROI 的，需转换回全局坐标
                    global_u = bounding_box[0] + u
                    global_v = bounding_box[1] + v
                    point = rs.rs2_deproject_pixel_to_point(
                        depth_intrin, [global_u, global_v], depth_val / 1000.0)
                    [x,y,z]= point
                    points.append(point)
                    count+=1
                    total_x+=x
                    total_y+=y
                    total_z+=z

        print(f"Extracted {len(points)} points")
        if (len(points)>0):
            print(points[0])
        print(total_x/count,' ; ',total_y/count,' ; ',total_z/count)

        # 显示彩色图像以便可视化
        color_image_vis = cv2.rectangle(color_image.copy(), (bounding_box[0], bounding_box[1]), 
                                        (bounding_box[2], bounding_box[3]), (0, 255, 0), 2)
        images = np.hstack((color_image, depth_colormap))
        cv2.imshow('Color Image', images)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()