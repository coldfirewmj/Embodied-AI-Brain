import torch
import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = YOLO("yolov8m.pt").to(device)
conf_threshold=0.5
infer_param = {
    'conf': conf_threshold,
    'device': device,
    'dnn': False
}
#from scipy.spatial.distance import cdist
# 1. 配置管道
pipe = rs.pipeline()
cfg = rs.config()

# ⚠️ 必须开启深度流才能计算点云
# D455 推荐分辨率: 848x480 或 640x480
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 2. 创建点云处理对象
pc = rs.pointcloud()

# 3. 创建对齐对象 (将深度对齐到彩色，为了生成彩色点云)
align_to = rs.stream.color
align = rs.align(align_to)
# ⭐️ 新增：创建颜色映射器 (用于将深度值转为彩色图像)
colorizer = rs.colorizer()
try:
    print("⏳ 正在启动相机...")
    pipe.start(cfg)
    print("✅ 相机已启动，按 'q' 退出，按 's' 保存点云文件")

    while True:
        # 等待帧
        frames = pipe.wait_for_frames()
        # 4. 对齐帧 (关键步骤：让深度图的像素和彩色图的像素坐标对应)
        aligned_frames = align.process(frames)

        # 获取对齐后的深度帧和彩色帧
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        # 验证是否两帧都有
        if not aligned_depth_frame or not color_frame:
            continue
        # ⭐️ 新增：应用颜色映射器到对齐后的深度帧
        colorized_depth = colorizer.process(aligned_depth_frame)
        
        # --- 核心：生成点云 ---
        
        # A. 告诉点云对象，我们要用这一帧彩色图像作为纹理
        pc.map_to(color_frame)

        # B. 计算点云 (生成 3D 坐标)
        points = pc.calculate(aligned_depth_frame)

        # --- 数据转换 (转为 Numpy以便处理) ---
        
        # 获取顶点坐标 (x, y, z)
        # 原始数据是结构化数组，我们需要将其转换为标准的 (N, 3) float32 数组
        vtx = np.asanyarray(points.get_vertices())
        # view 转换是将内存重新解释，reshape 变成 N行3列
        vertices = vtx.view(np.float32).reshape(-1, 3)

        # 获取纹理坐标 (u, v) - 如果你需要纹理映射
        tex = np.asanyarray(points.get_texture_coordinates())
        # texture_coords = tex.view(np.float32).reshape(-1, 2)

        # --- 可视化 (显示彩色图作为参考) ---
        color_image = np.asanyarray(color_frame.get_data())
  
        # 0,0==============>640
        # ||
        # ||
        # \/
        # 480
        # print(color_image.shape)
        # 结果为(480,640,3)
        # ⭐️ 深度图现在也是 BGR 格式的 8 位图像，可以直接用于 imshow
        depth_colormap = np.asanyarray(colorized_depth.get_data())
        # print( (np.asanyarray(aligned_depth_frame.get_data())).shape)
        # 在图像上打印当前点云点的数量
        # cv2.putText(color_image, f"Points: {len(vertices)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # 拼接显示(彩色图在左，彩色深度图在右)
        # images = np.hstack((color_image, color_image))
        # 在图像上打印信息 (可选)
        # cv2.putText(images, "Color | Depth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 这里results是长度为1的list，print(len(results))可知
        results = model(color_image, **infer_param)
        box_info_list = []
        new_image = color_image.copy()
        if results[0].boxes is not None:
            print(f"当前检测目标数：{len(results[0].boxes)}")
            class_names = results[0].names
            for box in results[0].boxes:
                [x1,y1,x2,y2] = box.xyxy.cpu().numpy().tolist()[0]
                print([x1,y1,x2,y2])
                print(box.cls.cpu().numpy()[0])
                class_idx = int(box.cls.cpu().numpy()[0])
                class_name = class_names[class_idx]
                cv2.rectangle(new_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(new_image, class_name, (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
        cv2.imshow("RealSense Color (Aligned)", new_image)
        
        key = cv2.waitKey(1)
        
        # 按 'q' 退出
        if key == ord('q'):
            break
        
        # 按 's' 保存为 .ply 文件 (可以用 MeshLab 或 CloudCompare 打开)
        if key == ord('s'):
            print("💾 正在保存 pointcloud.ply ...")
            points.export_to_ply("pointcloud.ply", color_frame)
            print("✅ 保存成功！")

finally:
    pipe.stop()
    cv2.destroyAllWindows()
