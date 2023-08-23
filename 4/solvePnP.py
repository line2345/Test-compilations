import cv2
import numpy as np

# 三维坐标点
object_points = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], dtype=np.float32)

# 图像坐标点
image_points = np.array([[100, 150], [300, 150], [300, 300], [100, 300]], dtype=np.float32)

# 摄像头参数
focal_length = 500
image_center = (0, 0)  # 假设光学中心位于图像中心

# 相机内参矩阵
camera_matrix = np.array([[focal_length, 0, image_center[0]], [0, focal_length, image_center[1]], [0, 0, 1]])

# PnP解算
success, rotation_vector, translation_vector = cv2.solvePnP(object_points, image_points, camera_matrix, None)

# 旋转向量转换为旋转矩阵
rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

# 打印结果
print("平移向量（摄像头坐标系）：")
print(translation_vector)
print("旋转矩阵：")
print(rotation_matrix)