import h5py
import cv2
import os
import numpy as np
import pathlib

# 获取主目录路径
HOME_PATH = str(pathlib.Path("../").parent.resolve())

# 定义路径
hdf5_dir = os.path.join(HOME_PATH, 'GR00T/Data_transfer/hdf5_dataset')
output_dir_top = os.path.join(HOME_PATH, 'GR00T/Data_transfer/gr00t_dataset/videos/chunk-000/observation.images.ego_view_top')
output_dir_angle = os.path.join(HOME_PATH, 'GR00T/Data_transfer/gr00t_dataset/videos/chunk-000/observation.images.ego_view_angle')

# 目标图像尺寸
TARGET_SIZE = (224, 224)

def load_hdf5_images(hdf5_file):
    """从HDF5文件中读取top和angle两个视角的图像数据"""
    with h5py.File(hdf5_file, 'r') as f:
        # 检查是否存在两个视角的图像数据
        if 'observations/images/top' in f and 'observations/images/angle' in f:
            top_images = f['observations/images/top'][:]
            angle_images = f['observations/images/angle'][:]
            return top_images, angle_images
        else:
            raise ValueError(f"HDF5文件 {hdf5_file} 中缺少预期的图像数据集路径")

def resize_images(images):
    """将图像序列调整为224x224尺寸"""
    resized_images = []
    for img in images:
        # 调整图像尺寸
        resized_img = cv2.resize(img, TARGET_SIZE)
        resized_images.append(resized_img)
    return np.array(resized_images)

def save_video_from_images(images, output_video_file, frame_rate=50):
    """将图像序列保存为MP4视频"""
    if len(images) == 0:
        print(f"警告: 没有图像数据可保存到 {output_video_file}")
        return
    
    # 定义视频编写器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4V编码器，兼容性更好
    out = cv2.VideoWriter(output_video_file, fourcc, frame_rate, TARGET_SIZE)
    
    # 将每一帧图像写入视频文件
    for img in images:
        # # 如果图像是RGB格式，转换为BGR
        # if len(img.shape) == 3 and img.shape[-1] == 3:
        #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img)
    
    out.release()

def process_hdf5_folder(hdf5_dir, output_dir_top, output_dir_angle):
    """处理文件夹中的所有HDF5文件，生成两个视角的视频"""
    # 确保输出目录存在
    os.makedirs(output_dir_top, exist_ok=True)
    os.makedirs(output_dir_angle, exist_ok=True)
    
    # 获取文件夹中的所有HDF5文件
    hdf5_files = [f for f in os.listdir(hdf5_dir) if f.endswith('.hdf5')]
    
    for filename in hdf5_files:
        hdf5_path = os.path.join(hdf5_dir, filename)
        base_name = os.path.splitext(filename)[0]
        
        try:
            # 从HDF5文件加载图像数据
            top_images, angle_images = load_hdf5_images(hdf5_path)
            
            # 调整图像尺寸为224x224
            top_images = resize_images(top_images)
            angle_images = resize_images(angle_images)

            # 生成输出视频文件路径
            output_video_top = os.path.join(output_dir_top, f"{base_name}.mp4")
            output_video_angle = os.path.join(output_dir_angle, f"{base_name}.mp4")
            
            # 保存两个视角的视频
            save_video_from_images(top_images, output_video_top)
            save_video_from_images(angle_images, output_video_angle)
            
            print(f"成功生成224x224视频: {output_video_top} 和 {output_video_angle}")
            
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")

# 处理HDF5文件夹
process_hdf5_folder(hdf5_dir, output_dir_top, output_dir_angle)