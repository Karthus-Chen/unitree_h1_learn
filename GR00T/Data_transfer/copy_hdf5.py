import os
import shutil
import pathlib

# 获取主目录路径
HOME_PATH = str(pathlib.Path("../").parent.resolve())

# 定义路径
dataset_path = os.path.join(HOME_PATH, 'DataCollecter/dataset')
gr00t_dataset_path = os.path.join(HOME_PATH, 'GR00T/Data_transfer/hdf5_dataset')

# 确保目标目录存在
os.makedirs(gr00t_dataset_path, exist_ok=True)

# 遍历源目录中的所有文件
for filename in os.listdir(dataset_path):
    if filename.startswith('episode_') and filename.endswith('.hdf5'):
        try:
            # 提取数字部分
            num_str = filename.split('_')[1].split('.')[0]
            episode_num = int(num_str)
            
            # 格式化为6位数字，不足补零
            new_num = f"{episode_num:06d}"
            new_filename = f"episode_{new_num}.hdf5"
            
            # 构建完整路径
            src_file = os.path.join(dataset_path, filename)
            dst_file = os.path.join(gr00t_dataset_path, new_filename)
            
            # 复制文件
            shutil.copy2(src_file, dst_file)
            print(f"Copied: {filename} -> {new_filename}")
            
        except (ValueError, IndexError) as e:
            print(f"Skipping invalid filename: {filename} - {str(e)}")

print("File copying completed!")