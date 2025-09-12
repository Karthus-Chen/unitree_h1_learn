import h5py
import json
import os
from pathlib import Path

# 获取主目录路径
HOME_PATH = str(Path("../").parent.resolve())

# 定义路径
hdf5_dir = os.path.join(HOME_PATH, 'GR00T/Data_transfer/hdf5_dataset')
meta_dir = os.path.join(HOME_PATH, 'GR00T/Data_transfer/gr00t_dataset/meta')

# 确保输出目录存在
os.makedirs(meta_dir, exist_ok=True)

# JSONL文件路径
episodes_jsonl = os.path.join(meta_dir, "episodes.jsonl")
tasks_jsonl = os.path.join(meta_dir, "tasks.jsonl")

def generate_metadata_files():
    """生成episodes.jsonl和tasks.jsonl文件"""
    episodes_data = []
    tasks_data = []
    
    # 获取所有HDF5文件并按名称排序
    hdf5_files = sorted([f for f in os.listdir(hdf5_dir) if f.endswith('.hdf5')])
    
    for episode_index, hdf5_file in enumerate(hdf5_files):
        hdf5_path = os.path.join(hdf5_dir, hdf5_file)
        
        with h5py.File(hdf5_path, 'r') as f:
            # 从HDF5文件中读取任务描述
            task_description = f['/task'][()].decode('utf-8')
            
            # 获取episode长度（从timestamp）
            timestamp = f['/timestamp'][:]
            episode_length = len(timestamp)
            
            # 添加到episodes数据（保留valid）
            episodes_data.append({
                "episode_index": episode_index,
                "tasks": [task_description, "valid"],  # 保留valid
                "length": int(episode_length)
            })
            
            # 添加到tasks数据（只包含实际任务）
            tasks_data.append({
                "task_index": episode_index,
                "task": task_description  # 不包含valid
            })
    
    # 写入episodes.jsonl
    with open(episodes_jsonl, 'w') as f:
        for episode in episodes_data:
            f.write(json.dumps(episode) + '\n')
    
    # 写入tasks.jsonl
    with open(tasks_jsonl, 'w') as f:
        for task in tasks_data:
            f.write(json.dumps(task) + '\n')
    
    print(f"共处理 {len(episodes_data)} 个episodes")
    print(f"共记录 {len(tasks_data)} 个tasks")
    print("示例任务:", tasks_data[0]["task"] if tasks_data else "无")

if __name__ == "__main__":
    generate_metadata_files()