import h5py
import pandas as pd
from pathlib import Path
import numpy as np
import pathlib
import os
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
# 获取主目录路径
HOME_PATH = str(pathlib.Path("../").parent.resolve())

# 定义路径
hdf5_dir = os.path.join(HOME_PATH, 'GR00T/Data_transfer/hdf5_dataset')
parquet_dir = os.path.join(HOME_PATH, 'GR00T/Data_transfer/gr00t_dataset/data/chunk-000/')
os.makedirs(parquet_dir, exist_ok=True)


def convert_hdf5_to_parquet(hdf5_file, parquet_file, file_index, global_index):
    """ 将 HDF5 文件转换为 Parquet 文件 """
    with h5py.File(hdf5_file, 'r') as f:
        # 读取 HDF5 数据
        env_qpos_proprioception = f['/observations/qpos'][:, :17] 
        action = f['/action'][:, :17]
        timestamp = f['/timestamp'][:]
        
        num_frames = len(timestamp)
        data = []
        
        for i in range(num_frames):
            next_reward = 1 if i == num_frames - 1 else 0  # 最后一行 next.reward = 1
            next_done = i == num_frames - 1  # 最后一行 next.done = True
            
            data.append({
                'observation.state': env_qpos_proprioception[i].tolist(),
                'action': action[i].tolist(),
                'timestamp': timestamp[i],
                'annotation.human.action.task_description': file_index,
                'task_index': file_index,
                'annotation.human.validity': 1,
                'episode_index': file_index,
                'index': global_index,
                'next.reward': next_reward,
                'next.done': next_done
            })
            
            global_index += 1  # 递增全局索引
        
        # 转换为 DataFrame
        df = pd.DataFrame(data)
        
        # 保存为 Parquet 格式
        table = pa.Table.from_pandas(df)
        pq.write_table(table, parquet_file)
        
        print(f"✅ 转换完成: {hdf5_file} → {parquet_file}")
    
    return global_index  # 返回最新的索引值



# 全局索引初始化
current_index = 1  

# 处理所有 HDF5 文件
hdf5_files = sorted([f for f in os.listdir(hdf5_dir) if f.endswith('.hdf5')])  # 确保顺序正确
print(f"📂 发现 {len(hdf5_files)} 个 HDF5 文件: {hdf5_files}")

for file_index, hdf5_file in enumerate(hdf5_files):
    hdf5_path = os.path.join(hdf5_dir, hdf5_file)
    parquet_path = os.path.join(parquet_dir, hdf5_file.replace('.hdf5', '.parquet'))
    
    # 转换 HDF5 -> Parquet，并更新索引
    current_index = convert_hdf5_to_parquet(hdf5_path, parquet_path, file_index, current_index)

print("🎉 全部 HDF5 文件转换完成！")

    