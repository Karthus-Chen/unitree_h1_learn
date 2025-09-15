import os
import glob
import json
import numpy as np
import pandas as pd


import pathlib
import os
# 获取主目录路径
HOME_PATH = str(pathlib.Path("../").parent.resolve())

# 定义路径
data_dir = os.path.join(HOME_PATH, 'GR00T/Data_transfer/gr00t_dataset/data/chunk-000')
old_stats_file = os.path.join(HOME_PATH, 'GR00T/demo_data/robot_sim.PickNPlace/meta/stats.json')
new_stats_file = os.path.join(HOME_PATH, 'GR00T/Data_transfer/gr00t_dataset/meta/stats.json')


def load_all_data(data_dir, pattern="episode_*.parquet"):
    files = glob.glob(os.path.join(data_dir, pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} found in {data_dir}")
    print(f"Found {len(files)} files.")

    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def list_of_lists_to_array(series):
    """把Series中每个cell的list转成2D np.array"""
    arr = np.array(series.tolist())
    return arr

def compute_stats_array(arr):
    """计算2D数组按列的mean,std,max,min,q01,q99"""
    stats = {
        "mean": np.nanmean(arr, axis=0).tolist(),
        "std": np.nanstd(arr, axis=0).tolist(),
        "max": np.nanmax(arr, axis=0).tolist(),
        "min": np.nanmin(arr, axis=0).tolist(),
        "q01": np.nanpercentile(arr, 1, axis=0).tolist(),
        "q99": np.nanpercentile(arr, 99, axis=0).tolist(),
    }
    return stats

def compute_stats_scalar(series):
    arr = series.to_numpy(dtype=np.float64)
    stats = {
        "mean": np.nanmean(arr).item(),
        "std": np.nanstd(arr).item(),
        "max": np.nanmax(arr).item(),
        "min": np.nanmin(arr).item(),
        "q01": np.nanpercentile(arr, 1).item(),
        "q99": np.nanpercentile(arr, 99).item(),
    }
    return stats

def main():


    # 读取旧stats.json
    with open(old_stats_file, 'r') as f:
        old_stats = json.load(f)

    df = load_all_data(data_dir)

    # 计算observation.state
    obs_state_arr = list_of_lists_to_array(df['observation.state'])
    obs_state_stats = compute_stats_array(obs_state_arr)
    old_stats['observation.state'] = obs_state_stats

    # 计算action
    action_arr = list_of_lists_to_array(df['action'])
    action_stats = compute_stats_array(action_arr)
    old_stats['action'] = action_stats

    # 需要计算的数值标量字段
    scalar_fields = [
        "timestamp",
        "annotation.human.validity",
        "episode_index",
        "index",
        "next.reward",
        "next.done"
    ]

    for field in scalar_fields:
        if field in df.columns:
            stats = compute_stats_scalar(df[field])
            old_stats[field] = stats
        else:
            print(f"Warning: {field} not found in dataframe columns, skipping.")


    # 保存新json
    with open(new_stats_file, 'w') as f:
        json.dump(old_stats, f, indent=4)

    print(f"Saved updated stats to {new_stats_file}")

if __name__ == "__main__":
    main()


























