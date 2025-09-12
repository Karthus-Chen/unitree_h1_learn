import numpy as np
import time
import copy as cp
import pathlib
HOME_PATH = str(pathlib.Path("../").parent.resolve())
import sys
sys.path.append(HOME_PATH)
print(HOME_PATH)
import Mujoco_env.utils_.KDL_utils.transform as T
from Mujoco_env.envs.h1_ik import make_sim_env
import pathlib
from DataCollecter.h1_policy import H1Policy
import random
import cv2
import os
import h5py

R2D=180/3.1415926
DATASET_DIR = HOME_PATH + '/DataCollecter'+ '/dataset'  #sim_transfer为任务名字
NUM_EPISODES = 10  #数据集采集个数
CONTROL_FREQ = 50  # 控制频率(Hz)
CONTROL_DT = 1.0 / CONTROL_FREQ  # 控制时间间隔(s)
Task_name = 'Clean table'
# Transmitting the red stick
# Clean table




def sample_transfer_pose():
    box_random = 0.03  # box刷新位置随机
    peg_position = [random.uniform(0.4-box_random, 0.42+box_random), 
                   random.uniform(0.10-box_random, 0.10+box_random), 
                   1.03]
    peg_euler = np.array([0.0, 0.0, 0.0])
    peg_pose = np.concatenate([peg_position, peg_euler])
    print("peg_pose   : ", peg_pose)
    return peg_pose

if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    
    env = make_sim_env(freq=CONTROL_FREQ)

    if Task_name == 'Transmitting the red stick':
        episode_len = 600  # 总步数
    elif Task_name == 'Clean table':
        episode_len = 580

    camera_names = ['top', 'angle']
    policy = H1Policy(task_name=Task_name)



    for episode_idx in range(NUM_EPISODES):
        data_dict = {
        '/observations/qpos': [],
        '/action': [],
        '/timestamp': [],
        }
        obs = []
        print(f'\n{episode_idx=}')
        peg_pose = sample_transfer_pose()
        env.reset(peg_pose)
        
        # 确保数据采集按照控制频率进行
        start_time = time.time()
        last_time = start_time
        
        for step in range(episode_len):
            # 计算应该执行的时间
            target_time = start_time + step * CONTROL_DT
            
            # 等待到目标时间
            while time.time() < target_time:
                time.sleep(0.001)  # 短暂休眠避免CPU占用过高
            
            # 记录实际时间戳
            current_time = time.time()
            data_dict['/timestamp'].append(current_time - start_time)
            
            # 执行控制步骤
            action_pos = policy.predict(peg_pose, step)
            action_jnt = env.ik_func(action_pos)
            env.step_all_simple(action_jnt)
            obs.append(env.obs)

        
        # 保存到HDF5文件
        dataset_path = os.path.join(DATASET_DIR)
        if not os.path.exists(DATASET_DIR):
            os.makedirs(DATASET_DIR)
            

        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        state32=np.zeros(32)    
        action32=np.zeros(32)
        #左臂7+右臂7+腰1+左手1+右手1 (7+7+1+1+1)
        for i in range(episode_len):
            state32[:17]=obs[i]['qpos'].copy()*R2D
            action32[:17]=obs[i]['action'].copy()*R2D
            data_dict['/observations/qpos'].append(state32.copy())
            data_dict['/action'].append(action32.copy())
            for cam_name in camera_names:
                target_size = (320,240)
                color_resized = cv2.resize(obs[i]['images'][cam_name], target_size, interpolation=cv2.INTER_LINEAR)
                data_dict[f'/observations/images/{cam_name}'].append(color_resized)
                
        file_name=dataset_path +'/'+'episode_'+ str(episode_idx)+'.hdf5'
        print(file_name)
        with h5py.File(file_name, 'w') as root:
            max_timesteps = episode_len
            obs_ = root.create_group('observations')
            image = obs_.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 240, 320, 3), dtype='uint8',
                                        chunks=(1, 240, 320, 3), )
            qpos = obs_.create_dataset('qpos', (max_timesteps, 32))
            action = root.create_dataset('action', (max_timesteps, 32))
            timestamp = root.create_dataset('timestamp', (max_timesteps,))
            for name, array in data_dict.items():
                root[name][...] = np.array(array)   

            dt = h5py.string_dtype('utf-8')
            root.create_dataset("task", data=Task_name, dtype=dt)
        del obs
        del data_dict