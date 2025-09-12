import torch
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange
import sys
import pathlib
HOME_PATH = str(pathlib.Path("../").parent.resolve())
sys.path.append(HOME_PATH+'/ACT')

from utils.utils import set_seed
from utils.io_utils import IOUtils
from utils.model_interface import ModelInterface
import random
import time
from torchvision import transforms
import cv2
import sys
import torch
import time
import argparse

sys.path.append(HOME_PATH)
from Mujoco_env.envs.h1_ik import make_sim_env

from DataCollecter.h1_record import sample_transfer_pose


R2D=180/3.1415926


class H1DexEnvInference:
    def __init__(self, mujoco_env,
                 peg_pose,
                 ):
        self.mujoco_env=mujoco_env
        self.peg_pose=peg_pose
        
    def get_image(self,obs):
        top=obs['images']["top"]
        angle=obs['images']["angle"]

        target_size = (320,240)  # Replace with your desired dimensions
        
        top = cv2.resize(top, target_size, interpolation=cv2.INTER_LINEAR)
        angle = cv2.resize(angle, target_size, interpolation=cv2.INTER_LINEAR)
        
        self.view_flag=True
        return top,angle
    
    def write_img(self,img1,img2):
        pass
        output_image_path = "output_image.jpg"  # 输出文件的路径
        cv2.imwrite(output_image_path, np.hstack((img1,img2)))

    def get_state(self):
        obs=self.mujoco_env._get_qpos_obs()['qpos']
        return obs
    
    def step(self,action):

        action[:]=action[:]/R2D
        self.mujoco_env.step_all_simple(action)

    def reset(self):
        self.mujoco_env.reset(self.peg_pose)
        self.mujoco_env.render()
        time.sleep(1)
        
def get_image(img1,img2):
    # img1=img1[:, :, ::-1]
    # img2=img2[:, :, ::-1]
    output_image_path = "output_image.jpg"  # 输出文件的路径
    cv2.imwrite(output_image_path, np.hstack((img1,img2)))
    curr_images = []
    img1=rearrange(img1, 'h w c -> c h w')
    img2=rearrange(img2, 'h w c -> c h w')
    curr_images.append(img1)
    curr_images.append(img2)
    # curr_images.append(img3)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    
    return curr_image

 
def eval_bc(config, pt_name='model_0.pt',num_rollouts = 20):
    print("-------------------------")
    print(pt_name)
    print("-------------------------")
    set_seed(config['seed'])
    config['dataset_dir']=HOME_PATH+config['dataset_dir']
    model_interface = ModelInterface(config)
    model_interface.setup()
    # policy = IOUtils.load_policy_pt(config, HOME_PATH+'/ACT/ckpt_models/'+pt_name)
    policy = IOUtils.load_policy(config, HOME_PATH+'/ACT/ckpt_models/'+pt_name)
    stats = IOUtils.load_stats(HOME_PATH+'/'+config['ckpt_dir'])
      

    success_num=run_episode(config, policy, stats,num_rollouts)
    
def get_task_emb(task,clip_tokenizer,clip_model):
    tokens = clip_tokenizer(
    task,
    padding="max_length",
    max_length=77,  # CLIP's default max length
    truncation=True,
    return_tensors="pt"
    ).to("cpu")
    with torch.no_grad():  # Disable gradient computation to save memory
        outputs = clip_model.text_model(**tokens)
        task_embeddings = outputs.pooler_output  # Shape: (batch_size, 512)
    task_emb= torch.nn.functional.normalize(task_embeddings, p=2, dim=1)
    return task_emb

def run_episode(config, policy, stats, num_rollouts):

    max_times = 400    
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    num_queries = config['chunk_size']

    Freq=80 #hz
    mujoco_env = make_sim_env(freq=Freq)
    peg_pose = sample_transfer_pose()#随机
    env = H1DexEnvInference(
        mujoco_env=mujoco_env,
        peg_pose=peg_pose
    )

    desired_dt = 1.0 / Freq  # frequency

    for rollout_id in range(num_rollouts):

        env.reset()
        # Transmitting the red stick
        # Clean table

        max_times=580

        if config['temporal_agg']:
            all_time_actions = torch.zeros([max_times, max_times+num_queries, 32]).cuda()
            query_frequency=1
        else:
            query_frequency = num_queries
        print(rollout_id+1,"/",num_rollouts)

        with torch.inference_mode():
            for t in tqdm(range(max_times)):
                start=time.time()
                if t % query_frequency == 0:
                    obs=np.zeros(32)
                    obs[0:17]=env.get_state()*R2D
                    np.set_printoptions(precision=4, suppress=True)
                    # print("obs:",obs[0:17])
                    obs[17:32]=0
                    qpos_numpy = np.array(obs.copy())
                    qpos = pre_process(qpos_numpy)
                    qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                    img1,img2=env.get_image(env.mujoco_env._get_image_obs())
                    curr_image = get_image(img1,img2)

                    if config['policy_class'] == "ACTTV":
                        if t % query_frequency == 0:  #隔30帧运行一次
                            
                            all_actions = policy(qpos, curr_image)  
                        if config['temporal_agg']:
                            all_time_actions[[t], t:t+num_queries] = all_actions  #num_queries=chunk_size

                            actions_for_curr_step = all_time_actions[:, t]
                            # print("actions_for_curr_step.dtype:",actions_for_curr_step.dtype)
                            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                            actions_for_curr_step = actions_for_curr_step[actions_populated]
                            k = 0.1  #0.01
                            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                            exp_weights = exp_weights / exp_weights.sum()
                            exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                            raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                        else:
                            raw_action = all_actions[:, t % query_frequency]
                    else:
                        raise NotImplementedError

                    raw_action = raw_action.squeeze(0).cpu().numpy()
                    action = post_process(raw_action)
                    env.step(action[:17])
                    all_cost_time=time.time()-start
                    # print("all_cost_time:",all_cost_time)
                    # if(cost_time<desired_dt):
                    #     time.sleep(desired_dt-cost_time)
                    # else:
                    #     print("timeout")
         


def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='Train BC model with specified epoch number.')
    parser.add_argument('--epoch', type=int, required=True, 
                       help='Epoch number of the model checkpoint to load (e.g., 8000)')
    args = parser.parse_args()
    io_utils = IOUtils()
    config = io_utils.load_config()
    pt_name_ = "policy_epoch_" + str(args.epoch) + ".ckpt"
    print(pt_name_)
    eval_bc(config, pt_name=pt_name_, num_rollouts=100000)


if __name__ == '__main__':
    main()

    


