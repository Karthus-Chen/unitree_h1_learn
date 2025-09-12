import time
from contextlib import contextmanager
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from termcolor import cprint
import threading
import copy as cp
import sys
import pathlib
import os
HOME_PATH = str(pathlib.Path("../").parent.resolve())
sys.path.append(HOME_PATH)
print(HOME_PATH)
from Mujoco_env.envs.h1_ik import make_sim_env
from DataCollecter.h1_record import sample_transfer_pose
service_path = os.path.join(HOME_PATH, 'GR00T/gr00t/eval')
sys.path.append(service_path)
from service import ExternalRobotInferenceClient


class Gr00tRobotInferenceClient:
    def __init__(
        self,
        host="localhost",
        port=5555,
        language_instruction="Transmitting the red stick"
    ):
        self.language_instruction = language_instruction
        self.policy = ExternalRobotInferenceClient(host=host, port=port)

    def get_action(self, img1,img2,state):
        # img1 = img1[:, :, ::-1]
        # img2 = img2[:, :, ::-1]
        # output_image_path = "output_image.jpg"  # 输出文件的路径
        # cv2.imwrite(output_image_path, img2)
        obs_dict = {
            "video.ego_view_top": img1[np.newaxis, :, :, :],
            "video.ego_view_angle": img2[np.newaxis, :, :, :],
            "state.left_arm":state[0:7][np.newaxis, :].astype(np.float64),
            "state.right_arm":state[7:14][np.newaxis, :].astype(np.float64),
            "state.torso":state[14:15][np.newaxis, :].astype(np.float64),
            "state.left_hand":state[15:16][np.newaxis, :].astype(np.float64),
            "state.right_hand":state[16:17][np.newaxis, :].astype(np.float64),
            "annotation.human.action.task_description": [self.language_instruction]
        }
        res = self.policy.get_action(obs_dict)
        return res



#################################################################################

class H1DexEnvInference:
    """
    The deployment is running on the local computer of the robot.
    """
    def __init__(self, mujoco_env,
                 peg_pose,
                 ):
        self.mujoco_env=mujoco_env
        self.peg_pose=peg_pose

    def get_image(self,obs):

        top=obs['images']["top"]
        angle=obs['images']["angle"]

        target_size = (224,224)  # Replace with your desired dimensions
        
        top = cv2.resize(top, target_size, interpolation=cv2.INTER_LINEAR)
        angle = cv2.resize(angle, target_size, interpolation=cv2.INTER_LINEAR)
        
        self.view_flag=True

        return top,angle

    def get_state(self):
        obs=self.mujoco_env._get_qpos_obs()['qpos']
        # print(obs)
        return obs
    
    def step(self,action):
        self.mujoco_env.step_all_simple(action)

    def reset(self):
        self.mujoco_env.reset(self.peg_pose)
        self.mujoco_env.render()
        time.sleep(1)

#################################################################################


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_policy", action="store_true"
    )  # default is to playback the provided dataset
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5500)
    parser.add_argument("--action_horizon", type=int, default=16)
    parser.add_argument("--actions_to_execute", type=int, default=600)# step num
    args = parser.parse_args()

    ACTIONS_TO_EXECUTE = args.actions_to_execute
    USE_POLICY = 1
    ACTION_HORIZON = (
        args.action_horizon
    )  # we will execute only some actions from the action_chunk of 16
    MODALITY_KEYS = ["left_arm","right_arm","torso","left_hand","right_hand"]

    if USE_POLICY:
        client = Gr00tRobotInferenceClient(
            host=args.host,
            port=args.port,
            language_instruction="Transmitting the red stick",
        )


        Freq=50 #hz

        mujoco_env = make_sim_env(freq=Freq)
        peg_pose = sample_transfer_pose()#随机
        robot = H1DexEnvInference(
                             mujoco_env=mujoco_env,
                             peg_pose=peg_pose
        )

        desired_dt = 1.0 / Freq  # frequency
        
        # Transmitting the red stick
        # Clean table
        for i in range(10):
            robot.reset()
            last_cmd=client.language_instruction
            cmd = input("Please input cmd: ")
            if(cmd=="\n"):
                cmd=last_cmd
            else:
                client.language_instruction=cmd
            print("cmd: ",client.language_instruction)

            for i in tqdm(range(int(int(ACTIONS_TO_EXECUTE)/int(ACTION_HORIZON)*1.5)), desc="Executing actions"):
                start_time = time.time()
                
                img1,img2 = robot.get_image(robot.mujoco_env._get_image_obs())
                state = robot.get_state()*(180/3.14)
                # np.set_printoptions(precision=4, suppress=True)
                
                action = client.get_action(img1,img2, state)
                # print(state)
                for i in range(ACTION_HORIZON):
                    step_start = time.time()
                    
                    concat_action = np.concatenate(
                        [np.atleast_1d(action[f"action.{key}"][i]) for key in MODALITY_KEYS],
                        axis=0,
                    )
                    robot.step(concat_action*(3.14/180))
                    
                    # Calculate remaining time to maintain Freq
                    elapsed = time.time() - step_start
                    remaining = max(0, desired_dt - elapsed)
                    time.sleep(remaining)

