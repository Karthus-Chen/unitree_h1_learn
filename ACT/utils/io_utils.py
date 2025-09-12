import os
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
import yaml
from einops import rearrange

import pathlib
HOME_PATH = str(pathlib.Path("../").parent.resolve())

CONFIG_DIR= HOME_PATH+"/ACT/config.yaml"

class IOUtils:
    @staticmethod
    def save_checkpoint(model_state_dict, epoch, ckpt_dir, seed,tag=None,min_val_loss=None):
        if tag == 'last':
            ckpt_path = f"{ckpt_dir}/policy_last.ckpt"
            torch.save(model_state_dict, ckpt_path)
            print(f"Last checkpoint saved at {ckpt_path}")
        elif tag == 'best':
            ckpt_path = f"{ckpt_dir}/policy_best.ckpt"
            torch.save(model_state_dict, ckpt_path)
            print(f"Best ckpt with val loss {min_val_loss:.6f} at epoch{epoch} saved!")
        else:
            ckpt_path = f"{ckpt_dir}/checkpoint_epoch_{epoch}_seed_{seed}.ckpt"
            torch.save(model_state_dict, ckpt_path)
            print(f"Checkpoint saved at {ckpt_path} for epoch {epoch}")


    @staticmethod
    def load_stats(ckpt_dir,ckpt_name='0.ckpt'):
        if(ckpt_name == 'policy_epoch_good_1.ckpt'):
            stats_path = 'good_models/good_1/dataset_stats.pkl'
        else:
            stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')

        with open(stats_path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def save_results(ckpt_dir, ckpt_name, summary_str, episode_returns, highest_rewards):
        result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
        with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
            f.write(summary_str)
            f.write(repr(episode_returns))
            f.write('\n\n')
            f.write(repr(highest_rewards))

    @staticmethod
    def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
        for key in train_history[0]:
            plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
            plt.figure()
            train_values = [summary[key].item() for summary in train_history]
            val_values = [summary[key].item() for summary in validation_history]
            plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
            plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
            plt.tight_layout()
            plt.legend()
            plt.title(key)
            plt.savefig(plot_path)
        print(f'Saved plots to {ckpt_dir}')

    @staticmethod
    def load_config():
        # print("当前工作目录 (os.getcwd()):", os.getcwd())
        
        # # 打印程序实际查找的配置文件路径
        # print("程序正在查找的CONFIG_DIR路径:", CONFIG_DIR)
        
        # # 打印CONFIG_DIR的绝对路径（便于确认实际位置）
        # print("CONFIG_DIR的绝对路径:", os.path.abspath(CONFIG_DIR))
        #         # 检查文件是否存在
        # if not os.path.exists(CONFIG_DIR):
        #     print("错误：配置文件不存在！")
        #     # 列出当前目录下的文件，确认是否有config.yaml
        #     print("当前目录下的文件:", os.listdir(os.getcwd()))
        with open(CONFIG_DIR, 'r') as file:
            config = yaml.safe_load(file)
        return config


    @staticmethod
    def save_stats(ckpt_dir, stats):
        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)


    @staticmethod
    def load_policy(config, ckpt_name):
        from utils.model_interface import ModelInterface
        model_interface = ModelInterface(config)
        policy = model_interface.make_policy()
        ckpt_path = os.path.join(config['ckpt_dir'], ckpt_name)
        policy.load_state_dict(torch.load(ckpt_path))
        policy.cuda()
        policy.eval()
        return policy
    
    def load_policy_pt(config,policy_path):
        policy_path = os.path.join(config['ckpt_dir'], policy_path)
        policy = torch.jit.load(policy_path, map_location="cuda")
        policy.cuda()
        policy.eval()
        return policy

