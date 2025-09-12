import os
import torch
from utils.utils import load_data, set_seed
from detr.policy import ACTPolicy, CNNMLPPolicy,ACTTVPolicy

class ModelInterface:
    def __init__(self, config):
        self.config = config

    def setup(self):
        set_seed(self.config['seed'])
        
        self.setup_task_parameters()
        self.setup_policy_parameters()

    def episode_cnt(self,config):
        import os
        episode_count = 0
        for file_name in os.listdir(config['dataset_dir']):
            if file_name.startswith("episode_"):
                    episode_count += 1
        print("###### episode_count ######")
        print(episode_count)
        print("###### episode_count ######")
        return episode_count

    def setup_task_parameters(self):

        self.config['num_episodes']=self.episode_cnt(self.config)

        print(self.config['dataset_dir'] )
        print(self.config['num_episodes'] )
        print(self.config['episode_len'] )
        print(self.config['camera_names'] )

    def setup_policy_parameters(self):
        if self.config['policy_class'] == 'ACT':
            self.config['policy_config'] = {
                'lr': self.config['lr'],
                'num_queries': self.config['chunk_size'],
                'kl_weight': self.config['kl_weight'],
                'hidden_dim': self.config['hidden_dim'],
                'dim_feedforward': self.config['dim_feedforward'],
                'lr_backbone': self.config['lr_backbone'],
                'backbone': self.config['backbone'],
                'enc_layers': self.config['enc_layers'],
                'dec_layers': self.config['dec_layers'],
                'nheads': self.config['nheads'],
                'camera_names': self.config['camera_names'],
            }
        elif self.config['policy_class'] == 'ACTTV':
            backbone = 'dino_v2'
            self.config['policy_config'] = {
                'lr': self.config['lr'],
                'num_queries': self.config['chunk_size'],
                'kl_weight': self.config['kl_weight'],
                'hidden_dim': self.config['hidden_dim'],
                'dim_feedforward': self.config['dim_feedforward'],
                'lr_backbone': self.config['lr_backbone'],
                'backbone': backbone,
                'enc_layers': self.config['enc_layers'],
                'dec_layers': self.config['dec_layers'],
                'nheads': self.config['nheads'],
                'camera_names': self.config['camera_names'],
                'state_dim': 32,
                'action_dim': 32,
                'qpos_noise_std': self.config['qpos_noise_std'],
            }
        elif self.config['policy_class'] == 'CNNMLP':
            self.config['policy_config'] = {
                'lr': self.config['lr'], 
                'lr_backbone': self.config['lr_backbone'], 
                'backbone': self.config['backbone'], 
                'num_queries': 1,
                'camera_names': self.config['camera_names'],
            }

        else:
            raise NotImplementedError

    def make_policy(self):
        if self.config['policy_class'] == 'ACT':
            return ACTPolicy(self.config['policy_config'])
        elif self.config['policy_class'] == 'ACTTV':
            return ACTTVPolicy(self.config['policy_config'])
        elif self.config['policy_class'] == 'CNNMLP':
            return CNNMLPPolicy(self.config['policy_config'])
        else:
            raise NotImplementedError

    def make_optimizer(self, policy):
        return policy.configure_optimizers()

    def load_data(self):
        return load_data(
            self.config['dataset_dir'], 
            self.config['num_episodes'], 
            self.config['camera_names'], 
            self.config['batch_size'], 
            self.config['batch_size']
        )

    def load_policy(self, ckpt_name):
        policy = self.make_policy()
        ckpt_path = os.path.join(self.config['ckpt_dir'], ckpt_name)
        policy.load_state_dict(torch.load(ckpt_path))
        policy.cuda()
        policy.eval()
        return policy
