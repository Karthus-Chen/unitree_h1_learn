import torch
import os
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from itertools import repeat
import matplotlib.pyplot as plt
import time
import sys
import pathlib
HOME_PATH = str(pathlib.Path("../").parent.resolve())
import sys
sys.path.append(HOME_PATH+"/ACT")
# print(HOME_PATH)
from utils.utils import set_seed, compute_dict_mean, detach_dict, load_data
from utils.io_utils import IOUtils
from utils.model_interface import ModelInterface
import matplotlib.pyplot as plt
import pickle
import argparse


def repeater(data_loader):
    epoch = 0
    for loader in repeat(data_loader):
        for data in loader:
            yield data
        print(f'Epoch {epoch} done')
        epoch += 1

def train_bc(train_dataloader, val_dataloader, config):
    
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    save_every = config['save_every']
    set_seed(seed)

    model_interface = ModelInterface(config)
    model_interface.setup()

    policy = model_interface.make_policy()
    policy.cuda()
    optimizer = model_interface.make_optimizer(policy)

    min_val_loss = np.inf
    best_ckpt_info = None

    train_dataloader = repeater(train_dataloader)
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        if epoch % 500 == 0:
        # validation
            with torch.inference_mode():
                policy.eval()
                validation_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = forward_pass(data, policy)
                    validation_dicts.append(forward_dict)
                    if batch_idx > 20:
                        break

                validation_summary = compute_dict_mean(validation_dicts)
                
                epoch_val_loss = validation_summary['loss']
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
            for k in list(validation_summary.keys()):
                validation_summary[f'val/{k}'] = validation_summary.pop(k)            
            #wandb.log(validation_summary, step=epoch)
            print(f'Val loss:   {epoch_val_loss:.5f}')
            summary_string = ''
            for k, v in validation_summary.items():
                summary_string += f'{k}: {v.item():.3f} '
            print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        
        data = next(train_dataloader)
        forward_dict = forward_pass(data, policy)
        # backward
        loss = forward_dict['loss']
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        epoch_summary = detach_dict(forward_dict)

        # epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)
        #wandb.log(epoch_summary, step=epoch)

        if epoch % save_every == 0 and epoch >= save_every:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            # plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    # plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def training(config):
    ckpt_dir = config['ckpt_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    set_seed(1)
    
    config['dataset_dir']=HOME_PATH+config['dataset_dir']   
    train_dataloader, val_dataloader, stats, _ = load_data(
        config['dataset_dir'],  
        config['camera_names'], 
        config['batch_size'], 
        config['batch_size'],
        config['policy_class']
        )

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)



    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')



def validate(policy, dataloader):
    policy.eval()
    epoch_dicts = []
    with torch.inference_mode():
        for data in dataloader:
            forward_dict = forward_pass(data, policy)
            epoch_dicts.append(forward_dict)
    epoch_summary = compute_dict_mean(epoch_dicts)
    return epoch_summary['loss'], epoch_summary


def train_epoch(policy, optimizer, dataloader):
    policy.train()
    epoch_dicts = []
    for data in dataloader:
        optimizer.zero_grad()
        forward_dict = forward_pass(data, policy)
        loss = forward_dict['loss']
        loss.backward()
        optimizer.step()
        epoch_dicts.append(detach_dict(forward_dict))
    epoch_summary = compute_dict_mean(epoch_dicts)
    return epoch_summary['loss'], epoch_summary


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(
    ), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    # print(f"forward_pass: image_data shape = {image_data.shape}")
    # print(f"forward_pass: qpos_data shape = {qpos_data.shape}")
    # print(f"forward_pass: action_data shape = {action_data.shape}")
    # print(f"forward_pass: is_pad shape = {is_pad.shape}")
    return policy(qpos_data, image_data, action_data, is_pad)


def print_summary(summary):
    summary_string = ' '.join(
        [f'{k}: {v.item():.3f}' for k, v in summary.items()])
    print(summary_string)

def main():
    io_utils = IOUtils()
    config = io_utils.load_config()
    training(config)

if __name__ == '__main__':
    main()

