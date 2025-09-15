<div align="center">
  <img src="media/Title.png" width="500" alt="Unitree H1 Learn">
</div>
<div align="center">
<img src="media/robot_tasks.gif" width="500" height="500" alt="robot_tasks">
</div>

# 创建虚拟环境
```bash
conda create -n robot_sim python==3.10.16
conda activate robot_sim
pip install --upgrade setuptools
pip install mujoco==3.2.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install mujoco-python-viewer==0.1.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install ruckig -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install urdf_parser_py -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install h5py -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install einops -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install zmq -i https://pypi.tuna.tsinghua.edu.cn/simple

cd GR00T
pip install -e .[base] -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install --no-build-isolation flash-attn==2.7.1.post4 
conda install conda-forge::python-orocos-kdl
#conda install conda-forge::pinocchio
```

# 使用方法
```bash
#均在项目文件夹下运行 ~/unitree_h1_learn$ xxxxx
#export https_proxy="http://127.0.0.1:7897" # 7897是网络代理端口
#录制数据集
python3 DataCollecter/h1_record.py

#ACT
#训练
python3 ACT/demos/train_act.py
#推理
python3 ACT/demos/h1_act_eval.py --epoch 4000

#GR00T
#数据处理
python3 GR00T/Data_transfer/copy_hdf5.py 
python3 GR00T/Data_transfer/hd_to_mp4.py 
python3 GR00T/Data_transfer/hd_to_par.py 
python3 GR00T/Data_transfer/hd_to_task_json.py
python3 GR00T/Data_transfer/par_to_stats_json.py
#修改 GR00T/gr00t/experiment/data_config.py 
#添加 info.json  modality.json
#训练
. GR00T/scripts/1-finetune.sh 
#推理
. GR00T/scripts/2-eval_service.sh 
. GR00T/scripts/3-eval_client.sh
```
