python GR00T/scripts/gr00t_finetune.py \
   --dataset-path GR00T/Data_transfer/gr00t_dataset \
   --num-gpus 1 \
   --lora_rank 16 \
   --batch_size 16 \
   --save_steps 2000 \
   --output-dir GR00T/h1_check_points \
   --max-steps 20000 \
   --data-config Unitree_H1 \
   --video-backend torchvision_av