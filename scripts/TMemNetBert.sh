#!/bin/sh
#SBATCH -J TMemNetBert_wow_v0.0
#SBATCH -o logs/TMemNetBert_wow_v0.0.%j.out
#SBATCH -p 2080ti
#SBATCH -t 72:00:00

#SBATCH --gres=gpu:2
#SBATCH --ntasks=2

cd $SLURM_SUBMIT_DIR
python train.py\
    --total_gpu_num=2\
    --model_type='TMemNetBert'\
    --use_cs_ids=false\
    --knowledge_alpha=0.25\
    --output_dir='output/TMemNetBert_wow_v0.0'\
    --num_train_epochs=20\
    --per_device_train_batch_size=1\
    --per_device_eval_batch_size=4\
    --gradient_accumulation_steps=4\
    --learning_rate=2e-5\
    --logging_num_per_epoch=50\
    --save_num_per_epoch=1\
    --dataloader_num_workers=3\
    --disable_tqdm