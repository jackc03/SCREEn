#!/usr/bin/bash
#SBATCH -JTrainingFACE                 # Job name
#SBATCH -N 1                           # Number of nodes
#SBATCH --ntasks-per-node=1            # One task (process) per node
#SBATCH --gres=gpu:H100:2              # 2 H100 GPUs
#SBATCH --cpus-per-gpu=6               # 6 CPUs per GPU (12 total)
#SBATCH --mem-per-cpu=22G              # Memory per CPU (6 CPUs * 22G = 132G/GPU)
#SBATCH -t 8:00:00                     # Time limit
#SBATCH -o Report-%j.out               # Output file
#SBATCH --mail-type=END,FAIL           # Notification type
#SBATCH --mail-user=jcochran66@gatech.edu


module load cuda/11.8
module load anaconda3
conda activate sr_design

nvidia-smi
srun --exclusive torchrun --nproc_per_node 2 run_screen.py --mode="train" --epochs=10 --batch_size=1 --num_workers=6
# srun --exclusive torchrun --nproc_per_node 2 run_screen.py --mode="demo" --demo_samples=10 --weights ckpt/epoch_003.pt --num_workers=6