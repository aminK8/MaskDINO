#!/bin/bash
#SBATCH --account=PAS0536
#SBATCH --job-name=panoptic
#SBATCH --time=0-1:00:00


#SBATCH --nodes=1 --ntasks-per-node=2
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2


#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=karimimonsefi.1@osu.edu

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
export WORLD_SIZE=2

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

module spider cuda
module load cuda/12.3.0
module load miniconda3
source activate maskdino


 srun python demo_2.py --config-file ./configs/coco/panoptic-segmentation/maskdino_higharc_R50_bs16_50ep_3s_dowsample1_2048.yaml \
  --input /fs/scratch/PAS0536/amin/MultiGen-20M/images/aesthetics_6_25_plus_3m --opts MODEL.WEIGHTS ../maskdino_r50_50ep_300q_hid2048_3sd1_panoptic_pq53.0.pth