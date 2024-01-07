#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=2
#SBATCH --gres=gpu
#SBATCH --time=01:00:00
#SBATCH --mem=24GB
#SBATCH -o assignment_2-%j
#SBATCH --export=ALL

export seed=64
export epoch=20
export emb_name="42B"
export emb_dim=300
export batch_size=128


export WORKDIR="$HOME/WORK/NLP-with-Deep-Learning/assignment_2"
export SCRDIR="/scratch/general/vast/$USER/{$seed}job_{0.001}_emb_{$emb_name}_{$emb_dim}_$batch_size"

mkdir -p $SCRDIR
cp -r $WORKDIR/* $SCRDIR
cd $SCRDIR


source ~/miniconda3/etc/profile.d/conda.sh
conda activate envKD



#python ./mini_project2.py --epochs $epoch --embedding_name $emb_name --embedding_dim $emb_dim --concatenation 0 --lab_emb_incl 0 --batch_size $batch_size > output_{$emb_name}_{$emb_dim}_Mean_{$batch_size}

#python ./mini_project2.py --epochs $epoch --embedding_name $emb_name --embedding_dim $emb_dim --concatenation 1 --lab_emb_incl 0 --batch_size $batch_size > output_{$emb_name}_{$emb_dim}_Concat_{$batch_size}

#python ./mini_project2.py --epochs $epoch --embedding_name $emb_name --embedding_dim $emb_dim --concatenation 0 --lab_emb_incl 1 --batch_size $batch_size > output_{$emb_name}_{$emb_dim}_Mean_{$batch_size}_lab_incl

python ./mini_project2.py --epochs $epoch --embedding_name $emb_name --embedding_dim $emb_dim --concatenation 1 --lab_emb_incl 1 --batch_size $batch_size > output_{$emb_name}_{$emb_dim}_Concat_{$batch_size}_lab_incl
