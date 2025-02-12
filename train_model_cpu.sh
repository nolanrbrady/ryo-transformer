#!/bin/bash
#SBATCH --job-name=3d_vision_transformer # Job name
#SBATCH --nodes=1                     # Run all processes on a single node
#SBATCH --ntasks=20                   # a single CPU
#SBATCH --mem=500gb                   # Increase Job memory request
#SBATCH --time=1-00:00:00             # Time limit hrs:min:sec
#SBATCH --output=3d_vision_transformer_%j.log   # Standard output and error log
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=nobr3541@colorado.edu     # Where to send mail
#SBATCH --partition=bigmem            # Partition

# Load modules below for gpu such as CUDA
#module load cuda11.8/toolkit/11.8.0
#module load tensorflow/2.15.0.post1  # Adjust this to the correct module name

# activate environment

python 3d_vision_transformer_binary_classification_caption.py