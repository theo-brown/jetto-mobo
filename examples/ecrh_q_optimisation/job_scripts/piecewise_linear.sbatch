#!/bin/bash
#SBATCH -J piecewise_linear
#SBATCH -A UKAEA-AP002-CPU
#SBATCH -p icelake
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL

# Set up environment
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-icl
module load miniconda/3
source /home/tab53/.bashrc
conda activate jetto-mobo

# Allow Pytorch to multithread
unset OMP_NUM_THREADS

# Fix CUDA error
LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH

# Directory structure
cd /home/tab53/rds/rds-ukaea-ap002-mOlK9qn0PlQ/tab53/jetto-mobo/examples/ecrh_q_optimisation
OUTPUT_DIR="data/piecewise_linear"

# Save command as a variable, using 'here document' syntax
read -r -d '' COMMAND <<- EOM
$CONDA_PREFIX/bin/python main.py \
--batch_size 30 \
--initial_batch_size 30 \
--output_dir $OUTPUT_DIR \
--parameterisation piecewise_linear \
--alpha 0.01
EOM

# Resume if target directory exists
if [ -d "$OUTPUT_DIR" ]; then
    COMMAND="$COMMAND --resume"
fi

# Execute
echo "Executing command: $COMMAND"
eval $COMMAND
