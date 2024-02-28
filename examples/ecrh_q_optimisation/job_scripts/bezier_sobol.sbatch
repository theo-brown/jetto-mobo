#!/bin/bash
#SBATCH -J BezVec
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

# Parameters
N_SOBOL_STEPS=1
let "N_SOBOL_SAMPLES = $SLURM_CPUS_PER_TASK * $N_SOBOL_STEPS"
let "N_BAYESOPT_STEPS = 8 - $N_SOBOL_STEPS"

# Directory structure
cd /home/tab53/rds/rds-ukaea-ap002-mOlK9qn0PlQ/tab53/jetto-mobo/examples/ecrh_q_optimisation
OUTPUT_DIR=data/vector/bezier_sobol_batch32

# Save command as a variable, using 'here document' syntax
read -r -d '' COMMAND <<- EOM
$CONDA_PREFIX/bin/python main.py \
--batch_size $SLURM_CPUS_PER_TASK \
--initial_batch_size $N_SOBOL_SAMPLES \
--n_iterations $N_BAYESOPT_STEPS \
--output_dir $OUTPUT_DIR \
--parameterisation bezier \
--sobol_only \
--n_parameters 10 \
--discard_failures
EOM

# Resume if target directory exists
if [ -d "$OUTPUT_DIR" ]; then
    COMMAND="$COMMAND --resume"
else 
    mkdir -p $OUTPUT_DIR
fi

# Save command to file
echo "$COMMAND" > $OUTPUT_DIR/command.txt

# Execute
echo "Executing command: $COMMAND"
eval $COMMAND
