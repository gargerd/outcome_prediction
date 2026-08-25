#!/bin/bash
#SBATCH --job-name=s_9_6_ML_LLM_integrated_gradients
#SBATCH -o slurm_llm_lstm_logs/s_9_6_ML_LLM_integrated_gradients_%j.log
#SBATCH -e slurm_llm_lstm_logs/s_9_6_ML_LLM_integrated_gradients_%j.log
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --time=2-08:00:00
#SBATCH --nice=10000

#set -euo pipefail

# TEMP: disable -e so we see errors instead of silent aborts
set -uo pipefail
set -x  # trace commands in slurm log

echo "Loading anaconda"
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"

echo "Activate conda env"
conda activate scarches

config="LSTM_LLM_train_param_combinations.tsv"

########################################
# GPU + parallelism setup
########################################

# Which GPUs did Slurm give us?
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    GPU_IDS=(0)   # fallback if somehow not set
else
    IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
fi

NUM_GPUS=${#GPU_IDS[@]}
JOBS_PER_GPU=2                        # how many parallel runs per GPU
MAX_PARALLEL=$((NUM_GPUS * JOBS_PER_GPU))

echo "Visible GPUs: ${GPU_IDS[*]}"
echo "NUM_GPUS=$NUM_GPUS"
echo "JOBS_PER_GPU=$JOBS_PER_GPU"
echo "MAX_PARALLEL (total processes) = $MAX_PARALLEL"

# Directory for per-run logs
logdir="slurm_llm_lstm_logs/parallel_runs"
mkdir -p "$logdir"

running=0
slot=0

########################################
# Main loop over parameter combinations
# TSV columns (your order):
#   col1: dataset_name
#   col2: period_end_day
#   col3: data_inclusion_type
#   col4: array_id
########################################

{
    # If TSV has a header, consume it; comment this out if no header
    read -r header || true

    while IFS=$'\t' read -r dataset_name period_end_day data_inclusion_type array_id; do
        # Skip empty lines
        [ -z "${dataset_name:-}" ] && continue

        # Throttle parallelism: if we've launched MAX_PARALLEL jobs, wait for them to finish
        if (( running >= MAX_PARALLEL )); then
            echo "Reached MAX_PARALLEL=$MAX_PARALLEL, waiting for all current jobs to finish..."
            wait
            running=0
        fi

        # Assign GPU in round-robin (with 1 GPU, always index 0)
        gpu_index=$(( slot % NUM_GPUS ))
        gpu_local_id=${GPU_IDS[$gpu_index]}

        echo "Launching array_id=${array_id} on GPU index ${gpu_index} (CUDA_VISIBLE_DEVICES=${gpu_local_id})"
        echo "  dataset_name=${dataset_name}"
        echo "  data_inclusion_type=${data_inclusion_type}"
        echo "  period_end_day=${period_end_day}"

        (
            # One run
            export CUDA_VISIBLE_DEVICES=${gpu_local_id}

            # Very simple log file name to avoid weird characters in path
            logfile="${logdir}/id${array_id}_${dataset_name}_${data_inclusion_type}_${period_end_day}.log"

            echo "[$(date)] Starting run array_id=${array_id} on GPU ${gpu_local_id}" > "$logfile"
            echo "  dataset_name=${dataset_name}" >> "$logfile"
            echo "  data_inclusion_type=${data_inclusion_type}" >> "$logfile"
            echo "  period_end_day=${period_end_day}" >> "$logfile"
            echo >> "$logfile"

            ## A specific python within the desired environment (here: scarches)is executing the python file ==> this needs to be adjusted if environment or HPC changes
            /data/gpfs/projects/punim2121/anaconda3/envs/scarches/bin/python s9_8_LST_on_LLM_embeddings.py \
                --dataset_name "$dataset_name" \
                --data_inclusion_type "$data_inclusion_type" \
                --period_end_day "$period_end_day" \
                >> "$logfile" 2>&1

            echo >> "$logfile"
            echo "[$(date)] Finished run array_id=${array_id}" >> "$logfile"
        ) &

        ((running++))
        ((slot++))

    done
} < "$config"

# Wait for all remaining jobs
wait
echo "All parameter combinations finished."
