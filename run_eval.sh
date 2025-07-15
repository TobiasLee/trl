#!/bin/bash

# Evaluation script for trained reward models
# Usage: ./run_eval.sh [config_file]

set -e

# Default config file
CONFIG_FILE=${1:-"eval_config_example.yaml"}

echo "Starting reward model evaluation..."
echo "Config file: $CONFIG_FILE"
echo "Timestamp: $(date)"

# Set environment variables for better GPU utilization
export CUDA_VISIBLE_DEVICES=${2:-"7"}
export TOKENIZERS_PARALLELISM=false

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate log filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/eval_${TIMESTAMP}.log"

echo "Logging output to: $LOG_FILE"

# Run evaluation with logging
python eval_rm.py "$CONFIG_FILE" 2>&1 | tee "$LOG_FILE"

echo "Evaluation completed!"
echo "Results saved to evaluation_results.csv and evaluation_results.json"
echo "Full log available at: $LOG_FILE" 
