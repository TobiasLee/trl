#!/bin/bash

# Default values
NUM_GPUS=8
MASTER_PORT=29500

# Help message
print_usage() {
    echo "Usage: $0 [--gpus N] [--port PORT] [--config CONFIG.{json,yaml,yml}] [-- EXTRA_ARGS...]"
    echo "  --gpus N        Number of GPUs to use (default: 8)"
    echo "  --port PORT     Master port for distributed training (default: 29500)"
    echo "  --config        Path to config file (JSON or YAML format)"
    echo "  -- EXTRA_ARGS   Additional arguments passed directly to train_rm.py"
    exit 1
}

# Parse script arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            # Verify config file exists and has valid extension
            if [[ ! -f "$CONFIG_FILE" ]]; then
                echo "Error: Config file '$CONFIG_FILE' does not exist"
                exit 1
            fi
            if [[ ! "$CONFIG_FILE" =~ \.(json|yaml|yml)$ ]]; then
                echo "Error: Config file must be a .json, .yaml, or .yml file"
                exit 1
            fi
            shift 2
            ;;
        -h|--help)
            print_usage
            ;;
        --)
            shift
            EXTRA_ARGS="$@"
            break
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            ;;
    esac
done

# Construct the training command
CMD="torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    train_rm.py"

# Add config file if specified
if [ ! -z "$CONFIG_FILE" ]; then
    CMD="$CMD $CONFIG_FILE"
fi

# Add any extra arguments
if [ ! -z "$EXTRA_ARGS" ]; then
    CMD="$CMD $EXTRA_ARGS"
fi

# Print the command being executed
echo "Running: $CMD"

# Execute the command
eval $CMD 
