#!/bin/bash

# Slurm Submission Script
# this is used to execute slurm scripts while also saving their contents. this helps prevent creating tons of different slurm scripts for different hpc compute servers and is most helpful for re-executing slurm scripts if jobs failed or other issues occured. It also logs the contents so you can track what you have submitted or rerun failed jobs easily

# Usage: ./submit_job.sh <job_config_script>

set -e  # Exit immediately if a command exits with a non-zero status

usage() {
    echo "Usage:"
    echo "  $0 <job_config_script>"
    echo "  $0 <HEADER_TYPE> <job_config_script>"
    echo "Examples:"
    echo "  $0 ./job_configs/train_model.sh"
    echo "  $0 reference_a100 ./job_configs/train_model.sh"
    exit 1
}

# Initialize variables
HEADER_TYPE="none"
JOB_CONFIG_PATH=""

# Parse command-line arguments
if [ "$#" -eq 1 ]; then
    JOB_CONFIG_PATH="$1"
elif [ "$#" -eq 2 ]; then
    HEADER_TYPE="$1"
    JOB_CONFIG_PATH="$2"
else
    echo "Error: Incorrect number of arguments."
    usage
fi

# Check if a Conda environment is active
if [ -z "$CONDA_PREFIX" ]; then
    echo "Error: No active Conda environment detected."
    echo "Please activate a Conda environment before submitting the job."
    exit 1
fi

# Optional: Print the active Conda environment
echo "Active Conda environment: $(basename "$CONDA_PREFIX")"

# Define the directory paths
HEADER_DIR="./job_scripts/slurm_headers"

# Path to Slurm headers
HEADER_FILE="${HEADER_DIR}/${HEADER_TYPE}.slurm"

# List of valid HEADER_TYPE values
VALID_HEADERS=("example_h100") # support for  "none" removed due to bug

# Helper Function to check if an element is in an array
contains() {
    local element
    for element in "${VALID_HEADERS[@]}"; do
        if [[ "$element" == "$1" ]]; then
            return 0
        fi
    done
    return 1
}

# Validate HEADER_TYPE
if ! contains "$HEADER_TYPE"; then
    echo "Error: Invalid HEADER_TYPE '$HEADER_TYPE' specified in '$JOB_CONFIG_PATH'."
    echo "Valid options are: ${VALID_HEADERS[*]}"
    exit 1
fi

# Check if the corresponding header file exists (except for 'none', which might have minimal headers)
if [ "$HEADER_TYPE" != "none" ] && [ ! -f "$HEADER_FILE" ]; then
    echo "Error: Slurm header file '$HEADER_FILE' does not exist."
    exit 1
fi

# Create a temporary file to hold the combined script
TEMP_SCRIPT=$(mktemp ./job_scripts/temp_script_XXXXXX.slurm)

# Ensure the temporary script is removed on exit
trap 'rm -f "$TEMP_SCRIPT"' EXIT

# Add the selected Slurm header if not 'none'
if [ "$HEADER_TYPE" != "none" ]; then
    cat "$HEADER_FILE" > "$TEMP_SCRIPT"
    echo "" >> "$TEMP_SCRIPT"  # Add a newline for separation
fi

# Append the job configuration script
cat "$JOB_CONFIG_PATH" >> "$TEMP_SCRIPT"

# Ensure the temporary script is executable
chmod +x "$TEMP_SCRIPT"

# Determine the execution method based on HEADER_TYPE
if [ "$HEADER_TYPE" != "none" ]; then
    # Run the Slurm script using sbatch and capture the output
    execution_output=$(sbatch "$TEMP_SCRIPT")
    execution_method="sbatch"
else
    # Run the script using bash and capture the output
    execution_output=$(bash "$TEMP_SCRIPT")
    execution_method="bash"
fi

# Create the log directory if it doesn't exist
mkdir -p ./logs/job_scripts

# Log file path
LOG_FILE="./logs/job_scripts/executed_slurm_script_contents.log"

# Append the combined script contents to the log file with delimiters, timestamp, and script name
{
    echo "--------------------------------------------------"
    echo "Executed on: $(date)"
    echo "Job Config Script: $(basename "$JOB_CONFIG_PATH")"
    echo "Header Type: $HEADER_TYPE"
    echo "Execution Method: $execution_method"
    echo "--------------------------------------------------"
    cat "$TEMP_SCRIPT"
    echo ""
} >> "$LOG_FILE"

# Echo the sbatch output along with the confirmation message
echo "Script '$JOB_CONFIG_PATH' executed using $execution_method. Output: $execution_output"
