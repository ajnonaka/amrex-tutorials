#!/bin/bash
# model.x - wrapper script for running simulations with datalog extraction

# Configuration
EXE="main3d.gnu.ex"
INPUTS="inputs"
INCLUDE_HEADER=true
POSTPROCESSOR="./postprocess_datalog.sh"

# Get arguments
INPUT_FILE="$1"
OUTPUT_FILE="$2"

# Validate arguments
if [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_FILE" ]; then
    echo "Usage: $0 <input_file> <output_file>"
    echo "Example: $0 ptrain.txt ytrain.txt"
    exit 1
fi

# Check required files
for file in "$INPUT_FILE" "$EXE" "$INPUTS" "$POSTPROCESSOR"; do
    if [ ! -f "$file" ]; then
        echo "Error: Required file $file not found!"
        exit 1
    fi
done

# Make postprocessor executable
chmod +x "$POSTPROCESSOR"

# Read parameter names if available
if [ -f "pnames.txt" ]; then
    readarray -t PARAM_NAMES < pnames.txt
    echo "Parameter names: ${PARAM_NAMES[@]}"
fi

# Function to run single simulation
run_simulation() {
    local run_counter=$1
    shift
    local param_values=("$@")
    
    # Create run directory
    local run_dir="run_$(printf "%04d" $run_counter)"
    mkdir -p "$run_dir"
    
    # Build command arguments
    local cmd_args=""
    if [ ${#PARAM_NAMES[@]} -gt 0 ]; then
        for i in "${!PARAM_NAMES[@]}"; do
            if [ $i -lt ${#param_values[@]} ]; then
                cmd_args+="${PARAM_NAMES[$i]}=${param_values[$i]} "
            fi
        done
    fi
    
    # Run simulation
    echo "=== Running simulation $run_counter ===" >&2
    echo "Parameters: ${param_values[@]}" >&2
    
    cd "$run_dir"
    ../$EXE ../$INPUTS $cmd_args > simulation.log 2>&1
    local exit_code=$?
    cd ..
    
    # Check if simulation produced output
    if [ ! -f "$run_dir/datalog.txt" ]; then
        echo "Error: Simulation $run_counter failed - no datalog.txt" >&2
        return 1
    fi
    
    # Call postprocessor
    local result=$($POSTPROCESSOR "$run_dir" "$run_counter" "outnames.txt")
    
    if [ $? -eq 0 ]; then
        echo "$result"
        return 0
    else
        return 1
    fi
}

# Initialize output file
> "$OUTPUT_FILE"

# Process all simulations
run_counter=0
header_written=false

while IFS= read -r line; do
    # Skip empty lines and comments
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    
    # Parse parameters
    param_values=($line)
    
    if [ ${#param_values[@]} -eq 0 ]; then
        continue
    fi
    
    # Run simulation and get result
    result=$(run_simulation $run_counter "${param_values[@]}")
    
    if [ $? -eq 0 ]; then
        # Write header on first successful run
        if [ "$header_written" = false ] && [ "$INCLUDE_HEADER" = true ] && [ -f "outnames.txt" ]; then
            readarray -t output_names < outnames.txt
            echo "# ${output_names[@]}" >> "$OUTPUT_FILE"
            header_written=true
        fi
        
        # Write result
        echo "$result" >> "$OUTPUT_FILE"
        echo "✓ Run $run_counter completed"
    else
        echo "✗ Run $run_counter failed"
    fi
    
    ((run_counter++))
    
done < "$INPUT_FILE"

echo "Completed processing $run_counter runs"
if [ -f "$OUTPUT_FILE" ]; then
    echo "Output written to $OUTPUT_FILE"
fi
