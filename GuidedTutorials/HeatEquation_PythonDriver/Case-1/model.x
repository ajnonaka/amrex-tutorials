#!/bin/bash
# model.x - wrapper script that extracts min/max from plotfile headers

# Source the functions from external file
# Use the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/functions.sh"

# Alternative ways to source:
# source ./functions.sh           # If functions.sh is in current directory
# . ./functions.sh               # Alternative syntax for source
# source /path/to/functions.sh    # Absolute path

# CONFIGURABLE VARIABLES
EXE="main.ex"                    # Executable name
LEVEL=0                          # Default level
TARGET_I=10                      # Target i coordinate
TARGET_J=20                      # Target j coordinate  
TARGET_K=30                      # Target k coordinate

# Create parameter names file
cat > pnames.txt << EOF
input_1
input_2
input_3
input_4
input_5
EOF

# Create parameter marginal PC file for uncertainty quantification
cat > param_margpc.txt << EOF
1 0.3
3 0.1
1 0.5
EOF

# PC configuration for uncertainty quantification
PC_TYPE=HG # Hermite-Gaussian PC
INPC_ORDER=1

# Get positional arguments
INPUT_FILE="$1"
OUTPUT_FILE="$2"

# Validate arguments
if [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_FILE" ]; then
    echo "Usage: $0 <input_file> <output_file>"
    echo "Example: $0 ptrain.txt ytrain.txt"
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE not found!"
    exit 1
fi

# Check if parameter names file exists
if [ ! -f "pnames.txt" ]; then
    echo "Warning: pnames.txt not found! Using default parameter names."
fi

# Read parameter names if file exists
if [ -f "pnames.txt" ]; then
    readarray -t PARAM_NAMES < pnames.txt
    echo "Parameter names: ${PARAM_NAMES[@]}"
fi

echo "Target coordinates: ($TARGET_I,$TARGET_J,$TARGET_K)"
echo "Using executable: $EXE"
echo "Processing level: $LEVEL"

# Function to process all simulations
function process_all_simulations {
    local input_file=$1
    local output_file=$2
    
    local run_counter=0
    local output_initialized=false
    
    # Read input file line by line
    while IFS= read -r line; do
        # Skip empty lines and comments
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        
        # Parse parameter values
        local param_values=($(echo "$line"))
        
        if [ ${#param_values[@]} -eq 0 ]; then
            echo "Warning: Empty parameter line at run $run_counter, skipping"
            continue
        fi
        
        echo "Processing run $run_counter..."
        
        # Run the simulation
        local run_dir=$(run_single_simulation $run_counter "${param_values[@]}")
        
        if [ $? -eq 0 ] && [ -d "$run_dir" ]; then
            # Process the plotfile and extract data
            local result=$(process_single_plotfile "$run_dir" $run_counter)
            
            if [ $? -eq 0 ]; then
                # Initialize output file on first successful run
                if [ "$output_initialized" = false ]; then
                    initialize_output_file "$output_file"
                    output_initialized=true
                fi
                
                # Write result to output file
                echo "$result" >> "$output_file"
                echo "Run $run_counter completed successfully"
            else
                echo "Failed to process run $run_counter"
            fi
        else
            echo "Failed to run simulation $run_counter"
        fi
        
        ((run_counter++))
        
    done < "$input_file"
    
    echo "Completed processing $run_counter runs"
    echo "Output written to $output_file"
    
    if [ -f "outnames.txt" ]; then
        echo "Output variables:"
        cat outnames.txt
    fi
}

# Call the main processing function
process_all_simulations "$INPUT_FILE" "$OUTPUT_FILE"
