#!/bin/bash
# functions.sh - AMReX plotfile processing functions

# Function to generate output names using variable names from Header and components from Cell_H
function generate_outnames_from_combined_sources {
    local plotfile_path=$1
    local level=$2

    local level_header="$plotfile_path/Level_${level}/Cell_H"
    local main_header="$plotfile_path/Header"

    if [ ! -f "$level_header" ] || [ ! -f "$main_header" ]; then
        echo "Error: Required header files not found" >&2
        return 1
    fi

    echo "DEBUG: Reading from Header: $main_header" >&2
    echo "DEBUG: Reading from Cell_H: $level_header" >&2

    # Get number of variables from Header (line 2)
    local num_variables=$(sed -n '2p' "$main_header")
    echo "DEBUG: Number of variables from Header: $num_variables" >&2

    # Get actual number of components from Cell_H data
    local dimensions_line=$(grep "^[0-9]*,[0-9]*$" "$level_header" | head -1)
    local num_components=$(echo $dimensions_line | cut -d, -f2)
    echo "DEBUG: Actual components from Cell_H: $num_components" >&2
    echo "DEBUG: Cell_H dimensions line: $dimensions_line" >&2

    # Generate output names file
    > outnames.txt

    # Read variable names from Header (starting at line 3)
    local line_num=3
    for ((var=0; var<num_variables; var++)); do
        # Read variable name
        local var_name=$(sed -n "${line_num}p" "$main_header")
        echo "DEBUG: Variable $var: name='$var_name'" >&2

        # Skip the next line in Header (which might be dimensions, not components)
        ((line_num++))
        local header_value=$(sed -n "${line_num}p" "$main_header")
        echo "DEBUG: Skipping Header line $line_num: '$header_value' (probably not components)" >&2
        ((line_num++))

        # Generate min/max entries using ACTUAL components from Cell_H
        for ((comp=0; comp<num_components; comp++)); do
            echo "${var_name}_${comp}_min" >> outnames.txt
            echo "${var_name}_${comp}_max" >> outnames.txt
            echo "DEBUG: Added ${var_name}_${comp}_min and ${var_name}_${comp}_max" >&2
        done

        # ALL OUTPUT TO STDERR
        echo "Added variable '$var_name' with $num_components actual components" >&2
    done

    # ALL OUTPUT TO STDERR
    echo "Generated outnames.txt with $num_variables variables and $num_components components each" >&2
    echo "DEBUG: Final outnames.txt contents:" >&2
    cat outnames.txt >&2
}

# Function to generate output names from plotfile header
function generate_outnames_from_header {
    local header_file=$1

    if [ ! -f "$header_file" ]; then
        echo "Error: Header file $header_file not found!"
        return 1
    fi

    # Read number of variables from line 2
    local num_variables=$(sed -n '2p' "$header_file")

    # Generate output names file
    > outnames.txt

    local line_num=3
    for ((var=0; var<num_variables; var++)); do
        # Read variable name
        local var_name=$(sed -n "${line_num}p" "$header_file")
        ((line_num++))

        # Read number of components for this variable
        local num_components=$(sed -n "${line_num}p" "$header_file")
        ((line_num++))

        # Generate min/max entries for each component of this variable
        for ((comp=0; comp<num_components; comp++)); do
            echo "${var_name}_${comp}_min" >> outnames.txt
            echo "${var_name}_${comp}_max" >> outnames.txt
        done

        echo "Added variable '$var_name' with $num_components components"
    done

    echo "Generated outnames.txt with $num_variables variables"
}

# Function to find which box contains the target ijk coordinates from Cell_H file
function find_box_index {
    local target_i=$1
    local target_j=$2
    local target_k=$3
    local cell_header_file=$4

    if [ ! -f "$cell_header_file" ]; then
        echo "Error: Cell header file $cell_header_file not found!" >&2
        echo -1
        return
    fi

    # Find the line with box array definition like "(8 0"
    local box_array_line=$(grep -n "^([0-9]* [0-9]*$" "$cell_header_file" | cut -d: -f1)

    if [ -z "$box_array_line" ]; then
        echo "Error: Could not find box array definition in $cell_header_file" >&2
        echo -1
        return
    fi

    # Extract number of boxes from that line
    local num_boxes=$(sed -n "${box_array_line}p" "$cell_header_file" | sed 's/^(\([0-9]*\) [0-9]*$/\1/')

    echo "Found $num_boxes boxes in $cell_header_file" >&2

    # Read each box definition and check if target coordinates fall within
    local box_index=0
    local current_line=$((box_array_line + 1))

    for ((box=0; box<num_boxes; box++)); do
        # Read box definition line like "((0,0,0) (15,15,15) (0,0,0))"
        local box_def=$(sed -n "${current_line}p" "$cell_header_file")

        # Parse the box coordinates using regex
        # Format: ((i_lo,j_lo,k_lo) (i_hi,j_hi,k_hi) (0,0,0))
        if [[ $box_def =~ \(\(([0-9]+),([0-9]+),([0-9]+)\)\ \(([0-9]+),([0-9]+),([0-9]+)\)\ \(([0-9]+),([0-9]+),([0-9]+)\)\) ]]; then
            local i_lo=${BASH_REMATCH[1]}
            local j_lo=${BASH_REMATCH[2]}
            local k_lo=${BASH_REMATCH[3]}
            local i_hi=${BASH_REMATCH[4]}
            local j_hi=${BASH_REMATCH[5]}
            local k_hi=${BASH_REMATCH[6]}

            echo "Box $box: ($i_lo,$j_lo,$k_lo) to ($i_hi,$j_hi,$k_hi)" >&2

            # Check if target coordinates fall within this box
            if [ $target_i -ge $i_lo ] && [ $target_i -le $i_hi ] && \
               [ $target_j -ge $j_lo ] && [ $target_j -le $j_hi ] && \
               [ $target_k -ge $k_lo ] && [ $target_k -le $k_hi ]; then
                echo "Target ($target_i,$target_j,$target_k) found in box $box" >&2
                echo $box
                return
            fi
        else
            echo "Error: Could not parse box definition: $box_def" >&2
        fi

        ((current_line++))
    done

    # If not found, return -1
    echo "Target coordinates ($target_i,$target_j,$target_k) not found in any box" >&2
    echo -1
}

# Function to extract min/max values from cell header
function extract_minmax_from_cell_header {
    local plotfile_path=$1
    local level=$2
    local target_i=$3
    local target_j=$4
    local target_k=$5
    local output_names_file=$6

    local level_header="$plotfile_path/Level_${level}/Cell_H"

    if [ ! -f "$level_header" ]; then
        echo "Error: Level header $level_header not found"
        return 1
    fi

    # Find which box contains the target coordinates
    local box_index=$(find_box_index $target_i $target_j $target_k "$level_header")

    if [ $box_index -eq -1 ]; then
        echo "Error: Target coordinates not found in any box"
        return 1
    fi

    # Find the min/max sections in the level header
    local min_start_line=$(grep -n "^[0-9]*,[0-9]*$" "$level_header" | head -1 | cut -d: -f1)
    local max_start_line=$(grep -n "^[0-9]*,[0-9]*$" "$level_header" | tail -1 | cut -d: -f1)

    if [ -z "$min_start_line" ] || [ -z "$max_start_line" ]; then
        echo "Error: Could not find min/max sections in level header"
        return 1
    fi

    # Get the dimensions from the first line
    local dimensions=$(sed -n "${min_start_line}p" "$level_header")
    local num_boxes=$(echo $dimensions | cut -d, -f1)
    local num_components=$(echo $dimensions | cut -d, -f2)

    # Extract min values (lines after the first dimension line)
    local min_end_line=$((min_start_line + num_boxes))
    sed -n "$((min_start_line + 1)),${min_end_line}p" "$level_header" > temp_min.txt

    # Extract max values (lines after the second dimension line)
    local max_end_line=$((max_start_line + num_boxes))
    sed -n "$((max_start_line + 1)),${max_end_line}p" "$level_header" > temp_max.txt

    # Get the value from the specific box (line number = box_index + 1)
    local target_line=$((box_index + 1))

    if [ $target_line -gt $num_boxes ]; then
        echo "Error: Box index $box_index exceeds available data lines"
        rm -f temp_min.txt temp_max.txt
        return 1
    fi

    # Extract all component values for the target box
    local min_line=$(sed -n "${target_line}p" temp_min.txt)
    local max_line=$(sed -n "${target_line}p" temp_max.txt)

    # Parse components (comma-separated)
    IFS=',' read -ra min_components <<< "$min_line"
    IFS=',' read -ra max_components <<< "$max_line"

    # Read output names
    readarray -t output_names < "$output_names_file"

    # Build output line based on requested output names
    local output_line=""
    for out_name in "${output_names[@]}"; do
        # Parse output name like "phi_0_min" or "phi_1_max"
        if [[ $out_name =~ ^(.*)_([0-9]+)_(min|max)$ ]]; then
            local var_name="${BASH_REMATCH[1]}"
            local component="${BASH_REMATCH[2]}"
            local min_or_max="${BASH_REMATCH[3]}"

            # Get the value
            if [ "$min_or_max" = "min" ]; then
                local value="${min_components[$component]}"
            else
                local value="${max_components[$component]}"
            fi

            # Remove trailing comma if present
            value="${value%,}"

            output_line+="$value "
        fi
    done

    # Clean up temp files
    rm -f temp_min.txt temp_max.txt

    echo "$output_line"
    return 0
}

# Function to get the last plotfile in a directory
function get_last_plotfile {
    local dir=${1:-.}

    if [ ! -d "$dir" ]; then
        echo ""
        return
    fi

    echo "Searching for plotfiles in: $dir" >&2

    # Find all plotfile directories, sorted by name (which sorts numerically for this format)
    local plotfiles=$(find "$dir" -maxdepth 1 -type d -name "plt*" | sort -V)

    # Also check output subdirectory if it exists
    if [ -d "$dir/output" ]; then
        local output_plotfiles=$(find "$dir/output" -maxdepth 1 -type d -name "plt*" | sort -V)
        plotfiles="$plotfiles $output_plotfiles"
    fi

    echo "Found plotfiles: $plotfiles" >&2

    if [ -z "$plotfiles" ]; then
        echo ""
        return
    fi

    # Find the latest plotfile with a complete Header
    local last_plotfile=""
    for pltFile in $plotfiles; do
        echo "Checking plotfile: $pltFile" >&2
        if [ -f "${pltFile}/Header" ]; then
            last_plotfile=$pltFile
            echo "Valid plotfile found: $pltFile" >&2
        else
            echo "No Header found in: $pltFile" >&2
        fi
    done

    if [ ! -z "$last_plotfile" ]; then
        # Extract just the plotfile name relative to the search directory
        local plotfile_name=$(basename "$last_plotfile")
        echo "Returning plotfile: $plotfile_name" >&2
        echo "$plotfile_name"
    else
        echo "No valid plotfiles found" >&2
        echo ""
    fi
}

# Function to run a single simulation with proper output handling
function run_single_simulation {
    local run_counter=$1
    local param_values=("${@:2}")

    # Create subdirectory for this run
    local run_dir="run_$(printf "%04d" $run_counter)"
    echo "Creating directory: $run_dir" >&2
    mkdir -p "$run_dir"

    # Change to run directory
    cd "$run_dir"
    echo "Changed to directory: $(pwd)" >&2

    # Build command line arguments
    local cmd_args=""
    if [ ${#PARAM_NAMES[@]} -gt 0 ]; then
        for i in "${!PARAM_NAMES[@]}"; do
            if [ $i -lt ${#param_values[@]} ]; then
                # Don't quote the values for AMReX command line format
                cmd_args+="${PARAM_NAMES[$i]}=${param_values[$i]} "
            fi
        done
    fi

    # Build the full command
    local full_command="../$EXE ../$INPUTS $cmd_args"
    echo "Running: $full_command" >&2

    # Run the command and capture output
    $full_command > simulation.log 2>&1
    local exit_code=$?

    echo "Simulation exit code: $exit_code" >&2

    # Check if plotfiles were created (more reliable than exit code for AMReX)
    local plotfiles_created=$(find . -maxdepth 1 -type d -name "plt*" | wc -l)
    echo "Number of plotfiles created: $plotfiles_created" >&2

    if [ $plotfiles_created -gt 0 ]; then
        echo "Simulation succeeded - plotfiles created" >&2
        echo "Plotfiles found:" >&2
        ls -la plt*/ >&2
        cd ..
        # ONLY output the directory name to stdout for capture
        echo "$run_dir"
        return 0
    else
        echo "Simulation failed - no plotfiles created" >&2
        echo "Simulation log:" >&2
        cat simulation.log >&2
        cd ..
        return 1
    fi
}

# Function to process plotfile using combined Header/Cell_H approach
function process_single_plotfile {
    local run_dir=$1
    local run_counter=$2

    echo "Processing plotfile in $run_dir" >&2

    # Get the last plotfile generated
    local plotfile=$(get_last_plotfile "$run_dir")

    echo "Found plotfile: '$plotfile'" >&2

    if [ -z "$plotfile" ]; then
        echo "Error: No plotfile found in $run_dir" >&2
        return 1
    fi

    local plotfile_path="$run_dir/$plotfile"

    # Check required files exist
    if [ ! -f "$plotfile_path/Header" ]; then
        echo "Error: Header not found at $plotfile_path/Header" >&2
        return 1
    fi

    if [ ! -f "$plotfile_path/Level_${LEVEL}/Cell_H" ]; then
        echo "Error: Cell_H not found at $plotfile_path/Level_${LEVEL}/Cell_H" >&2
        return 1
    fi

    echo "Using plotfile path: $plotfile_path" >&2

    # Generate outnames.txt from combined Header/Cell_H if it doesn't exist
    if [ ! -f "outnames.txt" ]; then
        echo "Generating outnames.txt from Header variables + Cell_H components..." >&2
        generate_outnames_from_combined_sources "$plotfile_path" "$LEVEL"
        if [ -f "outnames.txt" ]; then
            echo "Created outnames.txt:" >&2
            cat outnames.txt >&2
        else
            echo "Error: Failed to create outnames.txt" >&2
            return 1
        fi
    fi

    # Extract min/max values
    local result=$(extract_minmax_from_cell_header "$plotfile_path" "$LEVEL" "$TARGET_I" "$TARGET_J" "$TARGET_K" "outnames.txt")

    if [ $? -eq 0 ]; then
        echo "Extracted result: '$result'" >&2
        echo "$result"
        return 0
    else
        echo "Error: Failed to extract data from $run_dir" >&2
        return 1
    fi
}

# Updated initialize_output_file function
function initialize_output_file {
    local output_file=$1

    echo "Initializing output file: $output_file" >&2
    > "$output_file"

    if [ "$INCLUDE_HEADER" = true ] && [ -f "outnames.txt" ]; then
        readarray -t output_names < outnames.txt
        echo "# ${output_names[@]}" >> "$output_file"
        echo "Added header to $output_file" >&2
    else
        echo "Created headerless output file: $output_file" >&2
    fi
}
