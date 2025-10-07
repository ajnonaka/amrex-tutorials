#!/bin/bash
# postprocess_datalog.sh - Extract data from datalog.txt

# Arguments
RUN_DIR="$1"
RUN_COUNTER="${2:-0}"
OUTNAMES_FILE="${3:-outnames.txt}"

# Check if run directory exists
if [ ! -d "$RUN_DIR" ]; then
    echo "Error: Directory $RUN_DIR not found" >&2
    exit 1
fi

DATALOG="$RUN_DIR/datalog.txt"

# Check if datalog exists
if [ ! -f "$DATALOG" ]; then
    echo "Error: $DATALOG not found" >&2
    exit 1
fi

# Generate outnames.txt from header on first run
if [ "$RUN_COUNTER" -eq 0 ] && [ ! -f "$OUTNAMES_FILE" ]; then
    # Extract header line, remove #, clean up spaces
    header=$(grep '^#' "$DATALOG" | head -1 | sed 's/^#//' | tr -s ' ' | sed 's/^ *//' | sed 's/ *$//')

    if [ -n "$header" ]; then
        echo "$header" | tr ' ' '\n' > "$OUTNAMES_FILE"
        echo "Generated $OUTNAMES_FILE from datalog header" >&2
    fi
fi

# Extract last non-comment line
result=$(grep -v '^#' "$DATALOG" | tail -1)

if [ -z "$result" ]; then
    echo "Error: No data found in $DATALOG" >&2
    exit 1
fi

# Output the result
echo "$result"
