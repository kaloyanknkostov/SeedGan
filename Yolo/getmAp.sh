#!/bin/bash

# Default file is results.csv, but you can pass a different one as an argument
FILE="${1:-results.csv}"
COLUMN_NAME="metrics/mAP50-95(B)"

if [ ! -f "$FILE" ]; then
    echo "Error: File '$FILE' not found."
    exit 1
fi

# Use awk to find the column index dynamically and print it
awk -F, -v target="$COLUMN_NAME" '
    # Process the first line (Header)
    NR==1 {
        for (i=1; i<=NF; i++) {
            # Strip whitespace just in case
            gsub(/^[ \t]+|[ \t]+$/, "", $i)
            if ($i == target) {
                col_index = i
                break
            }
        }
        if (col_index == "") {
            print "Error: Column \"" target "\" not found in " FILENAME > "/dev/stderr"
            exit 1
        }
        # Print the header (optional, comment out if you only want numbers)
        print $col_index
    }
    # Process the rest of the lines
    NR>1 && col_index {
        # Strip whitespace from the value and print
        gsub(/^[ \t]+|[ \t]+$/, "", $col_index)
        print $col_index
    }
' "$FILE"
