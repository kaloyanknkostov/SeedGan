#!/bin/bash

# Script: strip_txt.sh
# Description: Finds all .txt files in the current directory and replaces 
#              all occurrences of commas (,) with spaces ( ) within those files.
#
# WARNING: This script modifies the files IN PLACE. It's recommended to back up 
#          your data before running.

echo "--- Starting batch processing of .txt files ---"

# Loop through all files matching the pattern "*.txt"
for file in *.txt; do
    # Check if the file exists (to handle the case where no .txt files are found)
    if [ -f "$file" ]; then
        echo "Processing file: $file"
        
        # Use sed to perform the substitution:
        # 's/,/ /g' : substitute (s) comma (,) with space ( ) globally (g)
        # -i      : edits the file in place (be careful!)
        sed -i 's/,/ /g' "$file"
        
        echo "Successfully replaced commas with spaces in $file."
    fi
done

echo "--- Batch processing complete ---"

