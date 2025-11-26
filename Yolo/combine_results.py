import argparse
import os
import sys

import pandas as pd


def get_csv_files(inputs):
    """
    Scans a list of inputs.
    If input is a file, use it.
    If input is a directory, find all 'results.csv' inside it recursively.
    """
    csv_files = []

    for input_path in inputs:
        if os.path.isfile(input_path):
            csv_files.append(input_path)
        elif os.path.isdir(input_path):
            print(f"Scanning directory: {input_path}...", file=sys.stdout)
            # Walk through directory to find results.csv
            for root, dirs, files in os.walk(input_path):
                for file in files:
                    if file == "results.csv":
                        csv_files.append(os.path.join(root, file))
        else:
            print(f"Warning: '{input_path}' does not exist.", file=sys.stderr)

    # Sort files to ensure columns are in a logical order (e.g. train, train2, train3)
    return sorted(csv_files)


def main():
    parser = argparse.ArgumentParser(description="Combine YOLO CSV columns.")
    parser.add_argument("inputs", nargs="+", help="Files or Directories to search")
    parser.add_argument(
        "-c",
        "--column",
        type=str,
        default="metrics/mAP50-95(B)",
        help="The column name to extract (default: metrics/mAP50-95(B))",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="combined.csv",
        help="Output filename (default: combined.csv)",
    )

    args = parser.parse_args()

    # 1. Gather all files
    files_to_process = get_csv_files(args.inputs)

    if not files_to_process:
        print("No csv files found!", file=sys.stderr)
        return

    print(f"Found {len(files_to_process)} files. Processing...")

    collected_data = {}

    # 2. Extract Data
    for file_path in files_to_process:
        try:
            df = pd.read_csv(file_path)

            # Clean whitespace from headers
            df.columns = df.columns.str.strip()

            if args.column not in df.columns:
                print(
                    f"Skipping {file_path}: Column '{args.column}' not found.",
                    file=sys.stderr,
                )
                continue

            # Get folder name for the header
            folder_name = os.path.basename(os.path.dirname(os.path.abspath(file_path)))

            # Handle duplicate folder names (e.g. if you have two 'train' folders in different paths)
            original_name = folder_name
            counter = 2
            while folder_name in collected_data:
                folder_name = f"{original_name}_{counter}"
                counter += 1

            collected_data[folder_name] = df[args.column]

        except Exception as e:
            print(f"Error reading {file_path}: {e}", file=sys.stderr)

    # 3. Save to File
    if collected_data:
        result_df = pd.DataFrame(collected_data)
        result_df.to_csv(args.output, index=False)
        print(f"Success! Combined data saved to: {args.output}")
    else:
        print("No valid data found to save.", file=sys.stderr)


if __name__ == "__main__":
    main()
