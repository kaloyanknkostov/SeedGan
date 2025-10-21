"""
Convert all classes to weed or crop remove vegitation and remove stem cordinates
    """

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    # Entering the folder
    # ____________________________

    parser = argparse.ArgumentParser(
        description="Process images and labels in a specified directory."
    )

    parser.add_argument(
        "input_folder",
        type=Path,  # Use pathlib.Path for easy file system operations
        help="The path to the root directory containing the dataset (e.g., /home/user/Code/Thesis/data/SeedGanData/original/labels).",
    )
    args = parser.parse_args()
    folder_path = args.input_folder
    if not folder_path.is_dir():
        print(f"Error: The path '{folder_path}' does not exist or is not a directory.")
        return
    # Checking the folder
    # -----------------------------------------------------------------------------------------------------------------------------------------
    labels = folder_path / "labels"
    images = folder_path / "images"
    if not (labels.is_dir() and images.is_dir()):
        print("Wrong folder")
        return
    labelsAfter = folder_path / "labelsAfter"
    imagesAfter = folder_path / "imagesAfter"
    if not (labelsAfter.is_dir() and imagesAfter.is_dir()):
        os.mkdir(labelsAfter)
        os.mkdir(imagesAfter)

    removed = 0
    for item in labels.iterdir():
        print(item.name)
        if os.path.getsize(item) == 0:
            removed = removed + 1
        if os.path.getsize(item) != 0:
            df = pd.read_csv(item, header=None)
            df.drop(columns=[5, 6], axis=1, inplace=True)
            mask_to_keep = (df[4] != 0) & (df[4] != 255)
            df = df[mask_to_keep].copy()
            conditions = [
             (df[4] < 29) | (df[4] == 93) | (df[4] == 94),  
             df[4] >= 29                                 
            ]
            choices = [0, 1]
            df[4] = np.select(conditions, choices, default=df[4])
            if not df.empty:
                df.to_csv(labelsAfter / item.name, index=False, header=False)
                copy = item.with_suffix(".jpg")
                shutil.copy2(images / copy.name, imagesAfter)


if __name__ == "__main__":
    main()
