"""
Convert Left, Top, Right, Bottom, Label ID, cordinates  to Label id x_center,y_center,width,height
    """

import argparse
import os
from pathlib import Path

import pandas as pd


def convert_to_yolo_format(df: pd.DataFrame, W: int, H: int) -> pd.DataFrame:
    df.columns = ["x_min", "y_min", "x_max", "y_max", "class_id"]
    # Calculate center x and center y (normalized by W and H)
    # x_center = ((x_min + x_max) / 2) / W
    df["x_center"] = ((df["x_min"] + df["x_max"]) / 2) / W
    # y_center = ((y_min + y_max) / 2) / H
    df["y_center"] = ((df["y_min"] + df["y_max"]) / 2) / H
    # Calculate box width and box height (normalized by W and H)
    # w_norm = (x_max - x_min) / W
    df["w_norm"] = (df["x_max"] - df["x_min"]) / W
    # h_norm = (y_max - y_min) / H
    df["h_norm"] = (df["y_max"] - df["y_min"]) / H
    yolo_df = df[["class_id", "x_center", "y_center", "w_norm", "h_norm"]].copy()
    yolo_df.iloc[:, 1:] = yolo_df.iloc[:, 1:].round(6)
    return yolo_df


def main():
    IMAGE_WIDTH = 1920
    IMAGE_HEIGHT = 1088
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
    for item in folder_path.iterdir():
        print(item.name)
        df = pd.read_csv(item, header=None)
        df = convert_to_yolo_format(df, W=IMAGE_WIDTH, H=IMAGE_HEIGHT)
        df.to_csv(item, index=False, header=False)


if __name__ == "__main__":
    main()
