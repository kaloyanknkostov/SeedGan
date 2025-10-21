import cv2
import pandas as pd
import numpy as np

def visualize_bounding_boxes(image_data: np.ndarray, csv_data: pd.DataFrame):
    """
    Draws bounding boxes onto an image based on normalized YOLO-style CSV data.

    Args:
        image_data (np.ndarray): The OpenCV image (H, W, 3) as a NumPy array.
        csv_data (pd.DataFrame): DataFrame containing normalized bounding box data.
                                 Expected columns: ['class_id', 'x_center_norm', 'y_center_norm', 'w_norm', 'h_norm']
    """
    image = image_data.copy()
    H, W, _ = image.shape # Get image dimensions (Height, Width)
    
    # Define the class names and colors (in BGR format for OpenCV)
    class_map = {
        0: {"name": "Class 0", "color": (255, 0, 0)},   # Blue 
        1: {"name": "Class 1", "color": (0, 255, 255)}, # Yellow
    }
    
    # Assign the expected column names to the loaded CSV data
    csv_data.columns = ['class_id', 'x_center_norm', 'y_center_norm', 'w_norm', 'h_norm']
    
    for index, row in csv_data.iterrows():
        try:
            # Safely convert class_id to integer
            class_id = int(row['class_id'])
            
            # --- Conversion: Normalized YOLO -> Absolute Pixels ---
            # Normalized values
            x_c_norm = row['x_center_norm']
            y_c_norm = row['y_center_norm']
            w_norm = row['w_norm']
            h_norm = row['h_norm']
            
            # Absolute dimensions (center, width, height)
            x_center = x_c_norm * W
            y_center = y_c_norm * H
            box_width = w_norm * W
            box_height = h_norm * H
            
            # Absolute corners (x_min, y_min, x_max, y_max)
            x_min = int(x_center - box_width / 2)
            y_min = int(y_center - box_height / 2)
            x_max = int(x_center + box_width / 2)
            y_max = int(y_center + box_height / 2)
            
            # Clamp coordinates to image boundaries
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(W - 1, x_max)
            y_max = min(H - 1, y_max)

            # Look up class details
            if class_id not in class_map:
                print(f"Warning: Class ID {class_id} not defined in class_map. Skipping.")
                continue

            label = class_map[class_id]["name"]
            color = class_map[class_id]["color"]
            
            # --- A. Draw the Bounding Box (Rectangle) ---
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            
            # --- B. Add the Class Label Text ---
            font_scale = 0.7
            thickness = 2
            
            # Get text size to create a background box
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Draw a filled rectangle slightly above the box for the label background
            cv2.rectangle(
                image, 
                (x_min, y_min - text_h - 10),  # Top-left corner of background
                (x_min + text_w + 5, y_min),   # Bottom-right corner of background
                color, 
                cv2.FILLED
            )

            # Put the text on top of the background
            cv2.putText(
                image, 
                label, 
                (x_min + 5, y_min - 5), # Position text slightly inside the background box
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (0, 0, 0), # Text color is black (BGR)
                thickness
            )

        except Exception as e:
            print(f"Error processing row {index}: {e}")
            continue

    # 3. Display the Result
    cv2.imshow("Bounding Box Visualizer", image)
    print("Displaying image. Press any key to close the window...")
    cv2.waitKey(0) # Wait infinitely for a key press
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
   IMAGE_PATH = '/home/kaloyan/Code/Thesis/data/SeedGanData/original/imagesAfter/ave-0035-0002.jpg'
   CSV_PATH = '/home/kaloyan/Code/Thesis/data/SeedGanData/original/labelsAfter/ave-0035-0002.csv'
   image_data = cv2.imread(IMAGE_PATH)
   bbox_data = pd.read_csv(CSV_PATH, header=None)
   visualize_bounding_boxes(image_data, bbox_data)

