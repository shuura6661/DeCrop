import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time
from tqdm import tqdm # Import tqdm for progress bar
import textwrap # For nicely printing class list
import re # For sanitizing class name for folder
import os # Added for file operations consistency if needed, though pathlib is primary

# --- Constants ---
# --------------------------------------------------------------------------
#                              CONFIGURATION
# --------------------------------------------------------------------------
# Path to the YOLOv8 model file (e.g., yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
MODEL_PATH = r'Preprocessing Data\0. Source Code\models\yolov8s.pt'

# Directory containing the input images
INPUT_DIR = Path(r'Preprocessing Data\1. Automatically Crop\input\Bingung')

# Base directory where class-specific output folders will be created
OUTPUT_DIR_BASE = Path(r'Preprocessing Data\1. Automatically Crop\output\Bingung')

# Minimum confidence score for YOLO detections (0.0 to 1.0)
CONFIDENCE_THRESHOLD = 0.05

# Name for the log file listing failed processing attempts
FAILED_LOG_FILE = 'failed_crops.txt'

# Width for printing separators and centering text in the console output
BORDER_WIDTH = 80
# --------------------------------------------------------------------------

# --- Helper Functions ---

def print_separator(char: str = "-", width: int = BORDER_WIDTH):
    """Prints a separator line to the console."""
    print(char * width)

def setup_directories(output_dir: Path) -> bool:
    """Creates the specified output directory if it doesn't exist.

    Args:
        output_dir (Path): The directory path to create.

    Returns:
        bool: True if the directory exists or was created successfully, False otherwise.
    """
    try:
        if not output_dir.exists():
            print(f"⏳ Creating output directory: {output_dir}")
            # Create parent directories if needed, don't raise error if it exists
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            print(f"📂 Output directory already exists: {output_dir}")
        return True
    except OSError as e:
        print(f"❌ Error creating directory {output_dir}: {e}")
        return False
    except Exception as e: # Catch other potential errors
        print(f"❌ Unexpected error setting up directory {output_dir}: {e}")
        return False

def calculate_square_crop(img_shape: tuple, box_int: np.ndarray) -> tuple | None:
    """Calculates centered square crop coordinates around a bounding box.

    Args:
        img_shape (tuple): Shape of the original image (height, width, channels).
        box_int (np.ndarray): Bounding box [x1, y1, x2, y2] as integers.

    Returns:
        tuple | None: Crop coordinates (x1, y1, x2, y2) or None if the box is invalid.
    """
    img_height, img_width = img_shape[:2]
    x1, y1, x2, y2 = box_int # Input box is expected to be integer NumPy array

    box_width = x2 - x1
    box_height = y2 - y1

    # Basic validation for the bounding box dimensions
    if box_width <= 0 or box_height <= 0:
        return None

    # Determine the side length of the square crop (max dimension of the box)
    size = max(box_width, box_height)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    half_size = size // 2

    # Calculate initial square coordinates centered on the object
    crop_x1 = center_x - half_size
    crop_y1 = center_y - half_size
    crop_x2 = crop_x1 + size # Use size directly for consistency
    crop_y2 = crop_y1 + size

    # --- Boundary Adjustments ---
    # Calculate how much the square goes out of the original image bounds
    over_x1 = max(0, -crop_x1)
    over_y1 = max(0, -crop_y1)
    over_x2 = max(0, crop_x2 - img_width)
    over_y2 = max(0, crop_y2 - img_height)

    # Shift the crop window back into bounds, trying to maintain centering
    # Note: If the required shift is larger than image allows, clipping occurs later
    crop_x1 += over_x1 - over_x2
    crop_y1 += over_y1 - over_y2
    crop_x2 += over_x1 - over_x2
    crop_y2 += over_y1 - over_y2

    # --- Final Clipping ---
    # Ensure coordinates are strictly within the image boundaries
    final_x1 = max(0, crop_x1)
    final_y1 = max(0, crop_y1)
    final_x2 = min(img_width, crop_x2)
    final_y2 = min(img_height, crop_y2)

    # Final check if the resulting crop has valid dimensions
    if final_x2 <= final_x1 or final_y2 <= final_y1:
        return None

    return final_x1, final_y1, final_x2, final_y2

def save_failed_list(failed_files: set, output_dir: Path, filename: str):
    """Saves the list of failed image/object details to a text file."""
    if not failed_files:
        print("\n✅ No processing failures recorded.")
        return

    filepath = output_dir / filename
    print(f"\n⚠️ Saving list of {len(failed_files)} failed processing items to: {filepath}")
    try:
        # Sort items for consistent log output
        sorted_items = sorted(list(failed_files))
        with open(filepath, 'w') as f:
            for item in sorted_items:
                f.write(f"{item}\n")
    except IOError as e:
        print(f"❌ Error writing failed log file '{filepath}': {e}")
    except Exception as e:
        print(f"❌ Unexpected error writing failed log file '{filepath}': {e}")

def print_final_summary(success_count: int, failed_detection_count: int, error_skip_count: int, total_files: int, duration: float):
    """Prints the final processing summary with aligned formatting."""
    total_attempted = success_count + failed_detection_count + error_skip_count
    # Define success rate based on images where detection was expected & crop succeeded
    # Exclude skipped images from the rate calculation base? Or include? User decision.
    # Current: Success rate based on (Succeeded / (Succeeded + Failed Detection))
    success_rate = (success_count / (success_count + failed_detection_count) * 100) if (success_count + failed_detection_count) > 0 else 0

    # Format duration
    total_seconds_int = int(duration)
    minutes, seconds = divmod(total_seconds_int, 60)

    print_separator()
    print("🏁 Processing Complete 🏁".center(BORDER_WIDTH))
    print_separator()
    # Use a consistent label width for alignment
    summary_label_width = 35 # Adjusted width
    print(f"{'📂 Input files found':<{summary_label_width}}: {total_files}")
    print(f"{'✅ Images cropped successfully (first obj)':<{summary_label_width}}: {success_count}")
    print(f"{'❌ Images with no matching detection':<{summary_label_width}}: {failed_detection_count}")
    print(f"{'➖ Images skipped (read/crop/save error)':<{summary_label_width}}: {error_skip_count}")
    # print(f"{'📊 Detection/Crop Success Rate':<{summary_label_width}}: {success_rate:.2f}%") # Optional rate
    print(f"{'⏱️ Total time taken':<{summary_label_width}}: {minutes} minutes {seconds} seconds")
    print_separator()

def display_classes(class_names_dict: dict):
    """Displays the available COCO classes formatted in exactly 4 columns with separators."""
    print("\nAvailable COCO Classes (ID: Name):")
    print_separator('.') # <-- Separator added back after the heading

    if not class_names_dict:
        print("   <No class names loaded>")
        print_separator('.') # Separator added back here too
        return # Exit if no classes

    # Sort items by ID for consistent order
    items = [f"{id_}: {name}" for id_, name in sorted(class_names_dict.items())]
    if not items:
        print("   <No classes available>")
        print_separator('.') # Separator added back here too
        return # Exit if empty after formatting

    # Determine padding based on the longest item string for alignment
    try:
         max_item_len = max(len(item) for item in items) + 3 # Add padding space (increased slightly)
    except ValueError:
         print("   <Error calculating item length>")
         print_separator('.') # Separator added back here too
         return

    # --- Force 4 Columns ---
    num_cols = 4
    num_rows = (len(items) + num_cols - 1) // num_cols # Calculate rows needed for 4 columns

    # Print items arranged in columns
    for r in range(num_rows):
        line_items = []
        for c in range(num_cols):
            # Calculate index based on row and column (filling columns first)
            idx = r + c * num_rows
            if idx < len(items):
                line_items.append(items[idx])
            else:
                line_items.append("") # Add empty string if cell is empty

        # Join items for the line, padding each to max_item_len for alignment
        # Use rstrip() to remove trailing spaces from the last item in the line
        print("".join(item.ljust(max_item_len) for item in line_items).rstrip())

    print_separator('.') # <-- Separator added back after the list

def get_target_id_from_user(available_classes: dict) -> int | None:
    """Prompts the user for a SINGLE class ID and validates it."""
    if not available_classes:
        print("❌ Cannot get target ID: No available classes loaded.")
        return None

    valid_ids_set = set(available_classes.keys())
    min_id = min(valid_ids_set) if valid_ids_set else 0
    max_id = max(valid_ids_set) if valid_ids_set else -1

    while True:
        print("\nEnter the SINGLE Class ID you want to detect and crop.")
        prompt = f"Enter one ID between {min_id} and {max_id} (or 'quit'): "
        input_str = input(prompt).strip()

        if not input_str:
            print("❌ Input cannot be empty. Please try again.")
            continue

        if input_str.lower() == 'quit':
            print("🛑 User requested exit.")
            return None

        try:
            selected_id = int(input_str)
            if selected_id in valid_ids_set:
                print(f"✅ Selected Class ID: {selected_id} ({available_classes[selected_id]})")
                return selected_id # Return the single valid ID
            else:
                print(f"❌ Error: Unknown or invalid class ID entered: {selected_id}")
                print(f"   Please use an ID from the list above ({min_id}-{max_id}).")
        except ValueError:
            print(f"❌ Error: Input must be a single whole number. You entered: '{input_str}'")

        print("   Please try again.")


def sanitize_foldername(name: str) -> str:
    """Removes or replaces characters invalid for directory/file names."""
    if not isinstance(name, str): # Ensure input is a string
        name = str(name)
    name = name.strip() # Remove leading/trailing whitespace
    # Replace spaces and potentially problematic characters with underscores
    # Removed: \ / * ? : " < > |
    name = re.sub(r'[\\/*?:"<>|\s]+', '_', name)
    # Replace multiple consecutive underscores with a single one
    name = re.sub(r'_+', '_', name)
    # Limit length if needed (optional)
    # max_len = 50
    # name = name[:max_len]
    return name if name else "unknown_class"

# --- Main Execution ---

def main():
    """Main function to run the interactive auto-cropping process for a single class."""
    print_separator('=')
    print("🚀 Starting YOLOv8 Auto Cropping Script (Single Class - First Object) 🚀".center(BORDER_WIDTH))
    print_separator('=')

    # --- Load Model ---
    try:
        print(f"🧠 Loading YOLOv8 model: {MODEL_PATH}...")
        model = YOLO(MODEL_PATH)
        # Verify model has class names (essential for this script)
        if not hasattr(model, 'names') or not isinstance(model.names, dict) or not model.names:
             raise ValueError("Model loaded successfully, but '.names' attribute is missing, empty, or not a dictionary.")
        print("✅ Model loaded successfully.")
        available_classes = model.names # Get {id: name} mapping
    except Exception as e:
        print(f"❌ Fatal Error: Could not load YOLO model: {e}")
        return # Exit script if model fails to load

    # --- Get SINGLE Target Class ID From User ---
    display_classes(available_classes)
    selected_class_id = get_target_id_from_user(available_classes)
    # Exit if user chose 'quit' or if no valid ID was provided
    if selected_class_id is None:
        print("\n🛑 Exiting script as requested or no valid class selected.")
        return

    # --- Determine Output Directory ---
    selected_class_name = available_classes.get(selected_class_id, f"unknown_{selected_class_id}")
    safe_class_folder_name = sanitize_foldername(selected_class_name)
    final_output_dir = OUTPUT_DIR_BASE / safe_class_folder_name

    # --- Print Configuration ---
    print_separator()
    print("CONFIGURATION".center(BORDER_WIDTH))
    print_separator('.')
    label_width = 22
    print(f"{'🔧 Model Path':<{label_width}}: {MODEL_PATH}")
    print(f"{'📁 Input Folder':<{label_width}}: {INPUT_DIR}")
    print(f"{'📂 Base Output Folder':<{label_width}}: {OUTPUT_DIR_BASE}")
    print(f"{'🎯 Target Class':<{label_width}}: {selected_class_id} ({selected_class_name})")
    print(f"{'💾 Saving Crops To':<{label_width}}: {final_output_dir}")
    print(f"{'📈 Conf Threshold':<{label_width}}: {CONFIDENCE_THRESHOLD}")
    print_separator()

    # --- Setup Output Directory ---
    start_time = time.time()
    if not setup_directories(final_output_dir):
        print("🛑 Exiting script due to directory setup failure.")
        return # Stop if output directory can't be created

    # --- Image Discovery ---
    print(f"\n🔍 Searching for images in: {INPUT_DIR}...")
    try:
        # More robust way to find common image types
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff',
                            '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIF', '*.TIFF'] # Case-insensitive
        image_files = []
        for ext in image_extensions:
            image_files.extend(INPUT_DIR.glob(ext))

        # Remove potential duplicates and sort
        image_files = sorted(list(set(image_files)))

        if not image_files:
            print(f"⚠️ No images with supported extensions found in {INPUT_DIR}.")
            print("   Supported extensions:", ', '.join(image_extensions))
            return # Exit if no images found
        print(f"✨ Found {len(image_files)} potential image(s) to process.")
    except Exception as e:
        print(f"❌ Error searching for images: {e}")
        return # Exit on error

    # --- Processing ---
    # Counters track image-level results based on processing the *first* detection
    img_success_count = 0       # Images where the first detection was successfully cropped/saved
    img_no_detection_count = 0  # Images where no objects of the target class were detected
    img_error_count = 0         # Images skipped due to read errors or errors processing the first object
    failed_object_details = set() # Log details about failures

    print_separator()
    print(f"⏳ Processing Images for Class: '{selected_class_name}' (First Detection Only)...")

    # Main processing loop with progress bar
    for img_path in tqdm(image_files, desc=f"Cropping {selected_class_name}", unit="image", ncols=BORDER_WIDTH, leave=True):
        image_processed_successfully = False # Reset flag for each image
        try:
            # Step 1: Read Image
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError("Image corrupt or unreadable")

            # Step 2: Perform detection ONLY for the selected class
            # Note: augment=True slows down inference but might improve detection slightly
            results = model.predict(img, classes=[selected_class_id], conf=CONFIDENCE_THRESHOLD, verbose=False, augment=True)

            # Step 3: Check for detections
            detections = results[0].boxes
            if detections is None or len(detections) == 0:
                img_no_detection_count += 1
                failed_object_details.add(f"{img_path.name} (No target detected)")
                continue # Skip to next image if no relevant objects found

            # Step 4: Process ONLY THE FIRST DETECTION
            box_data = detections[0] # Highest confidence detection
            box_coords_tensor = box_data.xyxy[0]

            # Step 5: Convert box coordinates (apply Tensor fix here)
            try:
                box_coords_numpy_int = box_coords_tensor.cpu().numpy().astype(int)
            except AttributeError:
                 # Should ideally not happen if results[0].boxes exists, but safety check
                 raise TypeError("Detected box data is not a convertible tensor")

            # Step 6: Calculate square crop coordinates
            crop_coords = calculate_square_crop(img.shape, box_coords_numpy_int)
            if crop_coords is None:
                raise ValueError("Invalid box dimensions for cropping")

            # Step 7: Perform the crop
            x1, y1, x2, y2 = crop_coords
            cropped_img = img[y1:y2, x1:x2]
            if cropped_img is None or cropped_img.shape[0] <= 0 or cropped_img.shape[1] <= 0:
                raise ValueError("Crop resulted in empty image")

            # Step 8: Save the cropped image (using original filename)
            output_filename = img_path.name # Use original name
            output_path = final_output_dir / output_filename
            # cv2.imwrite needs path as string
            save_success = cv2.imwrite(str(output_path), cropped_img)
            if not save_success:
                 raise IOError(f"Failed to save image to {output_path}")

            # If all steps above succeeded without exception
            image_processed_successfully = True

        # --- Consolidated Error Handling for this image ---
        except (ValueError, TypeError, IOError) as ProcErr: # Catch specific processing errors
             tqdm.write(f"⚠️ Skipping {img_path.name}: {ProcErr}")
             failed_object_details.add(f"{img_path.name} ({ProcErr})")
             img_error_count += 1
             continue # Continue with the next image
        except Exception as e: # Catch any other unexpected errors
            tqdm.write(f"❌ Unexpected error processing {img_path.name}: {e}")
            import traceback
            tqdm.write(traceback.format_exc()) # Print traceback for unexpected errors
            img_error_count += 1
            failed_object_details.add(f"{img_path.name} (Runtime Error: {e})")
            continue # Continue with the next image
        finally:
             # Update success count based on the flag
             if image_processed_successfully:
                 img_success_count += 1

    # --- Final Steps ---
    save_failed_list(failed_object_details, final_output_dir, FAILED_LOG_FILE)
    end_time = time.time()
    duration = end_time - start_time
    # Ensure final summary is printed below the progress bar
    print() # Print newline to avoid summary overlapping with tqdm bar
    print_final_summary(img_success_count, img_no_detection_count, img_error_count, len(image_files), duration)


# --- Script Entry Point ---
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user (Ctrl+C). Exiting.")
    except Exception as GlobalErr:
        print(f"\n❌ An unexpected global error occurred: {GlobalErr}")
        import traceback
        print(traceback.format_exc())
    finally:
        print("\nScript execution finished.")
        # Optional: Keep console open if double-clicked
        # if sys.stdin.isatty(): # Only pause if run in interactive terminal
        #      input("Press Enter to exit...")
