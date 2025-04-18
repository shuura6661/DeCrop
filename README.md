# DeCrop: YOLOv8 Auto Cropping Script

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

DeCrop is a Python utility script designed for automatically cropping specific objects from a batch of images using the YOLOv8 object detection model. It interactively prompts the user to select a single object class, processes images from an input directory, detects the chosen object, calculates a centered square crop around the first detected instance, and saves the cropped image to a class-specific output folder using the original filename.

## Features

* **Object Detection:** Utilizes the powerful YOLOv8 model via the `ultralytics` library.
* **Interactive Class Selection:** Prompts the user to choose a single COCO class ID to target for cropping after displaying the available classes.
* **Automatic Cropping:** Calculates a centered square crop (1:1 aspect ratio) around the *first* detected object instance.
* **Boundary Adjustment:** Intelligently adjusts the crop area to stay within the original image boundaries as much as possible while maintaining the square shape.
* **Organized Output:** Automatically creates a subdirectory named after the selected class within a base output directory to store the results.
* **Original Filenames:** Saves the cropped images using the same filename as the original input image. **(Warning: If multiple instances of the target class exist in one input image, only the crop from the *last* instance processed will be saved due to overwriting).**
* **Progress Tracking:** Displays a visual progress bar during processing using `tqdm`.
* **Summary Report:** Prints a summary of processed files, successes, failures, and total time taken.
* **Failure Logging:** Creates a text file (`failed_crops.txt` by default) listing images that failed during processing (e.g., no detection, read error, save error).

## Requirements

* **Python:** Version 3.8 or newer recommended.
* **Python Libraries:**
    * `opencv-python`
    * `numpy`
    * `ultralytics` (which includes `torch`, `torchvision`, etc.)
    * `tqdm`
    * `Pillow` (often required by `ultralytics`/`torchvision`)
* **YOLOv8 Model File:** A pre-trained YOLOv8 model file (e.g., `yolov8s.pt`, `yolov8m.pt`). Download from the [Ultralytics YOLOv8 GitHub Releases](https://github.com/ultralytics/yolov8/releases) if needed.

## Installation

1.  **Clone the repository (if applicable) or download the script.**
2.  **Install required libraries:** Open your terminal or command prompt (ideally within a Python virtual environment) and run:
    ```bash
    pip install opencv-python numpy ultralytics tqdm Pillow
    ```

## Configuration

Before running the script, you **must** configure the paths and parameters within the Python file (`.py`):

1.  **Open the script** in a text editor or IDE.
2.  Locate the `# --- Constants ---` section near the top.
3.  **Modify the following constants** to match your system setup:
    * `MODEL_PATH`: Set the full path to your downloaded or custom YOLOv8 model file (e.g., `r'C:\models\yolov8s.pt'` or `/home/user/models/yolov8s.pt`).
    * `INPUT_DIR`: Set the full path to the directory containing the images you want to process.
    * `OUTPUT_DIR_BASE`: Set the full path to the base directory where the script will create class-specific subfolders for the output crops.
    * `CONFIDENCE_THRESHOLD`: (Optional) Adjust the minimum confidence score (0.0 to 1.0) required for object detection. Default is 0.05.
    * `FAILED_LOG_FILE`: (Optional) Change the name for the log file if desired. Default is `failed_crops.txt`.
    * `BORDER_WIDTH`: (Optional) Adjust the width for console separators. Default is 80.
4.  **Save the script file.**

## Usage (Tutorial)

1.  **Navigate to Script Directory:** Open your terminal or command prompt and use the `cd` command to go to the directory where you saved the script file.
    ```bash
    cd path/to/your/script/directory
    ```
2.  **Run the Script:** Execute the script using Python. Replace `your_script_name.py` with the actual filename.
    ```bash
    python your_script_name.py
    ```
3.  **Select Target Class:**
    * The script will load the YOLOv8 model.
    * It will then display a list of **Available COCO Classes (ID: Name)** formatted in columns.
    * You will be prompted: `Enter the SINGLE Class ID you want to detect and crop.`
    * Examine the list and type the **numeric ID** corresponding to the object class you want to crop (e.g., `47` for `apple`, `11` for `stop sign`). Press `Enter`.
    * You can type `quit` to exit. If you enter an invalid ID, you will be prompted again.
4.  **Processing:**
    * The script will confirm the selected class and display the final configuration, including the specific output subfolder it will use (e.g., `.../bgremove_single_class/apple`).
    * It will search for images in the specified `INPUT_DIR`.
    * A `tqdm` progress bar will appear, showing the cropping progress for the selected class. Errors for specific files might be printed above the bar using `tqdm.write`.
5.  **Review Results:**
    * Once finished, a summary will be printed to the console showing processing statistics and total time.
    * Navigate to the `OUTPUT_DIR_BASE` folder you configured. Inside, you will find a **new subfolder** named after the class you selected (e.g., `apple`, `stop_sign`).
    * This subfolder contains the **cropped images**, saved with their **original filenames**.
    * The subfolder also contains the **`failed_crops.txt`** file (if any errors occurred), listing the input files that failed processing and the reason.

    **Important Note on Output:** As the script saves the output crop using the original input filename, if an input image contains multiple instances of the *same* target object, only the crop corresponding to the *first* detected instance (usually the one with the highest confidence) will be saved.

## License

This project uses the `ultralytics` library which is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. Therefore, this script is also likely subject to the terms of the AGPL-3.0 license if distributed. Please review the AGPL-3.0 license terms. If AGPL-3.0 is not suitable for your use case, consider obtaining a commercial license from Ultralytics.

---

Remember to replace placeholders like `your_script_name.py` and potentially add more details specific to your project goals or environment.
