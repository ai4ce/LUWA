# Feature Extraction and Distance Matrix Calculation

This Python script is designed for performing feature extraction and distance matrix calculation for a given dataset of images. It supports various models for feature extraction, norm types for distance calculation, and allows customization of parameters through command-line arguments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Customization](#customization)
- [Folder Structure](#folder-structure)
- [Output](#output)
- [Acknowledgments](#acknowledgments)

## Prerequisites

Before using this script, ensure you have the following dependencies installed:

- Python 3.x
- PyTorch
- torchvision
- NumPy
- PIL (Python Imaging Library)
- tqdm

You can typically install these dependencies using `pip`:

```bash
pip install torch torchvision numpy pillow tqdm
```

## Usage
To use this script, follow these steps:

1. Clone this repository or download the script to your local machine.
    ```
    git clone xxx
    ```
2. Navigate to the directory where the script is located:
    ```
    cd xxx
    ```

3. Run the script with the desired parameters. Here's the basic usage:
    ```
    python script_name.py --model [model] --image_type [image_type] --norm [norm_type] --csv [csv_file] --batch_size [batch_size] --data_root [data_root_directory]
    ```

    For example, 

    ```
    python main.py --model resnet18 --image_type texture --norm l2 --csv "CSV/256_20x.csv" 
    ```
4. Wait for the script to perform feature extraction and distance matrix calculation.
5. The results, including the distance matrix, will be saved in the data folder in the current directory.


## Customization

* Model Selection (--model): You can choose from the following models for feature extraction:
    - resnet18
    - alexnet
    - convnext
    - vgg11
    - vit
    - dinov2
* Image Type (--image_type): Specify the type of images in your dataset. Choose between "heightmap" and "texture."

* Norm Type (--norm): Select the norm type for distance calculation. Currently, only "l2" (Euclidean distance) is supported.

* CSV File (--csv): Provide the path to the CSV file containing image information.

* Batch Size (--batch_size): Set the batch size for feature extraction. The default is 10.

* Data Root Directory (--data_root): Specify the root directory of your dataset.

## Output
The script will generate the following output:

- A distance matrix saved as a NumPy array in the data folder. The filename format is csv_filename_image_type_model_norm.npy.

- Information about the image closest to the mean feature vector will be saved in a file called "closest_to_mean.txt."