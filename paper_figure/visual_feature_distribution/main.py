import os, sys
import csv
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from tqdm import tqdm
import argparse
from models import *


def read_image_tensor(image_path):
    image = Image.open(image_path)

    transform = transforms.ToTensor()
    tensor_image = transform(image)

    return tensor_image

def read_csv_file(file_path):
    data = []
    with open(file_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        next(reader)  # Skip the header row
        for row in reader:
            data.append(row)
    return data

def csv_to_path(file_path, image_type="texture"):
    path = []
    # read csv file
    csv_data = read_csv_file(file_path)
    # check image type
    assert image_type in ["texture", "heightmap"]
    # use loop to generate path
    RES = os.path.basename(file_path)[:-4].split("_")[0]
    ZOOM = os.path.basename(file_path)[:-4].split("_")[1]
    for row in csv_data:
        path.append(os.path.join(DATA_FOLDER_ROOT, RES, ZOOM, image_type, row['Worked Material'].lower(), row['Image Name']))
        assert os.path.exists(path[-1]), f"ERROR: CAN NOT LOCATED FILE: {path[-1]}"

    return path

def batch_feature_extractor(feature_extractor, preprocess, image_path_list, batch_size, device):

    feature_tensor_all = []
    print(f"\nBatch Feature Extraction")
    for i in tqdm(range(0, len(image_path_list), batch_size)):
        image_tensor_batch = []
        for i in image_path_list[i:i+batch_size]:
            image_tensor_batch.append(read_image_tensor(i))

        image_tensor_batch = torch.stack(image_tensor_batch)
        batch = preprocess(image_tensor_batch.to(device))
        feature = feature_extractor(batch)

        # feature_tensor_all.append(feature)
        feature_tensor_all.append(feature.detach().cpu().numpy())


    # feature_tensor_all = torch.cat(feature_tensor_all, dim=0)
    feature_tensor_all = np.vstack(feature_tensor_all)
    feature_tensor_all = feature_tensor_all.reshape(feature_tensor_all.shape[0], -1)
    

    return feature_tensor_all


def distance_matrix_gpu(feature_tensor, norm_type, device):

    num_vectors = feature_tensor.shape[0]
    distance_matrix = torch.zeros((num_vectors, num_vectors)).to(device)

    if norm_type == "l2":
        distance_matrix = torch.cdist(feature_tensor, feature_tensor, p=2)
    elif norm_type == "l1":
        distance_matrix = torch.cdist(feature_tensor, feature_tensor, p=1)
    elif norm_type == "inf":
        distance_matrix = torch.cdist(feature_tensor, feature_tensor, p=0)

    return distance_matrix

def find_cloest_to_mean(feature_array, image_path_list, csv_path):
    # calculate the mean
    mean_feature = np.mean(feature_tensor_all, axis=0)

    # Calculate the Euclidean distance between each feature tensor and the mean
    distances = np.linalg.norm(feature_array - mean_feature, axis=1)

    # Find the index of the feature tensor that is closest to the mean
    closest_idx = np.argmin(distances)

    filename = image_path_list[closest_idx]

    with open(f'closest_to_mean.txt', 'a') as f:
        content = f"{os.path.basename(csv_path[:-4])}\t{IMAGE_TYPE}\t{MODEL}\t{NORM_TYPE}\t{distances[closest_idx]:4f}\t{filename}"

        f.write(content + '\n')

if __name__ == "__main__":

    # change working dir to script location
    os.chdir(sys.path[0])

    print(f"\n##################################################")
    print(f"ML Toolkits: {os.path.basename(os.getcwd())}")

    parser = argparse.ArgumentParser(description="Feature extraction and distance matrix calculation")

    parser.add_argument("--model", choices=["resnet18", "alexnet", "convnext", "vgg11", "vit", "dinov2"], default="resnet18", help="Choose the model (default: resnet18)")
    parser.add_argument("--image_type", choices=["heightmap", "texture"], default="texture", help="Choose the image type (default: texture)")
    parser.add_argument("--norm", choices=["l2"], default="l2", help="Choose the norm type (default: l2)")
    parser.add_argument("--csv", required=True, help="Path to the CSV file containing image information")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for feature extraction (default: 10)")
    parser.add_argument("--data_root", default="/mnt/SSD_SATA_0/DATASET/LUA_Dataset", help="Root folder of the dataset (LUA_Dataset)")

    args = parser.parse_args()
    MODEL = args.model
    IMAGE_TYPE = args.image_type
    NORM_TYPE = args.norm
    CSV_PATH = args.csv
    BATCH_SIZE = args.batch_size
    DATA_FOLDER_ROOT = args.data_root
    
    # Check for GPU availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.is_available()}")
    else:
        device = torch.device("cpu")
        print(f"Using GPU: {torch.cuda.is_available()}")

    print(f"Model:\t\t{MODEL.upper()}")
    print(f"Image Type:\t{IMAGE_TYPE.upper()}")
    print(f'CSV FILE:\t{CSV_PATH}')
    print(f"Batch Size:\t{BATCH_SIZE}\n")

    

    # read csv file
    CSV_PATH = os.path.join(DATA_FOLDER_ROOT, CSV_PATH)
    image_path_list = csv_to_path(CSV_PATH)
    # image_path_list = [os.path.join(DATA_FOLDER_ROOT, i) for i in image_path_list]
    print(f"Number of Images: {len(image_path_list)}")

    ################## CHANGE THIS PART FOR DIFFERENT MODEL ##################
    # feature extractor initialization
    if MODEL == "resnet18":
        feature_extractor, preprocess = feature_extractor_resnet18()
    elif MODEL == "alexnet":
        feature_extractor, preprocess = feature_extractor_alexnet()
    elif MODEL == "convnext":
        feature_extractor, preprocess = feature_extractor_convnext()
    elif MODEL == "vgg11":
        feature_extractor, preprocess = feature_extractor_vgg11()
    elif MODEL == "vit":
        feature_extractor, preprocess = feature_extractor_vit()
    elif MODEL == "dinov2":
        feature_extractor, preprocess = feature_extractor_dinov2()
    else:
        print("ERROR: MODEL NOT SUPPORTED")

    ##########################################################################

    # perform batch feature extraction
    with torch.no_grad():
        feature_tensor_all = batch_feature_extractor(feature_extractor.to(device), preprocess, image_path_list, BATCH_SIZE, device)

    # find the image that is cloest to the mean
    find_cloest_to_mean(feature_tensor_all, image_path_list, CSV_PATH)
    
    print(f"Feature Tensor: {feature_tensor_all.shape}")
    # feature_tensor_all = feature_tensor_all.reshape(feature_tensor_all.size(0), -1)
    feature_tensor_all = torch.from_numpy(feature_tensor_all).to(device)

    # calculate distance matrix
    distance_matrix = distance_matrix_gpu(feature_tensor_all, NORM_TYPE, device)
    print(f"Distance Matrix: {distance_matrix.shape}")

    # save distance matrix
    filename = f"{os.path.basename(CSV_PATH[:-4])}_{IMAGE_TYPE}_{MODEL}_{NORM_TYPE}.npy"
    print(f"\nSaving distance matrix to {filename}...")
    np.save(f"data/{filename}", distance_matrix.cpu().numpy())

    print("Done\n")




