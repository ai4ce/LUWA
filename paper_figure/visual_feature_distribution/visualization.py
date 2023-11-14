from collections import namedtuple
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

MODEL = ['resnet18', 'vgg11', 'convnext', 'dinov2']
ZOOM = ['20x', '50x']
IMAGE_TYPE = ['heightmap', 'texture']
NORM_TYPE = ['l2']
RESOLUTION = ['256', '512', '865']

def normalize(input_array):
    min_val = np.min(input_array)
    max_val = np.max(input_array)
    return (input_array - min_val) / (max_val - min_val)

def load_npy_data(file_path):
    print(f"Loading {file_path}...")
    distance_matrix = np.load(file_path)
    # Select only the upper triangular part of the matrix, excluding the diagonal
    distance_matrix = np.triu(distance_matrix, k=1)
    distances = distance_matrix.flatten()[distance_matrix.flatten() != 0]

    return normalize(distances)

def batch_load_npy_data(file_path_list):
    distance_matrix_list = []
    for file_path in file_path_list:
        distance_matrix_list.append(load_npy_data(file_path))
    return distance_matrix_list


def load_distance(resolution="256"):
    distance_data = []

    for model in MODEL:
        for zoom in ZOOM:
            for image_type in IMAGE_TYPE:
                for norm_type in NORM_TYPE:
                    filename = f"data/{resolution}_{zoom}_{image_type}_{model}_{norm_type}.npy"
                    distance = load_npy_data(filename)
                    selected_distance = np.random.choice(distance, 1000, replace=False)
                    distance_data.append(EXP(resolution, zoom, image_type, model, norm_type, selected_distance))

    return distance_data



if __name__ == "__main__":

    os.chdir(sys.path[0])

    EXP = namedtuple("EXP", ["resolution", "zoom", "image_type", "model", "norm_type", "distance"])

    RESOLUTION = "865"

    if not os.path.exists(f"data/distance_{RESOLUTION}.pkl"):
        distance_data = load_distance(resolution=RESOLUTION)
        with open(f'data/distance_{RESOLUTION}.pkl', 'wb') as file:
            pickle.dump(distance_data, file)
    else:
        with open(f'data/distance_{RESOLUTION}.pkl', 'rb') as file:
            distance_data = pickle.load(file)



    # print(distance_data[0].resolution, distance_data[0].zoom, distance_data[0].image_type, distance_data[0].model, distance_data[0].norm_type, distance_data[0].distance)

    data_a = []; data_b = []; data_c = []; data_d = []
    for dist in distance_data:

        # DATA GROUP B
        if dist.model == "vgg11" and dist.zoom == "20x" and dist.image_type == "texture":
            data_a.append(dist.distance)
        if dist.model == "resnet18" and dist.zoom == "20x" and dist.image_type == "texture":
            data_a.append(dist.distance)
        if dist.model == "convnext" and dist.zoom == "20x" and dist.image_type == "texture":
            data_a.append(dist.distance)
        if dist.model == "dinov2" and dist.zoom == "20x" and dist.image_type == "texture":
            data_a.append(dist.distance)

        # DATA GROUP A
        if dist.model == "vgg11" and dist.zoom == "20x" and dist.image_type == "heightmap":
            data_b.append(dist.distance)
        if dist.model == "resnet18" and dist.zoom == "20x" and dist.image_type == "heightmap":
            data_b.append(dist.distance)
        if dist.model == "convnext" and dist.zoom == "20x" and dist.image_type == "heightmap":
            data_b.append(dist.distance)
        if dist.model == "dinov2" and dist.zoom == "20x" and dist.image_type == "heightmap":
            data_b.append(dist.distance)

         # DATA GROUP B
        if dist.model == "vgg11" and dist.zoom == "50x" and dist.image_type == "texture":
            data_c.append(dist.distance)
        if dist.model == "resnet18" and dist.zoom == "50x" and dist.image_type == "texture":
            data_c.append(dist.distance)
        if dist.model == "convnext" and dist.zoom == "50x" and dist.image_type == "texture":
            data_c.append(dist.distance)
        if dist.model == "dinov2" and dist.zoom == "50x" and dist.image_type == "texture":
            data_c.append(dist.distance)

         # DATA GROUP C
        if dist.model == "vgg11" and dist.zoom == "50x" and dist.image_type == "heightmap":
            data_d.append(dist.distance)
        if dist.model == "resnet18" and dist.zoom == "50x" and dist.image_type == "heightmap":
            data_d.append(dist.distance)
        if dist.model == "convnext" and dist.zoom == "50x" and dist.image_type == "heightmap":
            data_d.append(dist.distance)
        if dist.model == "dinov2" and dist.zoom == "50x" and dist.image_type == "heightmap":
            data_d.append(dist.distance)

        




    data_combined = [data_a, data_b, data_c, data_d]
    positions = [1, 2, 3, 4]

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(20, 8))

    # Boxplot
    sns.set_palette("colorblind")
    # print(len(data_combined[0]), len(data_combined[1]), len(data_combined[2]))
    # bplot1 = ax.boxplot(data_combined[0], positions=[pos - 0.24 for pos in positions],
    #                     widths=0.1, patch_artist=True, boxprops=dict(linewidth=2))
    # bplot2 = ax.boxplot(data_combined[1], positions=[pos - 0.08 for pos in positions],
    #                     widths=0.1, patch_artist=True, boxprops=dict(linewidth=2))
    # bplot3 = ax.boxplot(data_combined[2], positions=[pos + 0.08 for pos in positions],
    #                     widths=0.1, patch_artist=True, boxprops=dict(linewidth=2))
    # bplot4 = ax.boxplot(data_combined[3], positions=[pos + 0.24 for pos in positions],
    #                     widths=0.1, patch_artist=True, boxprops=dict(linewidth=2))
    
    bplot1 = ax.boxplot(data_combined[0], positions=[pos - 0.3 for pos in positions],
                        widths=0.1, patch_artist=True, boxprops=dict(linewidth=2))
    bplot2 = ax.boxplot(data_combined[1], positions=[pos - 0.1 for pos in positions],
                        widths=0.1, patch_artist=True, boxprops=dict(linewidth=2))
    bplot3 = ax.boxplot(data_combined[2], positions=[pos + 0.1 for pos in positions],
                        widths=0.1, patch_artist=True, boxprops=dict(linewidth=2))
    bplot4 = ax.boxplot(data_combined[3], positions=[pos + 0.3 for pos in positions],
                        widths=0.1, patch_artist=True, boxprops=dict(linewidth=2))
    
    # Set Seaborn color palette for box face color
    sns.set_palette("colorblind")
    palette = sns.color_palette()

    alpha = 1.0
    for bplot, color in zip([bplot1, bplot2, bplot3, bplot4], palette):
        for patch in bplot['boxes']:
            patch.set_facecolor('none')
            patch.set_edgecolor(color)
            

    # Set mean line color to black
    for bplot in [bplot1, bplot2, bplot3, bplot4]:
        for line in bplot['medians']:
            line.set_color('black')
            line.set_linewidth(2)

    # for whisker in bplot['whiskers']:
    #     whisker.set(linewidth=5)

    # Set font to Times New Roman for all labels
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = 'Times New Roman'


    # Set the axes labels and title
    # ax.set_xlabel('Model', fontsize=28)
    ax.set_ylabel('Feature Distance', fontsize=24)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_title(f'Resolution: {RESOLUTION}', fontsize=28)
    ax.set_xticks(positions)
    ax.set_xticklabels(['VGG11', 'ResNet18', 'ConvNext', 'DINOv2'], fontsize=28)

    # Create a Seaborn color palette and legend
    sns.set_palette("colorblind")
    labels = ['20x Texture', '20x Heightmap', '50x Texture', '50x Heightmap']
    handles = [plt.Rectangle((0, 0), 1, 1, color=sns.color_palette()[i]) for i in range(len(labels))]
    ax.legend(handles, labels, loc='upper left', fontsize=22, bbox_to_anchor=(0.03, -0.2), ncol=4)
    # Adjust the bottom spacing
    plt.subplots_adjust(bottom=0.3)  # Increase the whitespace below the graph

    # Create legend
    #colors = ['green', 'red', 'purple', 'pink']
    # colors = [(0, 0, 128), (128, 0, 0), (0, 128, 0), (128, 0, 128)]
    # colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]

    # labels = ['20x heightmap', '20x texture', '50x heigthmap', '50x texture']
    # handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
    # ax.legend(handles, labels, loc='upper right')

    plt.savefig(f"visual/feature_distribution_{RESOLUTION}.png", bbox_inches='tight', dpi=300)

    # Show the plot
    plt.show()


