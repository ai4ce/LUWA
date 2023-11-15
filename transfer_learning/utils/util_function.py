import cv2
from sklearn.manifold import TSNE
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import decomposition
import itertools

def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min = image_min, max = image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image


def plot_lr_finder(fig_name, lrs, losses, skip_start=5, skip_end=5):
    if skip_end == 0:
        lrs = lrs[skip_start:]
        losses = losses[skip_start:]
    else:
        lrs = lrs[skip_start:-skip_end]
        losses = losses[skip_start:-skip_end]

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(lrs, losses)
    ax.set_xscale('log')
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('Loss')
    ax.grid(True, 'both', 'x')
    plt.show()
    plt.savefig(fig_name)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def plot_confusion_matrix(fig_name, labels, pred_labels, classes):
    fig = plt.figure(figsize=(50, 50));
    ax = fig.add_subplot(1, 1, 1);
    cm = confusion_matrix(labels, pred_labels);
    cm = ConfusionMatrixDisplay(cm, display_labels=classes);
    cm.plot(values_format='d', cmap='Blues', ax=ax)
    fig.delaxes(fig.axes[1])  # delete colorbar
    plt.xticks(rotation=90, fontsize=50)
    plt.yticks(fontsize=50)
    plt.rcParams.update({'font.size': 50})
    plt.xlabel('Predicted Label', fontsize=50)
    plt.ylabel('True Label', fontsize=50)
    plt.savefig(fig_name)

def plot_confusion_matrix_SVM(fig_name, true_labels, predicted_labels, classes):
    fig = plt.figure(figsize=(100, 100))
    ax = fig.add_subplot(1, 1, 1)
    
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=classes)
    
    cm_display.plot(values_format='d', cmap='Blues', ax=ax)
    
    fig.delaxes(fig.axes[1])  # delete colorbar
    plt.xticks(rotation=90, fontsize=50)
    plt.yticks(fontsize=50)
    plt.rcParams.update({'font.size': 50})
    plt.xlabel('Predicted Label', fontsize=50)
    plt.ylabel('True Label', fontsize=50)
    plt.savefig(fig_name)


def plot_most_incorrect(fig_name, incorrect, classes, n_images, normalize=True):
    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize=(25, 20))

    for i in range(rows * cols):

        ax = fig.add_subplot(rows, cols, i + 1)

        image, true_label, probs = incorrect[i]
        image = image.permute(1, 2, 0)
        true_prob = probs[true_label]
        incorrect_prob, incorrect_label = torch.max(probs, dim=0)
        true_class = classes[true_label]
        incorrect_class = classes[incorrect_label]

        if normalize:
            image = normalize_image(image)

        ax.imshow(image.cpu().numpy())
        ax.set_title(f'true label: {true_class} ({true_prob:.3f})\n' \
                     f'pred label: {incorrect_class} ({incorrect_prob:.3f})')
        ax.axis('off')

    fig.subplots_adjust(hspace=0.4)
    plt.savefig(fig_name)

def get_pca(data, n_components = 2):
    pca = decomposition.PCA()
    pca.n_components = n_components
    pca_data = pca.fit_transform(data)
    return pca_data


def plot_representations(fig_name, data, labels, classes, n_images=None):
    if n_images is not None:
        data = data[:n_images]
        labels = labels[:n_images]

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='hsv')
    # handles, _ = scatter.legend_elements(num = None)
    # legend = plt.legend(handles = handles, labels = classes)
    plt.savefig(fig_name)


def plot_filtered_images(fig_name, images, filters, n_filters = None, normalize = True):

    images = torch.cat([i.unsqueeze(0) for i in images], dim = 0).cpu()
    filters = filters.cpu()

    if n_filters is not None:
        filters = filters[:n_filters]

    n_images = images.shape[0]
    n_filters = filters.shape[0]

    filtered_images = F.conv2d(images, filters)

    fig = plt.figure(figsize = (30, 30))

    for i in range(n_images):

        image = images[i]

        if normalize:
            image = normalize_image(image)

        ax = fig.add_subplot(n_images, n_filters+1, i+1+(i*n_filters))
        ax.imshow(image.permute(1,2,0).numpy())
        ax.set_title('Original')
        ax.axis('off')

        for j in range(n_filters):
            image = filtered_images[i][j]

            if normalize:
                image = normalize_image(image)

            ax = fig.add_subplot(n_images, n_filters+1, i+1+(i*n_filters)+j+1)
            ax.imshow(image.numpy(), cmap = 'bone')
            ax.set_title(f'Filter {j+1}')
            ax.axis('off');

    fig.subplots_adjust(hspace = -0.7)
    plt.savefig(fig_name)


def plot_filters(fig_name, filters, normalize=True):
    filters = filters.cpu()

    n_filters = filters.shape[0]

    rows = int(np.sqrt(n_filters))
    cols = int(np.sqrt(n_filters))

    fig = plt.figure(figsize=(30, 15))

    for i in range(rows * cols):

        image = filters[i]

        if normalize:
            image = normalize_image(image)

        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(image.permute(1, 2, 0))
        ax.axis('off')

    fig.subplots_adjust(wspace=-0.9)
    plt.savefig(fig_name)

def plot_tsne(fig_name, all_features, all_labels):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(all_features)
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=all_labels, cmap='viridis', s=5)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization')
    plt.show()
    plt.savefig(fig_name)


def plot_grad_cam(images, cams, predicted_labels, true_labels, classes, path):
    fig, axs = plt.subplots(nrows=2, ncols=len(images), figsize=(20, 10))
    
    for i, (img, cam, pred_label, true_label) in enumerate(zip(images, cams, predicted_labels, true_labels)):
        # Display the original image on the top row
        axs[0, i].imshow(img.permute(1,2,0).cpu().numpy())
        pred_class_name = classes[pred_label]
        true_class_name = classes[true_label]
        axs[0, i].set_title(f"Predicted: {pred_class_name}\nTrue: {true_class_name}", fontsize=12)
        axs[0, i].axis('off')

        # Add label to the leftmost plot
        if i == 0:
            axs[0, i].set_ylabel("Original Image", fontsize=14, rotation=90, labelpad=10)
        
        # Convert the original image to grayscale
        grayscale_img = cv2.cvtColor(img.permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2GRAY)
        grayscale_img = cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2RGB)

        # Overlay the Grad-CAM heatmap on the grayscale image
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam_img = heatmap + np.float32(grayscale_img)
        cam_img = cam_img / np.max(cam_img)

        # Display the Grad-CAM image on the bottom row
        axs[1, i].imshow(cam_img)
        axs[1, i].axis('off')

        # Add label to the leftmost plot
        if i == 0:
            axs[1, i].set_ylabel("Grad-CAM", fontsize=14, rotation=90, labelpad=10)
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

