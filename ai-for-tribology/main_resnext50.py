# Code modified from pytorch-image-classification
# obtained from https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/5_resnet.ipynb#scrollTo=4QmwmcXuPuLo

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torch.optim.lr_scheduler as lr_scheduler

import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.models import resnext50_32x4d


from sklearn import manifold


import numpy as np

import copy
from collections import namedtuple
import os
import random
import shutil
import time
from sklearn.model_selection import KFold
from datasets import TribologyDataset
from models import Bottleneck, ResNet
from utils import calculate_topk_accuracy, epoch_time, plot_lr_finder, plot_confusion_matrix, plot_most_incorrect, get_pca, plot_representations, plot_filters, plot_filtered_images

class LRFinder:
    def __init__(self, model, optimizer, criterion, device):

        self.optimizer = optimizer
        self.model = model
        self.criterion = criterion
        self.device = device

        torch.save(model.state_dict(), 'init_params.pt')

    def range_test(self, iterator, end_lr=10, num_iter=100,
                   smooth_f=0.05, diverge_th=5):

        lrs = []
        losses = []
        best_loss = float('inf')

        lr_scheduler = ExponentialLR(self.optimizer, end_lr, num_iter)

        iterator = IteratorWrapper(iterator)

        for iteration in range(num_iter):

            loss = self._train_batch(iterator)

            # update lr
            lr_scheduler.step()

            lrs.append(lr_scheduler.get_lr()[0])

            if iteration > 0:
                loss = smooth_f * loss + (1 - smooth_f) * losses[-1]

            if loss < best_loss:
                best_loss = loss

            losses.append(loss)

            if loss > diverge_th * best_loss:
                print("Stopping early, the loss has diverged")
                break

        # reset model to initial parameters
        model.load_state_dict(torch.load('init_params.pt'))

        return lrs, losses

    def _train_batch(self, iterator):

        self.model.train()

        self.optimizer.zero_grad()

        x, y = iterator.get_batch()

        x = x.to(self.device)
        y = y.to(self.device)

        y_pred = self.model(x)

        # print(y.shape)
        # print(y_pred.shape)

        loss = self.criterion(y_pred, y)

        loss.backward()

        self.optimizer.step()

        return loss.item()


class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


class IteratorWrapper:
    def __init__(self, iterator):
        self.iterator = iterator
        self._iterator = iter(iterator)

    def __next__(self):
        try:
            inputs, labels = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.iterator)
            inputs, labels, *_ = next(self._iterator)

        return inputs, labels

    def get_batch(self):
        return next(self)


def train(model, iterator, optimizer, criterion, scheduler, device):
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_3 = 0

    model.train()

    for (x, y) in iterator:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred = model(x)

        loss = criterion(y_pred, y)

        acc_1, acc_3 = calculate_topk_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

        scheduler.step()

        epoch_loss += loss.item()
        epoch_acc_1 += acc_1.item()
        epoch_acc_3 += acc_3.item()

    epoch_loss /= len(iterator)
    epoch_acc_1 /= len(iterator)
    epoch_acc_3 /= len(iterator)

    return epoch_loss, epoch_acc_1, epoch_acc_3


def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_3 = 0

    model.eval()

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            loss = criterion(y_pred, y)

            acc_1, acc_3 = calculate_topk_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc_1 += acc_1.item()
            epoch_acc_3 += acc_3.item()

    epoch_loss /= len(iterator)
    epoch_acc_1 /= len(iterator)
    epoch_acc_3 /= len(iterator)

    return epoch_loss, epoch_acc_1, epoch_acc_3

def get_predictions(model, iterator):

    model.eval()

    images = []
    labels = []
    probs = []

    with torch.no_grad():

        for (x, y) in iterator:

            x = x.to(device)

            y_pred = model(x)

            y_prob = F.softmax(y_pred, dim = -1)
            top_pred = y_prob.argmax(1, keepdim = True)

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim = 0)
    labels = torch.cat(labels, dim = 0)
    probs = torch.cat(probs, dim = 0)

    return images, labels, probs


def get_representations(model, iterator):
    model.eval()

    outputs = []
    intermediates = []
    labels = []

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)

            y_pred = model(x)

            outputs.append(y_pred.cpu())
            labels.append(y)

    outputs = torch.cat(outputs, dim=0)
    labels = torch.cat(labels, dim=0)

    return outputs, labels

if __name__ == '__main__':
    # set the random seeds for reproducability
    SEED = 1234

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # train/test split
    input_path = '/scratch/hw3245/Ai-for-Tribology/cropped_images/Dataset'
    results_acc_1 = {}
    results_acc_3 = {}
    classes_num = 6
    BATCHSIZE = 100
    train_dataset = TribologyDataset(input_path, split='train')
    test_dataset = TribologyDataset(input_path, split='test')

    classes = ['0.ANTLER',
     '1.BEECHWOOD',
     '2.BONE',
     '3.IVORY',
     '4.SPRUCEWOOD',
     '5.BU']

    # Start print
    print('--------------------------------')

    VALID_RATIO = 0.1

    num_train = len(train_dataset)
    num_valid = int(VALID_RATIO * num_train)
    train_dataset, valid_dataset = data.random_split(train_dataset, [num_train - num_valid, num_valid])
    print(f'Number of training samples: {len(train_dataset)}')
    print(f'Number of validation samples: {len(valid_dataset)}')
    print(f'Number of test samples: {len(test_dataset)}')

    # Define data loaders for training and testing data in this fold
    train_iterator = torch.utils.data.DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)
    valid_iterator = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCHSIZE, shuffle=True)
    test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False)


    # Define model 
    model = resnext50_32x4d(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 6)

    # Training the Model

    START_LR = 1e-7

    optimizer = optim.Adam(model.parameters(), lr=START_LR)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    END_LR = 10
    NUM_ITER = 100

    lr_finder = LRFinder(model, optimizer, criterion, device)
    lrs, losses = lr_finder.range_test(train_iterator, END_LR, NUM_ITER)

    lr_fig_path = f'./lr-finder-ResNext50.pdf'
    plot_lr_finder(lr_fig_path, lrs, losses, skip_start=30, skip_end=30)

    FOUND_LR = 1e-3

    params = [
        {'params': model.conv1.parameters(), 'lr': FOUND_LR / 10},
        {'params': model.bn1.parameters(), 'lr': FOUND_LR / 10},
        {'params': model.layer1.parameters(), 'lr': FOUND_LR / 8},
        {'params': model.layer2.parameters(), 'lr': FOUND_LR / 6},
        {'params': model.layer3.parameters(), 'lr': FOUND_LR / 4},
        {'params': model.layer4.parameters(), 'lr': FOUND_LR / 2},
        {'params': model.fc.parameters()}
    ]

    optimizer = optim.Adam(params, lr=FOUND_LR)

    EPOCHS = 10
    STEPS_PER_EPOCH = len(train_iterator)
    TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH

    MAX_LRS = [p['lr'] for p in optimizer.param_groups]

    scheduler = lr_scheduler.OneCycleLR(optimizer,
                                        max_lr=MAX_LRS,
                                        total_steps=TOTAL_STEPS)

    # train our model
    best_valid_loss = float('inf')

    for epoch in range(EPOCHS):

        start_time = time.monotonic()

        train_loss, train_acc_1, train_acc_3 = train(model, train_iterator, optimizer, criterion, scheduler, device)
        valid_loss, valid_acc_1, valid_acc_3 = evaluate(model, valid_iterator, criterion, device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # Saving the model
            save_path = f'./model-fold-ResNext50.pth'
            torch.save(model.state_dict(), save_path)
            # torch.save(model.state_dict(), 'tut5-model.pt')

        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc @1: {train_acc_1 * 100:6.2f}% | ' \
                f'Train Acc @3: {train_acc_3 * 100:6.2f}%')
        print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc @1: {valid_acc_1 * 100:6.2f}% | ' \
                f'Valid Acc @3: {valid_acc_3 * 100:6.2f}%')

    # Print about testing
    print('Starting testing')

    model.load_state_dict(torch.load(save_path))

    test_loss, test_acc_1, test_acc_3 = evaluate(model, test_iterator, criterion, device)

    print(f'Test Loss: {test_loss:.3f} | Test Acc @1: {test_acc_1 * 100:6.2f}% | ' \
            f'Test Acc @3: {test_acc_3 * 100:6.2f}%')

    # Examining the Model

    images, labels, probs = get_predictions(model, test_iterator)
    pred_labels = torch.argmax(probs, 1)

    cm_fig_path = f'./confusion-matrix-ResNext50.pdf'
    plot_confusion_matrix(cm_fig_path, labels, pred_labels, classes)

    corrects = torch.eq(labels, pred_labels)

    incorrect_examples = []

    for image, label, prob, correct in zip(images, labels, probs, corrects):
        if not correct:
            incorrect_examples.append((image, label, prob))

    incorrect_examples.sort(reverse=True, key=lambda x: torch.max(x[2], dim=0).values)


    outputs, labels = get_representations(model, train_iterator)

    output_pca_data = get_pca(outputs)
    pca_path = f'./output-pca-ResNext50.pdf'
    plot_representations(pca_path, output_pca_data, labels, classes)

    N_IMAGES = 5
    N_FILTERS = 7

    images = [image for image, label in [train_dataset[i] for i in range(N_IMAGES)]]
    filters = model.conv1.weight.data
    filtered_image_path = f'./output-filtered-image-ResNext50.pdf'
    plot_filtered_images(filtered_image_path, images, filters, N_FILTERS)

    filters_path = f'./output-filters-ResNext50.pdf'
    plot_filters(filters_path, filters)


