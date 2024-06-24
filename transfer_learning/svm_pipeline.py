import numpy as np
from sklearn.svm import LinearSVC

from skimage.feature import fisher_vector, learn_gmm

import numpy as np
import random
import os
from pathlib import Path

from data_utils.data_tribology import TribologyDataset
from utils.arg_utils import get_args
from utils.experiment_utils import get_name, get_logger, SIFT_extraction, conduct_voting
from utils.visualization_utils import plot_confusion_matrix
from vis_confusion_mtx import generate_confusion_matrix

def main(args):
    '''Reproducibility'''
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)

    '''Folder Creation'''
    basepath=os.getcwd()
    experiment_dir = Path(os.path.join(basepath,'experiments',args.model,args.resolution,args.magnification,args.modality,args.vote))
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(os.path.join(experiment_dir,'checkpoints'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    '''Logging'''
    model_name = get_name(args)
    print(model_name, 'STARTED', flush=True)
    logger = get_logger(experiment_dir, model_name)

    '''Data Loading'''
    train_csv_path = f"./LUA_Dataset/CSV/{args.resolution}_{args.magnification}_6w_train.csv"
    test_csv_path = f"./LUA_Dataset/CSV/{args.resolution}_{args.magnification}_6w_test.csv"
    img_path = f"./LUA_Dataset/{args.resolution}/{args.magnification}/{args.modality}"

    BATCHSIZE = args.batch_size
    train_dataset = TribologyDataset(csv_path = train_csv_path, img_path = img_path)
    test_dataset = TribologyDataset(csv_path = test_csv_path, img_path = img_path)

    # prepare the data augmentation
    means, stds = train_dataset.get_statistics()
    train_dataset.prepare_transform(means, stds, mode='train')
    test_dataset.prepare_transform(means, stds, mode='test')

    VALID_RATIO = 0.1

    num_train = len(train_dataset)
    num_valid = int(VALID_RATIO * num_train)
    # train_dataset, valid_dataset = data.random_split(train_dataset, [num_train - num_valid, num_valid])
    # logger.info(f'Number of training samples: {len(train_dataset)}')
    # logger.info(f'Number of validation samples: {len(valid_dataset)}')

    train_names, train_descriptor, train_labels = SIFT_extraction(train_dataset)
    test_names, test_descriptor, test_labels = SIFT_extraction(test_dataset)
    # val_descriptor, val_labels = SIFT_extraction(valid_dataset)
    print('DATA LOADED', flush=True)

    print('TRAINING STARTED', flush=True)

    # Train a K-mode GMM
    k = 16
    gmm = learn_gmm(train_descriptor, n_modes=k)

    # Compute the Fisher vectors
    training_fvs = np.array([
        fisher_vector(descriptor_mat, gmm)
        for descriptor_mat in train_descriptor
    ])

    testing_fvs = np.array([
        fisher_vector(descriptor_mat, gmm)
        for descriptor_mat in test_descriptor
    ])
    
    svm = LinearSVC().fit(training_fvs, train_labels)

    logger.info('-------------------End of Training-------------------')
    print('TRAINING DONE')
    logger.info('-------------------Beginning of Testing-------------------')
    print('TESTING STARTED')
    predictions = svm.predict(testing_fvs)
    conduct_voting(test_names, predictions)
    plot_confusion_matrix('visualization_results/SIFT+FVs_confusion_mtx.png', predictions, test_labels,classes=["ANTLER", "BEECHWOOD", "BEFOREUSE", "BONE", "IVORY","SPRUCEWOOD"])
    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == test_labels[i]:
            correct += 1
    test_acc = float(correct)/len(predictions)
    logger.info(f'Test Acc @1: {test_acc * 100:6.2f}%')

    logger.info('-------------------End of Testing-------------------')
    print('TESTING DONE')

if __name__ == '__main__':
    args = get_args()
    main(args)
