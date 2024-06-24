import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import torch.utils.data as data

import numpy as np
import random
import tqdm
import os
from pathlib import Path

from data_utils.data_tribology import TribologyDataset
from utils.experiment_utils import get_model, get_name, get_logger, train, evaluate, evaluate_vote, evaluate_vote_analysis
from utils.arg_utils import get_args

def main(args):
    '''Reproducibility'''
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    '''Folder Creation'''
    basepath=os.getcwd()
    experiment_dir = Path(os.path.join(basepath,'experiments',args.model,args.resolution,args.magnification,args.modality,args.pretrained,args.frozen,args.vote))
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(os.path.join(experiment_dir,'checkpoints'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    '''Logging'''
    model_name = get_name(args)
    print(model_name, 'STARTED')
    
    logger = get_logger(experiment_dir, 'vote_analysis')

    '''Data Loading'''
    train_csv_path = f"./LUA_Dataset/CSV/{args.resolution}_{args.magnification}_6w_train.csv"
    test_csv_path = f"./LUA_Dataset/CSV/{args.resolution}_{args.magnification}_6w_test.csv"
    img_path = f"./LUA_Dataset/{args.resolution}/{args.magnification}/{args.modality}"

    # results_acc_1 = {}
    # results_acc_3 = {}
    # classes_num = 6
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
    train_dataset, valid_dataset = data.random_split(train_dataset, [num_train - num_valid, num_valid])
    logger.info(f'Number of training samples: {len(train_dataset)}')
    logger.info(f'Number of validation samples: {len(valid_dataset)}')

    test_iterator = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=BATCHSIZE, 
                                                num_workers=4, 
                                                shuffle=False, 
                                                pin_memory=True,
                                                drop_last=False)
    print('DATA LOADED')

    # Define model 
    model = get_model(args)
    print('MODEL LOADED')

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)


    print('SETUP DONE')
    # train our model

    print('TRAINING STARTED')

    model.load_state_dict(torch.load(checkpoint_dir / f'epoch{args.epochs}.pth'))
    logger.info('-------------------Beginning of Testing-------------------')
    print('TESTING STARTED')

    vote_accuracy, correct_case_accuracy, incorrect_case_accuracy, incorrect_most_common, novote_accuracy = evaluate_vote_analysis(model, test_iterator, device)
    logger.info(f'Test Acc @1: {vote_accuracy * 100:6.2f}%')
    logger.info(f'No Vote Accuracy @1: {novote_accuracy * 100:6.2f}%')
    logger.info(f'Correct Case Consistency @1: {correct_case_accuracy * 100:6.2f}%')
    logger.info(f'Incorrect Case Consistency @1: {incorrect_case_accuracy * 100:6.2f}%')
    logger.info(f'Incorrect Most Common: {incorrect_most_common* 100:6.2f}%')

    logger.info('-------------------End of Testing-------------------')
    print('TESTING DONE')


if __name__ == '__main__':
    args = get_args()
    main(args)
