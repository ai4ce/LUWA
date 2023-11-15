# Code modified from pytorch-image-classification
# obtained from https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/5_resnet.ipynb#scrollTo=4QmwmcXuPuLo

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
from utils.experiment_utils import get_model, get_name, get_logger, train, evaluate, evaluate_vote
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
    if os.path.exists(checkpoint_dir / 'epoch10.pth'):
        print('CHECKPOINT FOUND')
        print('TERMINATING TRAINING')
        return 0 # terminate training if checkpoint exists
    
    logger = get_logger(experiment_dir, model_name)

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
    train_iterator = torch.utils.data.DataLoader(train_dataset, 
                                                 batch_size=BATCHSIZE, 
                                                 num_workers=4, 
                                                 shuffle=True, 
                                                 pin_memory=True,
                                                 drop_last=False)
    
    valid_iterator = torch.utils.data.DataLoader(valid_dataset, 
                                                 batch_size=BATCHSIZE, 
                                                 num_workers=4, 
                                                 shuffle=True, 
                                                 pin_memory=True,
                                                 drop_last=False)
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

    # Define optimizer and scheduler
    START_LR = args.start_lr
    optimizer = optim.Adam(model.parameters(), lr=START_LR)
    STEPS_PER_EPOCH = len(train_iterator)
    print('STEPS_PER_EPOCH:', STEPS_PER_EPOCH)
    print('VALIDATION STEPS:', len(valid_iterator))
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max(STEPS_PER_EPOCH,STEPS_PER_EPOCH//10))

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)

    EPOCHS = args.epochs

    print('SETUP DONE')
    # train our model

    print('TRAINING STARTED')
    for epoch in tqdm.tqdm(range(EPOCHS)):

        train_loss, train_acc_1, train_acc_3 = train(model, train_iterator, optimizer, criterion, scheduler, device)
        
        torch.cuda.empty_cache() # clear cache between train and val

        valid_loss, valid_acc_1, valid_acc_3 = evaluate(model, valid_iterator, criterion, device)

        torch.save(model.state_dict(), checkpoint_dir / f'epoch{epoch+1}.pth')

        logger.info(f'Epoch: {epoch + 1:02}')
        logger.info(f'\tTrain Loss: {train_loss:.3f} | Train Acc @1: {train_acc_1 * 100:6.2f}% | ' \
                f'Train Acc @3: {train_acc_3 * 100:6.2f}%')
        logger.info(f'\tValid Loss: {valid_loss:.3f} | Valid Acc @1: {valid_acc_1 * 100:6.2f}% | ' \
                f'Valid Acc @3: {valid_acc_3 * 100:6.2f}%')

    logger.info('-------------------End of Training-------------------')
    print('TRAINING DONE')
    logger.info('-------------------Beginning of Testing-------------------')
    print('TESTING STARTED')
    for epoch in tqdm.tqdm(range(EPOCHS)):
        model.load_state_dict(torch.load(checkpoint_dir / f'epoch{epoch+1}.pth'))

        if args.vote == 'vote':
            test_acc = evaluate_vote(model, test_iterator, device)
            logger.info(f'Test Acc @1: {test_acc * 100:6.2f}%')
        else:
            test_loss, test_acc_1, test_acc_3 = evaluate(model, test_iterator, criterion, device)

            logger.info(f'Test Acc @1: {test_acc_1 * 100:6.2f}% | ' \
                    f'Test Acc @3: {test_acc_3 * 100:6.2f}%')
    logger.info('-------------------End of Testing-------------------')
    print('TESTING DONE')


if __name__ == '__main__':
    args = get_args()
    main(args)
