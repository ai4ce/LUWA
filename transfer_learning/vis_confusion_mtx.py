import torch

import os
from pathlib import Path

from data_utils.data_tribology import TribologyDataset
from utils.experiment_utils import get_model, get_prediction
from utils.arg_utils import get_args
from utils.visualization_utils import plot_confusion_matrix

def generate_confusion_matrix(image_name, model, iterator, device):
    labels, predictions = get_prediction(model, iterator, device)
    plot_confusion_matrix('visualization_results/'+image_name+'_confusion_mtx.png', labels, predictions, classes=["ANTLER", "BEECHWOOD", "BEFOREUSE", "BONE", "IVORY","SPRUCEWOOD"])


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = get_model(args)

    basepath=os.getcwd()
    experiment_dir = Path(os.path.join(basepath,'experiments',args.model,args.resolution,args.magnification,args.modality,args.pretrained,args.frozen,args.vote))
    if args.model == 'ViT':
        experiment_dir = Path(os.path.join(basepath,'experiments','ViT_H',args.resolution,args.magnification,args.modality,args.pretrained,args.frozen,args.vote))
    checkpoint_dir = Path(os.path.join(experiment_dir,'checkpoints'))
    checkpoint_path = checkpoint_dir / f'epoch{str(args.epochs)}.pth'
    model.load_state_dict(torch.load(checkpoint_path))
    model = model.to(device)

    train_csv_path = f"./LUA_Dataset/CSV/{args.resolution}_{args.magnification}_6w_train.csv"
    test_csv_path = f"./LUA_Dataset/CSV/{args.resolution}_{args.magnification}_6w_test.csv"
    img_path = f"./LUA_Dataset/{args.resolution}/{args.magnification}/{args.modality}"
    BATCHSIZE = args.batch_size
    train_dataset = TribologyDataset(csv_path = train_csv_path, img_path = img_path)
    test_dataset = TribologyDataset(csv_path = test_csv_path, img_path = img_path)

    means, stds = train_dataset.get_statistics()
    train_dataset.prepare_transform(means, stds, mode='train')
    test_dataset.prepare_transform(means, stds, mode='test')

    test_iterator = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=BATCHSIZE, 
                                            num_workers=4, 
                                            shuffle=False, 
                                            pin_memory=True,
                                            drop_last=False)


    generate_confusion_matrix(args.model, model, test_iterator, device)

if __name__ == "__main__":
    args = get_args()
    main(args)

