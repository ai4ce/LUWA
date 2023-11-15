import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import logging
from collections import Counter
from utils.MAE import mae_vit_large_patch16_dec512d8b as MAE_large 

def get_model(args) -> nn.Module:
    if 'ResNet' in args.model:
        # resnet family
        if args.model == 'ResNet50':
            if args.pretrained == 'pretrained':
                model = torchvision.models.resnet50(weights='IMAGENET1K_V2')
            else:
                model = torchvision.models.resnet50()
        elif args.model == 'ResNet152':
            if args.pretrained == 'pretrained':
                model = torchvision.models.resnet152(weights='IMAGENET1K_V2')
            else:
                model = torchvision.models.resnet152()
        else:
            raise NotImplementedError
        if args.frozen == 'frozen':
            model = freeze_backbone(model)
        model.fc = nn.Linear(model.fc.in_features, 6)
    
    elif 'ConvNext' in args.model:
        if args.model == 'ConvNext_Tiny':
            if args.pretrained == 'pretrained':
                model = torchvision.models.convnext_tiny(weights='IMAGENET1K_V1')
            else:
                model = torchvision.models.convnext_tiny()
        elif args.model == 'ConvNext_Large':
            if args.pretrained == 'pretrained':
                model = torchvision.models.convnext_large(weights='IMAGENET1K_V1')
            else:
                model = torchvision.models.convnext_large()
        else:
            raise NotImplementedError
        if args.frozen == 'frozen':
            model = freeze_backbone(model)
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(int(num_ftrs), 6) 

    elif 'ViT' in args.model:
        if args.pretrained == 'pretrained':
            model = torchvision.models.vit_h_14(weights='IMAGENET1K_SWAG_LINEAR_V1')
        else:
            raise NotImplementedError('ViT does not support training from scratch')
        if args.frozen == 'frozen':
            model = freeze_backbone(model)
        model.heads[0] = torch.nn.Linear(model.heads[0].in_features, 6)

    elif 'DINOv2' in args.model:
        if args.pretrained == 'pretrained':
            model  = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg_lc')
        else:
            raise NotImplementedError('DINOv2 does not support training from scratch')
        if args.frozen == 'frozen':
            model = freeze_backbone(model)
        model.linear_head = torch.nn.Linear(model.linear_head.in_features, 6)
    
    elif 'MAE' in args.model:
        if args.pretrained == 'pretrained':
            model = MAE_large()
            model.load_state_dict(torch.load('/scratch/zf540/LUWA/workspace/utils/pretrained_weights/mae_visualize_vit_large.pth')['model'])
        else:
            raise NotImplementedError('MAE does not support training from scratch')
        if args.frozen == 'frozen':
            model = freeze_backbone(model)
        model = nn.Sequential(model, nn.Linear(1024, 6))
        print(model)
    else:
        raise NotImplementedError
    return model


def freeze_backbone(model):
    # freeze backbone
    # we will replace the classifier at the end with a trainable one anyway, so we freeze the default here as well
    for param in model.parameters():
        param.requires_grad = False
    return model

def get_name(args):
    name = args.model
    name += '_'+str(args.resolution)
    name += '_'+args.magnification
    name += '_'+args.modality
    if args.pretrained == 'pretrained':
        name += '_pretrained'
    else:
        name += '_scratch'
    if args.frozen == 'frozen':
        name += '_frozen'
    else:
        name += '_unfrozen'
    if args.vote == 'vote':
        name += '_vote'
    else:
        name += '_novote'
    return name

def get_logger(path, name):
    # set up logger

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(path.joinpath(f'{name}_log.txt'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------TRANING---------------------------------------------------')
    
    return logger

def calculate_topk_accuracy(y_pred, y, k = 3):
    with torch.no_grad():
        batch_size = y.shape[0]
        _, top_pred = y_pred.topk(k, 1)
        top_pred = top_pred.t()
        correct = top_pred.eq(y.view(1, -1).expand_as(top_pred))
        correct_1 = correct[:1].reshape(-1).float().sum(0, keepdim = True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim = True)
        acc_1 = correct_1 / batch_size
        acc_k = correct_k / batch_size
    return acc_1, acc_k

def train(model, iterator, optimizer, criterion, scheduler, device):
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_3 = 0

    model.train()

    for image, label, image_name in iterator:
        x = image.to(device)
        y = label.to(device)

        optimizer.zero_grad()

        y_pred = model(x)
        print(y_pred.shape)
        print(y.shape)
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
        for image, label, image_name in iterator:
            x = image.to(device)
            y = label.to(device)

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

def evaluate_vote(model, iterator, device):

    model.eval()

    image_names = []
    labels = []
    predictions = []

    with torch.no_grad():

        for image, label, image_name in iterator:

            x = image.to(device)

            y_pred = model(x)
            y_prob = F.softmax(y_pred, dim = -1)
            top_pred = y_prob.argmax(1, keepdim = True)

            image_names.extend(image_name)
            labels.extend(label.numpy())
            predictions.extend(top_pred.cpu().squeeze().numpy())

    conduct_voting(image_names, predictions)

    correct_count = 0
    for i in range(len(labels)):
        if labels[i] == predictions[i]:
            correct_count += 1
    accuracy = correct_count/len(labels)
    return accuracy

def conduct_voting(image_names, predictions):
    # we need to do this because not all stones have the same number of partition
    last_stone = image_names[0][:-8] # the name of the stone of the last image
    voting_list = []
    for i in range(len(image_names)):
        image_area_name = image_names[i][:-8]
        if image_area_name != last_stone:
            # we have run through all the images of the last stone. We start voting
            vote(voting_list, predictions, i)
            voting_list = [] # reset the voting list
        voting_list.append(predictions[i])
        last_stone = image_area_name # update the last stone name
    
    # vote for the last stone
    vote(voting_list, predictions, len(image_names))

def vote(voting_list, predictions, i):
    vote_result = Counter(voting_list).most_common(1)[0][0] # the most common prediction in the list
    predictions[i-len(voting_list):i] = [vote_result]*len(voting_list) # replace the predictions of the last stone with the vote result
        
        


# def get_predictions(model, iterator):

#     model.eval()

#     images = []
#     labels = []
#     probs = []

#     with torch.no_grad():

#         for (x, y) in iterator:

#             x = x.to(device)

#             y_pred = model(x)

#             y_prob = F.softmax(y_pred, dim = -1)
#             top_pred = y_prob.argmax(1, keepdim = True)

#             images.append(x.cpu())
#             labels.append(y.cpu())
#             probs.append(y_prob.cpu())

#     images = torch.cat(images, dim = 0)
#     labels = torch.cat(labels, dim = 0)
#     probs = torch.cat(probs, dim = 0)

#     return images, labels, probs


# def get_representations(model, iterator):
#     model.eval()

#     outputs = []
#     intermediates = []
#     labels = []

#     with torch.no_grad():
#         for (x, y) in iterator:
#             x = x.to(device)

#             y_pred = model(x)

#             outputs.append(y_pred.cpu())
#             labels.append(y)

#     outputs = torch.cat(outputs, dim=0)
#     labels = torch.cat(labels, dim=0)

#     return outputs, labels
