import os, sys
import torch
from torchinfo import summary
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import alexnet, AlexNet_Weights
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models import vgg11, VGG11_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights

def feature_extractor_resnet18():

    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()

    model = resnet18(weights=weights)
    model.eval()

    feature_extractor = torch.nn.Sequential(*list(model.children())[:-2])

    return feature_extractor, preprocess

def feature_extractor_alexnet():

    weights = AlexNet_Weights.DEFAULT
    preprocess = weights.transforms()

    model = alexnet(weights=weights)
    model.eval()

    feature_extractor = torch.nn.Sequential(*list(model.children())[:-2])

    return feature_extractor, preprocess

def feature_extractor_convnext():

    weights = ConvNeXt_Tiny_Weights.DEFAULT
    preprocess = weights.transforms()

    model = convnext_tiny(weights=weights)
    model.eval()

    feature_extractor = torch.nn.Sequential(*list(model.children())[:-2])

    return feature_extractor, preprocess


def feature_extractor_vgg11():

    weights = VGG11_Weights.DEFAULT
    preprocess = weights.transforms()

    model = vgg11(weights=weights)
    model.eval()

    feature_extractor = torch.nn.Sequential(*list(model.children())[:-2])

    return feature_extractor, preprocess

def feature_extractor_vit():

    weights = ViT_B_16_Weights.DEFAULT
    preprocess = weights.transforms()

    model = vit_b_16(weights=weights)
    model.eval()

    feature_extractor = torch.nn.Sequential(*list(model.children())[:-2])

    return feature_extractor, preprocess

def feature_extractor_dinov2():

    preprocess = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225]),
    ])

    model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    model.eval()

    feature_extractor = model
    
    return feature_extractor, preprocess
    
    


if __name__ == '__main__':

    # change working directory
    os.chdir(sys.path[0])
    device = torch.device("cuda:0")

    # UNIT TEST FOR ALL MODELS
    # mock image tensor
    image_tensor = torch.zeros((1, 3, 256, 256)).to(device)

    # feature extractor initializations
    feature_resnet18, preprocess_resnet18 = feature_extractor_resnet18()
    # feature_alexnet, preprocess_alexnet = feature_extractor_alexnet()
    feature_convnext, preprocess_convnext =  feature_extractor_convnext()
    # feature_vgg11, preprocess_vgg11 = feature_extractor_vgg11()
    feature_vit, preprocess_vit = feature_extractor_vit()
    feature_dinov2, preprocess_dinov2 = feature_extractor_dinov2()

    ### RESNET18
    batch = preprocess_resnet18(image_tensor)
    feature_resnet18 = feature_resnet18.to(device)
    feature = feature_resnet18(batch)
    print(f'ResNet18\t feature shape: {feature.shape}')

    # ### ALEXNET
    # batch = preprocess_alexnet(image_tensor)
    # feature_alexnet = feature_alexnet.to(device)
    # feature = feature_alexnet(batch)
    # print(f'AlexNet\t feature shape: {feature.shape}')

    # ### VGG11
    # batch = preprocess_vgg11(image_tensor)
    # feature_vgg11 = feature_vgg11.to(device)
    # feature = feature_vgg11(batch)
    # print(f'VGG11 feature shape: {feature.shape}')

    ### ViT
    batch = preprocess_vit(image_tensor)
    feature_vit = feature_vit.to(device)
    feature = feature_vit(batch)
    print(f'ViT\t\t feature shape: {feature.shape}')

    ### DINOv2
    batch = preprocess_dinov2(image_tensor)
    feature_dinov2 = feature_dinov2.to(device)
    feature = feature_dinov2(batch)
    print(f'DINOv2\t\t feature shape: {feature.shape}')

    ### CONVNEXT
    batch = preprocess_convnext(image_tensor)
    feature_alexnet = feature_convnext.to(device)
    feature = feature_convnext(batch)
    print(f'ConvNeXt\t feature shape: {feature.shape}')



    