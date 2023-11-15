import argparse

def get_args():
    # Training settings
    parser = argparse.ArgumentParser('train')
    
    parser.add_argument('--resolution', type=str, default='256', help='Resolution of input image')
    parser.add_argument('--magnification', type=str, default='20x', help='Magnification of input image')
    parser.add_argument('--modality', type=str, default='texture', help='Modality of input image')
    parser.add_argument('--model', type=str, default='ResNet50', help='Model to use')
    parser.add_argument('--pretrained', type=str, default='pretrained', help='Use pretrained model')
    parser.add_argument('--frozen', type=str, default='unfrozen', help='Freeze pretrained model')
    parser.add_argument('--vote', type=str, default='vote', help='Conduct voting')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--start_lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    return parser.parse_args()