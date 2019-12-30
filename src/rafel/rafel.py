import argparse
import os

from models.my_resnet import resnet50, resnet34, resnet101
from dataloaders.davis import DAVISAllSequence, DAVISCurrentFirstAndPrevious
from trainer import Trainer, Tester
from utils.input_output import check_all_arguments

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Rafel\'s network', prog='Rafel')

# parser.add_argument('-p', '--path', type=str, required=True,                                              help='path to DAVIS dataset')
parser.add_argument('-p', '--path', type=str, default='/scratch/gobi1/rafelps/datasets/DAVIS',              help='path to DAVIS dataset')

mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument('-t', '--train', action='store_true',                                                     help='activates training mode on trainval set')
mode.add_argument('-e', '--eval', nargs=3, metavar=('IMAGESET', 'RUN', 'EPOCH'), default=None,              help='activates evaluation mode on IMAGESET set from run RUN and epoch EPOCH checkpoint')

exe = parser.add_mutually_exclusive_group(required=False)
exe.add_argument('-rr', '--resume_run', nargs=2, metavar=('RUN', 'EPOCH'), default=None,                    help='resumes training from run RUN and epoch EPOCH checkpoint')
exe.add_argument('-nr', '--new_run', action='store_true',                                                   help='starts a new run')
exe.add_argument('-or', '--overwrite_run', type=int, metavar='RUN', default=None,                           help='overwrites run RUN')

training = parser.add_argument_group('training arguments')
training.add_argument('-lr', '--learning_rate', type=float, default=5e-4,                                   help='learning rate')
training.add_argument('-n', '--num_epochs', type=int, default=100,                                          help='number of training epochs')
training.add_argument('-wd', '--weight_decay', type=float, default=0,                                       help='weight decay')

model = parser.add_argument_group('model arguments')
model.add_argument('-os', '--output_stride', type=int, choices=[8, 16], default=8,                          help='output stride')
model.add_argument('-d', '--dilation', type=int, choices=[1, 2], default=1,                                 help='dilation in last layer convolutions')
model.add_argument('-fl', '--freeze_layers', action='store_true',                                           help='freezes all but last layer')
model.add_argument('-bn', '--freeze_bn', action='store_true',                                               help='freezes batch normalization layers')
model.add_argument('-in', '--image_normalization', action='store_true',                                     help='activates image normalization')

method = parser.add_argument_group('method arguments')
method.add_argument('-k', nargs=2, metavar=('k_0', 'k_prev'), type=int, default=(1, 10),                    help='average of top k scores')
method.add_argument('-l', '--lambd', nargs=2, metavar=('lambda_0', 'lambda_prev'), type=float, default=(0.5, 0.5),      help='weight for distance score')
method.add_argument('-wl', '--weighted_loss', action='store_false',                                         help='Disable weighted CEL')
method.add_argument('-w0', '--weight0', type=float, default=1.0,                                            help='Weight for frame0 reference probabilities')

testing = parser.add_argument_group('testing arguments')
testing.add_argument('-ep', '--export_predictions', type=str, metavar='DIRECTORY',                          help='store generated predictions in directory DIRECTORY')

print('Parsing arguments...')
args = parser.parse_args()
check_all_arguments(args)
train_mode = args.train

# ########## DATASETS AND DATALOADERS ##########
print('Preparing datasets...')
train_loader, val_loader, test_loader = None, None, None
if train_mode:
    train_set = DAVISCurrentFirstAndPrevious(davis_root_dir=args.path, image_set='train')#, multi_object=False)
    val_set = DAVISAllSequence(davis_root_dir=args.path, image_set='val')#, multi_object=False)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

else:
    test_set = DAVISAllSequence(davis_root_dir=args.path, image_set=args.image_set)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)


# ########## Net Init ##########
print('Network initialization...')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = resnet50(pretrained=True,
                 output_stride=args.output_stride,
                 dilation=args.dilation,
                 normalize=args.image_normalization,
                 freeze_layers=args.freeze_layers,
                 freeze_bn=args.freeze_bn)

if torch.cuda.device_count() > 1:
    print('We are using', torch.cuda.device_count(), 'GPUs')
    model = nn.DataParallell(model)
model = model.to(device)

# ########## TRAIN VAL LOOP ##########
if train_mode:
    print('Trainer initialization...')
    trainer = Trainer(device, model, train_loader, val_loader, args)
    trainer.train_model()

# ########## TEST ##########
else:
    print('Tester initialization...')
    tester = Tester(device, model, test_loader, args)
    tester.test()
