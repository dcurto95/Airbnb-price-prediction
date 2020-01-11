from datetime import datetime

import torch
from torch.utils.data import DataLoader

from dataloaders.airbnb import AIRBNB
from models import mlp
from trainer import Trainer, Tester

train_mode = False
run_name = "Best"

run_name = run_name + datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')

args = {
    'model': 'Best_2020-01-11_10-39-32',

    'fuzzy': False,
    'neigh': True,

    'n_epochs': 1000,
    'run_name': run_name,

    'optimizer': 'Adam',
    'lr': 5e-4,

    'batch_size': 250,

    'hidden': [300, 1],
    'activation': mlp.RELU
}

dataset_name = 'fuzzy' if args['fuzzy'] else 'cleaned'
dataset_name = dataset_name if args['neigh'] else dataset_name + '_wo_Neigh'

# ########## DATASETS AND DATALOADERS ##########
print('Preparing datasets...')

if train_mode:
    train_set = AIRBNB(path='../data/', data_set='train_' + dataset_name + '.csv')
    val_set = AIRBNB(path='../data/', data_set='validation_' + dataset_name + '.csv')
    train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False, num_workers=0, pin_memory=True)
    test_loader = None
    n_features = train_set.get_n_features()

else:
    train_loader, val_loader = None, None
    test_set = AIRBNB(path='../data/', data_set='test_' + dataset_name + '.csv')
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False, num_workers=0, pin_memory=True)
    n_features = test_set.get_n_features()


# ########## Net Init ##########
print('Network initialization...')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on ' + device.type)

model = mlp.MLP(n_features, n_hidden_units=args['hidden'], activation_function=args['activation'])

# MultiGPU
# if torch.cuda.device_count() > 1:
#     print('We are using', torch.cuda.device_count(), 'GPUs')
#     model = nn.DataParallell(model)
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
