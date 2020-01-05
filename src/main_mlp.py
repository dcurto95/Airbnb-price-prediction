from models import mlp
from trainer import Trainer, Tester

import torch
from torch.utils.data import DataLoader
from dataloaders.airbnb import AIRBNB
from datetime import datetime

run_name = "Test Rafel"

run_name = run_name + datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')
args = {
    'n_epochs': 10000,
    'run_name': run_name,

    'optimizer': 'SGD',
    'lr': 1e-4,
    'momentum': 0.9,

    'scheduler': {'gamma': 0.5, 'milestones': [5, 100, 200, 300, 400, 500, 600, 700, 800]},

    'batch_size': 'all',

    'hidden': [500, 500, 250, 50, 1],
    'activation': mlp.RELU
}

train_mode = True

# ########## DATASETS AND DATALOADERS ##########
print('Preparing datasets...')

if train_mode:
    train_set = AIRBNB(path='../data/', data_set='AB_NYC_2019_cleaned.csv')
    val_set = AIRBNB(path='../data/', data_set='AB_NYC_2019_cleaned.csv')
    train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False, num_workers=0, pin_memory=True)
    test_loader = None
    n_features = train_set.get_n_features()

else:
    train_loader, val_loader = None, None
    test_set = AIRBNB(path='../data/', data_set='test.csv')
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False, num_workers=4, pin_memory=True)
    n_features = test_set.get_n_features()


# ########## Net Init ##########
print('Network initialization...')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on ' + device.type)

# TODO DAVIDS Check that it is able to run in GPU

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
