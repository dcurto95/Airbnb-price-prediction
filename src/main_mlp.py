from models.mlp import MLP
from trainer import Trainer, Tester

import torch
from torch.utils.data import DataLoader
from dataloaders.airbnb import AIRBNB
from datetime import datetime

run_name = "Test Rafel"

run_name = run_name + datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')
args = {
    'n_epochs': 100,
    'run_name': run_name,

    'lr': 1e-6,
    'momentum': 0.9


}

train_mode = True

# ########## DATASETS AND DATALOADERS ##########
print('Preparing datasets...')

if train_mode:
    train_set = AIRBNB(path='../data/', data_set='train.csv')
    val_set = AIRBNB(path='../data/', data_set='val.csv')
    train_loader = DataLoader(train_set, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False, num_workers=4, pin_memory=True)
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

model = MLP(n_features)

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
