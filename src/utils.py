import torch
import os


def save_model(model_state_dict, run_name, train_loss, val_loss):
    file = run_name + '.pth'
    path = os.path.join('..', 'models')
    if not os.path.exists(path):
        os.makedirs(path)
    dict_ = {'model_state_dict': model_state_dict, 'train_loss': train_loss, 'val_loss': val_loss}
    torch.save(dict_, os.path.join(path, file))
    return True


def load_model(run_name):
    model_file = run_name + '.pth'
    path = os.path.join('..' 'models', model_file)
    assert os.path.isfile(path), 'Resume checkpoint not existing'
    model = torch.load(path)
    weights = model['model_state_dict']
    train_loss = model['train_loss']
    val_loss = model['val_loss']
    return weights, train_loss, val_loss


def create_log_dirs(args):
    run_name = args['run_name']
    path = os.path.join('..', 'logs', run_name)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, 'config.txt'), 'w') as f:
        f.write(str(args))
    return path
