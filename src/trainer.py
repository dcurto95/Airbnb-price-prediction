import copy
import os
import time

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils import save_model, load_model, create_log_dirs


class Trainer:
    def __init__(self, device, model, train_loader, val_loader, args):
        self.device = device
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args

        self.optimizer = optim.Adam(self.model.parameters(), lr=args['lr'], weight_decay=1e-4)
        self.criterion = torch.nn.functional.mse_loss

        self.best_epoch = None
        self.best_model_state = None
        self.best_val_loss = float("Inf")

        logs_path = create_log_dirs(args)
        self.train_writer = SummaryWriter(os.path.join(logs_path, 'train'))
        self.val_writer = SummaryWriter(os.path.join(logs_path, 'val'))

        self.print_each_epochs = 25

    def train_model(self):
        # ########## TRAIN VAL LOOP ##########
        since = time.time()

        for epoch in range(self.args['n_epochs']):
            since_epoch = time.time()

            # ########## TRAIN LOOP ##########
            self.model.train()

            epoch_train_loss = []

            for ii, (features, gt) in enumerate(self.train_loader):
                # Move all data to device and forward pass
                features = features.to(self.device)
                prediction = self.model(features)

                gt = gt.to(self.device)

                self.optimizer.zero_grad()

                loss = self.criterion(prediction, gt)
                loss.backward()
                self.optimizer.step()

                loss2 = self.criterion(torch.exp(prediction), torch.exp(gt))
                epoch_train_loss.append(loss2.item())

            # ########## VAL LOOP ##########
            epoch_val_loss = validate(self.model, self.val_loader, self.device, self.criterion)

            epoch_train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
            epoch_val_loss = sum(epoch_val_loss) / len(epoch_val_loss)

            self.train_writer.add_scalar('loss', epoch_train_loss, epoch + 1)
            self.val_writer.add_scalar('loss', epoch_val_loss, epoch + 1)

            if (epoch + 1) % self.print_each_epochs == 0 or epoch == 0 or epoch == (self.args['n_epochs'] - 1):
                print('Epoch {:5}/{:5} | time: {:7.4f}s | Train MSE loss: {:10.4f} | Val MSE loss: {:10.4f}'.format(
                    epoch + 1, self.args['n_epochs'], time.time() - since_epoch,
                    epoch_train_loss, epoch_val_loss))

            # Save best model
            if epoch_val_loss < self.best_val_loss:
                self.best_val_loss = epoch_val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                save_model(self.best_model_state, self.args['run_name'], epoch_train_loss, epoch_val_loss)

        print('Training completed. Elapsed time: {}s | Best validation loss: {}'.
              format(time.time() - since, self.best_val_loss))


def validate(model, loader, device, criterion):
    model.eval()
    with torch.no_grad():
        epoch_val_loss = []

        for ii, (features, gt) in enumerate(loader):
            features = features.to(device)
            prediction = model(features)

            gt = gt.to(device)

            loss = criterion(torch.exp(prediction), torch.exp(gt))

            epoch_val_loss.append(loss.item())
    return epoch_val_loss


class Tester:
    def __init__(self, device, model, test_loader, args):
        self.device = device
        self.model = model
        self.test_loader = test_loader
        self.args = args
        self.criterion = torch.nn.functional.mse_loss

        state, train_loss, val_loss = load_model(args['model'])
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device)
        print('Testing model  ' + args[
            'model'] + '.\nThis model obtained Training Loss = {:.4f} | Validation Loss = {:.4f}'.format(train_loss,
                                                                                                         val_loss))

    def test(self):
        since = time.time()
        test_loss = validate(self.model, self.test_loader, self.device, self.criterion)
        print('Total time spent: {:.0f}s | Mean test MSE: {:.4f}'.format(time.time() - since,
                                                                         sum(test_loss) / len(test_loss)))
