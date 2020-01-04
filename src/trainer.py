import time
import math
import os
import copy
from utils import save_model, load_model, create_log_dirs
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, device, model, train_loader, val_loader, args):
        self.device = device
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args

        # TODO choose optimizer and scheduler?
        self.optimizer = optim.SGD(self.model.parameters(), lr=args['lr'], momentum=args['momentum'])  # , weight_decay=args.weight_decay)
        self.scheduler = None
        self.criterion = torch.nn.functional.mse_loss

        self.best_epoch = None
        self.best_model_state = None
        self.best_val_loss = float("Inf")

        logs_path = create_log_dirs(args['run_name'])
        self.train_writer = SummaryWriter(os.path.join(logs_path, args['run_name'], 'train'))
        self.val_writer = SummaryWriter(os.path.join(logs_path, args['run_name'], 'val'))

        self.print_each_train_batch = 20

    def train_model(self):
        # ########## TRAIN VAL LOOP ##########
        since = time.time()
        n_train_iterations = int(math.ceil(len(self.train_loader.dataset) / self.train_loader.batch_size))
        n_val_iterations = int(math.ceil(len(self.val_loader.dataset) / self.val_loader.batch_size))

        for epoch in range(self.args['n_epochs']):
            since_epoch = time.time()

            # print('Starting Epoch {}/{}'.format(epoch + 1, self.args['n_epochs']))
            # print()

            # ########## TRAIN LOOP ##########
            # print('Training...')
            self.model.train()

            epoch_train_loss = []

            for ii, (features, gt) in enumerate(self.train_loader):
                since_it = time.time()

                # Move all data to device and forward pass
                # batch_size, n_features = features.shape
                features = features.to(self.device)
                prediction = self.model(features)

                gt = gt.to(self.device)

                self.optimizer.zero_grad()

                loss = self.criterion(prediction, gt)
                loss.backward()
                self.optimizer.step()

                epoch_train_loss.append(loss.item())
                # if ((ii + 1) % self.print_each_train_batch == 0) or (ii == 0) or (ii == n_train_iterations - 1):
                #     print("Iteration {:4}/{:4} | loss: {:05f} | Time spent: {:10.4f}ms".
                #           format(ii + 1, n_train_iterations, loss, (time.time() - since_it)*1000))
            # print()

            # print("Validating...")
            epoch_val_loss = validate(self.model, self.val_loader, self.device, self.criterion)

            epoch_train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
            epoch_val_loss = sum(epoch_val_loss) / len(epoch_val_loss)

            self.train_writer.add_scalar('loss', epoch_train_loss, epoch + 1)
            self.val_writer.add_scalar('loss', epoch_val_loss, epoch + 1)

            print()
            print('End of Epoch {}/{} | time: {}s | train loss: {} | val loss: {}'.
                  format(epoch + 1, self.args['n_epochs'], time.time() - since_epoch, epoch_train_loss, epoch_val_loss))
            print()

            # Save best model
            # if epoch_val_loss < self.best_val_loss:
            #     self.best_val_loss = epoch_val_loss
            #     self.best_model_state = copy.deepcopy(self.model.state_dict())
            #     self.best_epoch = epoch
            #     save_model(self.best_model_state, self.args['run_name'], epoch_train_loss, epoch_val_loss)

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

            loss = criterion(prediction, gt)

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
        print('Testing model  ' + args['model'] + '.\nThis model obtained Training Loss = {:.4f} | Validation Loss = {:.4f}'.format(train_loss, val_loss))

    def test(self):
        since = time.time()
        test_loss = validate(self.model, self.test_loader, self.device, self.criterion)
        print('Total time spent: {:.0f}s | Mean test MSE: {:.4f}'.format(time.time() - since, sum(test_loss)/len(test_loss)))
