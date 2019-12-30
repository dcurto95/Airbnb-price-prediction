import time
import math
import os
import copy
from utils.utils import probability_to_prediction, masked_softmax, resize_, mean_avoid_0, masked_weighted_cross_entropy_loss, compare_two_frames_k_avg, masked_weighted_cross_entropy_loss2
from utils.input_output import save_mask_test, save_model, load_model, load_dist_matrix, run_n
from utils.evaluation_metrics import my_eval_iou, eval_metrics
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from others import paths


class Trainer:
    def __init__(self, device, model, train_loader, val_loader, args):
        self.device = device
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args

        # TODO save optimizer to resume
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
        self.scheduler = None

        self.from_epoch = 0

        self.best_epoch = None
        self.best_model_state = None
        self.best_val_acc = 0

        self.run = run_n(args.new_run)

        if args.overwrite_run:
            self.run = args.run

        # ########## LOAD RUN TO RESUME ##########
        if args.resume_run:
            self.run = args.run
            checkpoint_path = os.path.join(paths.MODELS_PATH, str(args.run), 'checkpointEpoch' + str(args.epoch) + '.pth')
            state, epoch, _ = load_model(checkpoint_path)
            self.model.load_state_dict(state)
            # self.model = self.model.to(self.device)
            self.from_epoch = epoch + 1
            print('Resuming training from run ' + str(args.run) + ' epoch ' + str(epoch + 1) + '...')

            if args.epoch == 'Best':
                self.best_epoch = epoch
            else:
                checkpoint_path = os.path.join(paths.MODELS_PATH, str(args.run), 'checkpointEpochBest.pth')
                self.best_model_state, self.best_epoch, _ = load_model(checkpoint_path)

        else:  # New run or Overwrite
            save_model(self.model.state_dict(), -1, True, self.run)
            save_model(self.model.state_dict(), -1, False, self.run)
            print('Starting training run ' + str(self.run) + '...')

        self.train_writer = SummaryWriter(os.path.join(paths.LOGS_PATH, str(self.run), 'train'))
        self.val_writer = SummaryWriter(os.path.join(paths.LOGS_PATH, str(self.run), 'val'))

        self.to_epoch = self.from_epoch + args.num_epochs

        self.save_each_epoch = 5
        self.print_each_train_batch = 20

    def train_model(self):
        # ########## TRAIN VAL LOOP ##########
        since = time.time()
        n_train_iterations = int(math.ceil(len(self.train_loader.dataset) / self.train_loader.batch_size))
        distance_matrix = load_dist_matrix(self.args.output_stride).to(self.device)

        for epoch in range(self.from_epoch, self.to_epoch):
            since_epoch = time.time()

            print('Starting Epoch {}/{}'.format(epoch + 1, self.to_epoch))
            print()

            # ########## TRAIN LOOP ##########
            print('Training...')
            self.model.train()

            epoch_train_loss = []
            accuracies = torch.empty(0)

            for ii, (frames, masks, info) in enumerate(self.train_loader):
                seq_name = info['name']
                frame = info['frame']
                n_objects = info['n_objects']           # Including background
                since_it = time.time()

                # Move all frames to GPU and forward pass them
                batch_size, n_frames, ch, h, w = frames.shape
                frames = frames.view(-1, ch, h, w)
                frames = frames.to(self.device)
                frames = self.model(frames)
                n_tot_frames, n_features, low_res_h, low_res_w = frames.shape
                frames = frames.view(batch_size, n_frames, n_features, low_res_h, low_res_w)

                max_n_objects = torch.max(n_objects).item()
                assert max_n_objects == torch.max(masks[:, 0]).item() + 1, 'Error in max_n_objects'

                masks_0 = resize_(masks[:, 0], low_res_h, low_res_w)  # (batch, ch=1, h, w) -> (batch, ch=1, h', w')
                masks_prev = resize_(masks[:, 1], low_res_h, low_res_w)  # (batch, ch=1, h, w) -> (batch, ch=1, h', w')
                masks_0 = masks_0.to(self.device)
                masks_prev = masks_prev.to(self.device)

                scores_0, has_data_0 = compare_two_frames_k_avg(frames[:, 0], frames[:, 2], masks_0, max_n_objects, self.args.k[0], self.args.lambd[0], distance_matrix)
                # assert torch.sum(has_data_0, dim=-1).cpu() == n_objects, 'Mask reduction has caused a loss of some objects'

                scores_prev, has_data_prev = compare_two_frames_k_avg(frames[:, 1], frames[:, 2], masks_prev, max_n_objects, self.args.k[1], self.args.lambd[1], distance_matrix)

                # Computing loss, backpropagation and optimizer step
                masks_t = resize_(masks[:, 2], low_res_h, low_res_w)
                masks_t = masks_t.to(self.device)

                self.optimizer.zero_grad()
                #
                probabilities_0 = masked_softmax(scores_0, has_data_0)
                probabilities_prev = masked_softmax(scores_prev, has_data_prev)
                probabilities = self.args.weight0*probabilities_0 + (1-self.args.weight0)*probabilities_prev
                #
                # probabilities = torch.max(scores_0.permute(0, 2, 3, 1), scores_prev.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                # probabilities = masked_softmax(probabilities, has_data_0)
                #
                loss = masked_weighted_cross_entropy_loss2(probabilities, masks_t, has_data_0, self.args.weighted_loss)

                ###
                # Before
                # loss_0 = masked_weighted_cross_entropy_loss(scores_0, masks_t, has_data_0, self.args.weighted_loss)
                # loss_prev = masked_weighted_cross_entropy_loss(scores_prev, masks_t, has_data_prev, self.args.weighted_loss)
                # loss = loss_0 + 0.25*loss_prev
                ###

                loss.backward()
                self.optimizer.step()

                # Prediction and accuracy
                scores_0 = resize_(scores_0, h, w, mode='bilinear', align_corners=True)
                scores_prev = resize_(scores_prev, h, w, mode='bilinear', align_corners=True)

                # probabilities_0 = masked_softmax(scores_0, has_data_0)
                # probabilities_prev = masked_softmax(scores_prev, has_data_prev)
                # probabilities = torch.cat((probabilities_0, probabilities_prev), dim=1)

                probabilities = torch.cat((scores_0, scores_prev), dim=1)
                # probabilities = scores_0
                # probabilities = scores_prev

                predicted_masks = probability_to_prediction(probabilities, max_n_objects)
                gt_masks = masks[:, 2].to(self.device)

                ious = my_eval_iou(predicted_masks, gt_masks, max_n_objects)
                ious = mean_avoid_0(ious * has_data_0[:, 1:].float().cpu(), dim=1)
                accuracies = torch.cat((accuracies, ious), 0)

                epoch_train_loss.append(loss.item())
                if ((ii + 1) % self.print_each_train_batch == 0) or (ii == 0) or (ii == n_train_iterations - 1):
                    print("Iteration {:4}/{:4} | loss: {:05f} | Time spent: {:10.4f}ms".
                          format(ii + 1, n_train_iterations, loss, (time.time() - since_it)*1000))
            train_accuracy = torch.mean(accuracies)
            print()

            print("Validating...")
            self.model.eval()
            with torch.no_grad():
                epoch_val_loss = []
                val_accuracy = []
                val_accuracy_top1 = []
                val_accuracy_top5 = []
                for ii, (frames, masks, info) in enumerate(self.val_loader):
                    seq_name = info['name'][0]
                    n_frames = info['n_frames'][0].item()
                    n_objects = info['n_objects'][0].item()
                    original_shape = tuple([x.item() for x in info['original_shape']])

                    sequence_time = 0
                    since_frame_0 = time.time()

                    # Move frame 0 to GPU and forward pass it
                    frame_0 = frames[:, 0]
                    _, ch, h, w = frame_0.shape
                    frame_0 = frame_0.to(self.device)
                    frame_0 = self.model(frame_0)  # (1, ch, h, w)
                    frame_prev = frame_0
                    _, n_features, low_res_h, low_res_w = frame_0.shape

                    max_n_objects = torch.max(masks[:, 0]).item() + 1
                    assert n_objects == max_n_objects, 'Error in n_objects'

                    masks_0 = resize_(masks[:, 0], low_res_h, low_res_w)  # (batch, ch=1, h, w) -> (batch, ch=1, h', w')
                    masks_0 = masks_0.to(self.device)
                    masks_prev = masks_0

                    sequence_time += time.time() - since_frame_0
                    sequence_loss = []

                    batch_size = 1
                    t = 1
                    n_minibatchs = math.ceil((n_frames - 1) / batch_size)
                    accuracies = torch.empty(0)
                    for i in range(n_minibatchs):
                        since_frame_t_forward = time.time()
                        if i == n_minibatchs - 1:  # Last minibatch
                            frame_t = frames[0, t:]
                            masks_t = masks[0, t:]
                        else:
                            frame_t = frames[0, t:t+batch_size]
                            masks_t = masks[0, t:t+batch_size]
                        frame_t = frame_t.to(self.device)
                        frame_t = self.model(frame_t)

                        scores_0, has_data_0 = compare_two_frames_k_avg(frame_0, frame_t, masks_0, max_n_objects, self.args.k[0], self.args.lambd[0], distance_matrix)

                        scores_prev, has_data_prev = compare_two_frames_k_avg(frame_prev, frame_t, masks_prev, max_n_objects, self.args.k[1], self.args.lambd[1], distance_matrix)
                        sequence_time += time.time() - since_frame_t_forward

                        # Computing loss, backpropagation and optimizer step
                        masks_t_low = resize_(masks_t, low_res_h, low_res_w)
                        masks_t_low = masks_t_low.to(self.device)

                        loss_0 = masked_weighted_cross_entropy_loss(scores_0, masks_t_low, has_data_0, self.args.weighted_loss)
                        loss_prev = masked_weighted_cross_entropy_loss(scores_prev, masks_t_low, has_data_prev, self.args.weighted_loss)
                        loss = loss_0 + loss_prev
                        sequence_loss.append(loss.item())

                        # Save low_res_mask
                        since_frame_t_low = time.time()
                        # probabilities_0_low = masked_softmax(scores_0)
                        # probabilities_prev_low = masked_softmax(scores_prev)
                        # probabilities_low = torch.cat((probabilities_0_low, probabilities_prev_low), dim=1)

                        probabilities_low = torch.cat((scores_0, scores_prev), dim=1)
                        # probabilities_low = scores_0
                        # probabilities_low = scores_prev

                        predicted_masks_low = probability_to_prediction(probabilities_low, max_n_objects)
                        sequence_time += time.time() - since_frame_t_low

                        # Prediction and accuracy
                        since_frame_t_pred = time.time()
                        scores_0 = resize_(scores_0, original_shape[1], original_shape[0], mode='bilinear', align_corners=True)
                        scores_prev = resize_(scores_prev, original_shape[1], original_shape[0], mode='bilinear', align_corners=True)

                        # scores = torch.cat((scores_0, scores_prev), dim=1)
                        # probabilities = masked_softmax(scores)
                        # probabilities_0 = masked_softmax(scores_0, has_data_0)
                        # probabilities_prev = masked_softmax(scores_prev, has_data_prev)
                        # probabilities = torch.cat((probabilities_0, probabilities_prev), dim=1)

                        probabilities = torch.cat((scores_0, scores_prev), dim=1)
                        # probabilities = scores_0
                        # probabilities = scores_prev

                        predicted_masks = probability_to_prediction(probabilities, max_n_objects)
                        sequence_time += time.time() - since_frame_t_pred
                        gt_masks = resize_(masks_t, original_shape[1], original_shape[0])
                        gt_masks = gt_masks.to(self.device)

                        ious = my_eval_iou(predicted_masks, gt_masks, n_objects)
                        accuracies = torch.cat((accuracies, ious), 0)
                        sequence_accuracy = torch.mean(accuracies).item()

                        frame_prev = frame_t
                        masks_prev = predicted_masks_low
                        t += batch_size

                    val_accuracy.append(sequence_accuracy)
                    val_accuracy_top1.append(torch.mean(accuracies[0]).item())
                    val_accuracy_top5.append(torch.mean(accuracies[0:5]).item())
                    sequence_loss = sum(sequence_loss) / len(sequence_loss)
                    epoch_val_loss.append(sequence_loss)
                    print('{:<20} | FPS: {:7.4f} | mIoU: {:>7.4f}%'.format(seq_name, n_frames / sequence_time, sequence_accuracy*100))

            epoch_train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
            # epoch_train_loss = 0
            epoch_train_acc = train_accuracy.item()
            epoch_val_loss = sum(epoch_val_loss) / len(epoch_val_loss)
            epoch_val_acc = sum(val_accuracy) / len(val_accuracy)
            epoch_val_accuracy_top1 = sum(val_accuracy_top1) / len(val_accuracy_top1)
            epoch_val_accuracy_top5 = sum(val_accuracy_top5) / len(val_accuracy_top5)

            self.train_writer.add_scalar('loss', epoch_train_loss, epoch + 1)
            self.train_writer.add_scalar('accuracy', epoch_train_acc, epoch + 1)
            self.val_writer.add_scalar('loss', epoch_val_loss, epoch + 1)
            self.val_writer.add_scalar('accuracy', epoch_val_acc, epoch + 1)
            self.val_writer.add_scalar('accuracy_frame1', epoch_val_accuracy_top1, epoch + 1)
            self.val_writer.add_scalar('accuracy_frame1to5', epoch_val_accuracy_top5, epoch + 1)

            # Save model each save_each_epoch epochs and last
            if ((epoch + 1) % self.save_each_epoch == 0) or (epoch + 1) == self.to_epoch:
                save_model(self.model.state_dict(), epoch, False, self.run)

            # Save best model
            # if epoch_val_loss < self.best_val_loss:
            if epoch_val_acc > self.best_val_acc:
                self.best_val_acc = epoch_val_acc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                save_model(self.best_model_state, self.best_epoch, True, self.run)

            print()
            print('End of Epoch {}/{} | time: {} | train loss: {} | val loss: {} | val accuracy: {:.4f}%'.
                  format(epoch+1, self.to_epoch, time.time() - since_epoch, epoch_train_loss, epoch_val_loss, epoch_val_acc*100))
            print()

        # Save best model
        save_model(self.best_model_state, self.best_epoch, True, self.run)
        print('Training completed. Elapsed time: {}s | Best validation accuracy: {}'.
              format(time.time() - since, self.best_val_acc))


class Tester:
    def __init__(self, device, model, test_loader, args):
        self.device = device
        self.model = model
        self.test_loader = test_loader
        self.export_predictions = args.export_predictions
        self.run = args.run
        self.args = args

        _checkpoint_path = os.path.join(paths.MODELS_PATH, str(args.run), 'checkpointEpoch' + str(args.epoch) + '.pth')
        state, epoch, _ = load_model(_checkpoint_path)
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device)
        print('Testing model using checkpoint of run ' + str(args.run) + ' epoch ' + str(epoch + 1) + ' ...')

    def test(self):
        self.model.eval()
        with torch.no_grad():
            since = time.time()
            val_J = []
            # val_davis_accuracy = []
            val_F_score = []
            val_J_top1 = []
            val_J_top5 = []
            times = []
            n_framess = []
            fpss = []

            if self.export_predictions:
                output_file = open(os.path.join(paths.RESULTS_PATH, str(self.run), self.args.export_dir, 'output.txt'), 'w')
                output_file.write('Sequence,J-Mean,F-Mean\n')

            distance_matrix = load_dist_matrix(self.args.output_stride).to(self.device)
            for ii, (frames, masks, info) in enumerate(self.test_loader):
                seq_name = info['name'][0]
                n_frames = info['n_frames'][0].item()
                n_objects = info['n_objects'][0].item()
                original_shape = tuple([x.item() for x in info['original_shape']])
                has_gt = info['has_gt'][0]
                palette = [x.item() for x in info['palette'][0]]

                sequence_time = 0
                since_frame_0 = time.time()

                # Move frame 0 to GPU and forward pass it
                frame_0 = frames[:, 0]
                _, ch, h, w = frame_0.shape
                frame_0 = frame_0.to(self.device)
                frame_0 = self.model(frame_0)  # (1, ch, h, w)
                frame_prev = frame_0
                _, n_features, low_res_h, low_res_w = frame_0.shape

                max_n_objects = torch.max(masks[:, 0]).item() + 1
                assert n_objects == max_n_objects, 'Error in n_objects'

                masks_0 = resize_(masks[:, 0], low_res_h, low_res_w)  # (batch, ch=1, h, w) -> (batch, ch=1, h', w')
                masks_0 = masks_0.to(self.device)
                masks_prev = masks_0

                sequence_time += time.time() - since_frame_0

                batch_size = 1
                t = 1
                n_minibatchs = math.ceil((n_frames - 1) / batch_size)
                J = torch.empty(0)
                # davis_accuracies = torch.empty(0)
                F_scores = torch.empty(0)
                for i in range(n_minibatchs):
                    since_frame_t_forward = time.time()
                    if i == n_minibatchs - 1:  # Last minibatch
                        frame_t = frames[0, t:]
                        if has_gt:
                            masks_t = masks[0, t:]
                    else:
                        frame_t = frames[0, t:t + batch_size]
                        if has_gt:
                            masks_t = masks[0, t:t + batch_size]
                    frame_t = frame_t.to(self.device)
                    frame_t = self.model(frame_t)

                    scores_0, has_data_0 = compare_two_frames_k_avg(frame_0, frame_t, masks_0, max_n_objects, self.args.k[0], self.args.lambd[0], distance_matrix)

                    scores_prev, has_data_prev = compare_two_frames_k_avg(frame_prev, frame_t, masks_prev, max_n_objects, self.args.k[1], self.args.lambd[1], distance_matrix)
                    sequence_time += time.time() - since_frame_t_forward

                    # Save low_res_mask
                    since_frame_t_low = time.time()
                    # probabilities_0_low = masked_softmax(scores_0)
                    # probabilities_prev_low = masked_softmax(scores_prev)

                    probabilities_low = torch.cat((scores_0, scores_prev), dim=1)
                    # probabilities_low = scores_0
                    # probabilities_low = scores_prev

                    # probabilities_low = 0.5*scores_0 + 0.5*scores_prev
                    # probabilities_low = torch.cat((probabilities_0_low, probabilities_prev_low), dim=1)
                    predicted_masks_low = probability_to_prediction(probabilities_low, max_n_objects)
                    sequence_time += time.time() - since_frame_t_low

                    # Prediction and accuracy
                    since_frame_t_pred = time.time()
                    scores_0 = resize_(scores_0, original_shape[1], original_shape[0], mode='bilinear', align_corners=True)
                    scores_prev = resize_(scores_prev, original_shape[1], original_shape[0], mode='bilinear', align_corners=True)

                    # scores = torch.cat((scores_0, scores_prev), dim=1)
                    # probabilities = masked_softmax(scores)
                    # probabilities_0 = masked_softmax(scores_0, has_data_0)
                    # probabilities_prev = masked_softmax(scores_prev, has_data_prev)
                    # probabilities = torch.cat((probabilities_0, probabilities_prev), dim=1)

                    probabilities = torch.cat((scores_0, scores_prev), dim=1)
                    # probabilities = scores_0
                    # probabilities = scores_prev

                    # probabilities = 0.5*scores_0 + 0.5*scores_prev

                    predicted_masks = probability_to_prediction(probabilities, max_n_objects)
                    sequence_time += time.time() - since_frame_t_pred
                    if has_gt:
                        gt_masks = resize_(masks_t, original_shape[1], original_shape[0])
                        gt_masks = gt_masks.to(self.device)

                        metrics = eval_metrics(predicted_masks, gt_masks, n_objects)
                        J = torch.cat((J, metrics[0]), 0)
                        # davis_accuracies = torch.cat((davis_accuracies, metrics[1]), 0)
                        F_scores = torch.cat((F_scores, metrics[2]), 0)

                    if self.args.export_predictions:
                        save_mask_test(predicted_masks, seq_name, t, palette, self.run, self.args.export_dir)

                    # TODO try to not resize at dataloader, try prev_mask = gt_mask
                    frame_prev = frame_t
                    # masks_prev = resize_(predicted_masks, low_res_h, low_res_w)
                    masks_prev = predicted_masks_low
                    t += batch_size

                times.append(sequence_time)
                n_framess.append(n_frames)
                fps = n_frames / sequence_time
                fpss.append(fps)

                if has_gt:
                    frame_J = torch.mean(J, dim=1)
                    frame_F = torch.mean(F_scores, dim=1)

                    sequence_J = torch.mean(frame_J).item()
                    # sequence_davis_accuracy = torch.mean(frame_J).item()
                    sequence_F_score = torch.mean(frame_F).item()

                    if self.export_predictions:
                        object_J = torch.mean(J, dim=0)
                        object_F = torch.mean(F_scores, dim=0)
                        for n in range(n_objects-1):
                            output_file.write('{},{:.3f},{:.3f}\n'.format(seq_name + '_' + str(n+1), object_J[n].item(), object_F[n].item()))

                    val_J.append(sequence_J)
                    val_J_top1.append(torch.mean(frame_J[0]).item())
                    val_J_top5.append(torch.mean(frame_J[0:5]).item())
                    # val_davis_accuracy.append(sequence_davis_accuracy)
                    val_F_score.append(sequence_F_score)
                    print('{:<20} | FPS: {:7.4f} | J: {:>7.4f}% | F: {:>7.4f}%'.format(seq_name, fps, sequence_J * 100, sequence_F_score * 100))
                else:
                    print('{:<20} | FPS: {:7.4f}'.format(seq_name, fps))

            times = sum(times)
            n_framess = sum(n_framess)
            fps_mean = sum(fpss) / len(fpss)
            fps_real = n_framess / times
            print()

            if has_gt:
                val_J = sum(val_J) / len(val_J)
                val_J_top1 = sum(val_J_top1) / len(val_J_top1)
                val_J_top5 = sum(val_J_top5) / len(val_J_top5)
                # val_davis_accuracy = sum(val_davis_accuracy) / len(val_davis_accuracy)
                val_F_score = sum(val_F_score) / len(val_F_score)
                if self.export_predictions:
                    output_file.close()
                print('Total time spent: {:.0f}s\nJ: {:7.4f}% | F: {:7.4f}% | G_mean: {:7.4f}% | First frames J: {:7.4f}% | First 5 frames J: {:7.4f}%\nFPS real: {:.4f} | FPS mean: {:.4f}'.format(time.time() - since, val_J*100, val_F_score*100, (val_J + val_F_score)*50, val_J_top1*100, val_J_top5*100, fps_real, fps_mean))
            else:
                print('Total time spent: {:.0f}s\nFPS real: {:.4f} | FPS mean: {:.4f}'.format(time.time() - since, fps_real, fps_mean))
