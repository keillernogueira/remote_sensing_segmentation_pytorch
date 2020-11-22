import sys
import time
import datetime
import imageio

import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

from config import *
from utils import *

from dataloaders.factory import dataloader_factory

from networks.factory import model_factory
from networks.loss import CrossEntropyLoss

cudnn.benchmark = True
torch.set_default_tensor_type(torch.DoubleTensor)


def test(test_loader, test_set, net, output_path, epoch, gpu, save_images=False):
    # Setting network for evaluation mode.
    net.eval()

    prob_im = np.zeros([test_set.data.shape[0], test_set.data.shape[1], test_set.data.shape[2],
                        test_set.num_classes], dtype=np.float32)
    occur_im = np.zeros([test_set.data.shape[0], test_set.data.shape[1], test_set.data.shape[2],
                        test_set.num_classes], dtype=np.float32)

    with torch.no_grad():
        # Iterating over batches.
        for i, data in enumerate(test_loader):

            # Obtaining images, labels and paths for batch.
            inps, labs, masks, cur_maps, cur_xs, cur_ys = data

            inps = inps.squeeze()
            labs = labs.squeeze()
            masks = masks.squeeze()

            # Casting to cuda variables.
            inps = Variable(inps).cuda(gpu)
            labs = Variable(labs).cuda(gpu)
            masks = Variable(masks).cuda(gpu)

            # Forwarding.
            if save_images:
                outs, dec1, dec2, dec3 = net(inps, feat=True)
                feat_flat = torch.cat([outs, dec1, dec2, dec3], 1)
            else:
                outs = net(inps, feat=False)

            # Computing probabilities.
            soft_outs = F.softmax(outs, dim=1)

            # Obtaining prior predictions.
            prds = soft_outs.data.max(1)[1]

            # prds_flat = prds.cpu().numpy()
            # masks_flat = masks.cpu().numpy()

            for j in range(prds.shape[0]):
                cur_map = cur_maps[j]
                cur_x = cur_xs[j]
                cur_y = cur_ys[j]

                outs_p = outs.permute(0, 2, 3, 1).cpu().numpy()

                prob_im[cur_map, cur_x:cur_x + test_set.crop_size,
                        cur_y:cur_y + test_set.crop_size, :] += outs_p[j, :, :, :]
                occur_im[cur_map, cur_x:cur_x + test_set.crop_size, cur_y:cur_y + test_set.crop_size, :] += 1

        occur_im[np.where(occur_im == 0)] = 1
        prob_im_argmax = np.argmax(prob_im / occur_im.astype(float), axis=-1)

        # Saving predictions.
        if save_images:
            for k, img_name in enumerate(test_set.names):
                pred_path = os.path.join(output_path, img_name.replace('.tif', '_prd.png'))
                imageio.imsave(pred_path, prob_im_argmax[k])

        cm_test = create_cm(test_set.labels, prob_im_argmax)

        _sum = 0.0
        total = 0
        for k in range(len(cm_test)):
            _sum += (cm_test[k][k] / float(np.sum(cm_test[k])) if np.sum(cm_test[k]) != 0 else 0)
            total += cm_test[k][k]

        _sum_iou = (cm_test[1][1] / float(
            np.sum(cm_test[:, 1]) + np.sum(cm_test[1]) - cm_test[1][1])
                    if (np.sum(cm_test[:, 1]) + np.sum(cm_test[1]) - cm_test[1][1]) != 0
                    else 0)

        print("Epoch " + str(epoch) +
              " -- Time " + str(datetime.datetime.now().time()) +
              " Absolut Right Pred= " + str(int(total)) +
              " Overall Accuracy= " + "{:.4f}".format(total / float(np.sum(cm_test))) +
              " Normalized Accuracy= " + "{:.4f}".format(_sum / float(inps.shape[-1])) +
              " IoU= " + "{:.4f}".format(_sum_iou) +
              " Confusion Matrix= " + np.array_str(cm_test).replace("\n", "")
              )

        sys.stdout.flush()


def train(train_loader, net, criterion, optimizer, epoch, gpu):
    # Setting network for training mode.
    net.train()

    # Average Meter for batch loss.
    train_loss = list()

    # Iterating over batches.
    for i, data in enumerate(train_loader):
        # Obtaining images, labels and paths for batch.
        inps, labs, masks, _, _, _ = data

        # Casting tensors to cuda.
        inps, labs, masks = inps.cuda(gpu), labs.cuda(gpu), masks.cuda(gpu)

        inps.squeeze_(0)
        labs.squeeze_(0)
        masks.squeeze_(0)

        # Casting to cuda variables.
        inps = Variable(inps).cuda(gpu)
        labs = Variable(labs).cuda(gpu)
        masks = Variable(masks).cuda(gpu)

        # Clears the gradients of optimizer.
        optimizer.zero_grad()

        # Forwarding.
        outs = net(inps)
        soft_outs = F.softmax(outs, dim=1)

        # Obtaining predictions.
        prds = soft_outs.data.max(1)[1]

        # Computing loss.
        loss = criterion(outs, labs, masks)

        # Computing backpropagation.
        loss.backward()
        optimizer.step()

        # Appending images for epoch loss calculation.
        prds = prds.squeeze_(1).squeeze_(0).cpu().numpy()

        # Updating loss meter.
        train_loss.append(loss.data.item())

        # Printing.
        if (i + 1) % DISPLAY_STEP == 0:
            # print('[epoch %d], [iter %d / %d], [train loss %.5f]' %
            # (epoch, i + 1, len(train_loader), np.asarray(train_loss).mean()))
            acc, batch_cm_train = calc_accuracy_by_crop(labs, prds, outs.shape[1], None, masks)

            _sum = 0.0
            for k in range(len(batch_cm_train)):
                _sum += (batch_cm_train[k][k] / float(np.sum(batch_cm_train[k]))
                         if np.sum(batch_cm_train[k]) != 0 else 0)

            _sum_iou = (batch_cm_train[1][1] / float(
                np.sum(batch_cm_train[:, 1]) + np.sum(batch_cm_train[1]) - batch_cm_train[1][1])
                        if (np.sum(batch_cm_train[:, 1]) + np.sum(batch_cm_train[1]) - batch_cm_train[1][1]) != 0
                        else 0)

            print("Epoch " + str(epoch) + " -- Iter " + str(i+1) + "/" + str(len(train_loader)) +
                  " -- Time " + str(datetime.datetime.now().time()) +
                  " -- Training Minibatch: Loss= " + "{:.6f}".format(train_loss[-1]) +
                  " Absolut Right Pred= " + str(int(acc)) +
                  " Overall Accuracy= " + "{:.4f}".format(acc / float(np.sum(batch_cm_train))) +
                  " Normalized Accuracy= " + "{:.4f}".format(_sum / float(outs.shape[1])) +
                  " IoU= " + "{:.4f}".format(_sum_iou) +
                  " Confusion Matrix= " + np.array_str(batch_cm_train).replace("\n", "")
                  )

    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description='main')
    # general options
    parser.add_argument('--operation', type=str, required=True,
                        help='Operation [Options: Train | Test]')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to to save outcomes (such as images and trained models) of the algorithm.')
    parser.add_argument('--simulate_dataset', type=str2bool, default=False,
                        help='Used to speed up the development process.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU number.')

    # dataset options
    parser.add_argument('--dataset', type=str, help='Dataset [Options: road_detection].')
    parser.add_argument('--dataset_input_path', type=str, help='Dataset path.')
    parser.add_argument('--dataset_gt_path', type=str, help='Ground truth path.')
    parser.add_argument('--num_classes', type=int, help='Number of classes.')
    # parser.add_argument('--dataset_split_method', type=str, default='train_test',
    #                     help='Split method the dataset [Options: train_test]')

    # model options
    parser.add_argument('--model_name', type=str, default='dilated_grsl_rate8',
                        help='Model to test [Options: dilated_grsl_rate8]')
    parser.add_argument('--model_path', type=str, default=None, help='Model path.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epoch_num', type=int, default=2000, help='Number of epochs')
    parser.add_argument('--ssim', type=str2bool, default=False, help='Use SSIM loss.')

    # dynamic dilated convnet options
    parser.add_argument('--reference_crop_size', type=int, default=25, help='Reference crop size.')
    parser.add_argument('--reference_stride_crop', type=int, default=15, help='Reference crop stride')
    parser.add_argument('--distribution_type', type=str, default='multi_fixed',
                        help='Distribution type [Options: single_fixed, uniform, multi_fixed, multinomial]')
    parser.add_argument('--values', type=str, default=None, help='Values considered in the distribution.')
    parser.add_argument('--update_type', type=str, default='acc', help='Update type [Options: loss, acc]')

    args = parser.parse_args()
    if args.values is not None:
        args.values = [int(i) for i in args.values.split(',')]
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    print(args)

    # Making sure output directory is created.
    check_mkdir(args.output_path)

    # Setting datasets.
    train_set = dataloader_factory('Train', args.dataset, args.dataset_input_path, args.num_classes,
                                   args.output_path, args.model_name, args.reference_crop_size,
                                   args.reference_stride_crop, args.simulate_dataset)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=NUM_WORKERS, shuffle=True)

    test_set = dataloader_factory('Test', args.dataset, args.dataset_input_path, args.num_classes,
                                  args.output_path, args.model_name, args.reference_crop_size,
                                  args.reference_stride_crop, args.simulate_dataset,
                                  mean=train_set.mean, std=train_set.std)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=NUM_WORKERS, shuffle=False)

    # Setting network architecture.
    net = model_factory(args.model_name, train_set.num_channels, train_set.num_classes).cuda(args.gpu)
    # net = SegNet(3, num_classes=list_dataset.num_classes, hidden_classes=hidden).cuda(args['device'])
    print(net)

    # criterion = CrossEntropyLoss2d(weight=None, size_average=False, ignore_index=5).cuda(args['device'])
    criterion = CrossEntropyLoss(weight=torch.DoubleTensor([1.0, 3.0]), size_average=False).cuda(args.gpu)

    # Setting optimizer.
    # optimizer = optim.Adam([
    #     {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
    #      'lr': 2 * args['lr']},
    #     {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
    #      'lr': args['lr'], 'weight_decay': args['weight_decay']}
    # ], betas=(args['momentum'], 0.99))
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
                           betas=(0.9, 0.99))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    curr_epoch = 1
    if args.model_path is not None:
        curr_epoch = int(args.model_path.split('-')[-1])
    best_record = {'epoch': 0, 'lr': 1e-4, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'iou': 0}

    # Iterating over epochs.
    for epoch in range(curr_epoch, args.epoch_num + 1):
        # Training function.
        train(train_loader, net, criterion, optimizer, epoch, args.gpu)

        if epoch % VAL_INTERVAL == 0:
            torch.save(net.state_dict(), os.path.join(args.output_path, 'model_' + str(epoch) + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(args.output_path, 'opt_' + str(epoch) + '.pth'))

            # Computing test.
            test(test_loader, test_set, net, args.output_path, epoch, args.gpu, save_images=False)

        scheduler.step()


if __name__ == "__main__":
    main()
