from __future__ import print_function

import sys
import os
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn
import wandb
from data import Cifar10Dataset, Cifar100Dataset


from main_ce import set_loader
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, set_seed, save_model
from networks.resnet_big import SupConResNet, SupCEResNet, LinearClassifier


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=1,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='30,38,45',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100'], help='dataset')
    parser.add_argument('--task_labels', type=str, default=None,
                    help='labels to include in task, should be comma separated list e.g. 0,1')
                    

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--seed', type = int, default=0, help = "set random seeds")

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR', 'SupCE'], help='choose method')
    parser.add_argument('--pretrained_tag', type=str, default='',
                        help='pre-trained model tag')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'

    
    opt.save_folder = f'./save/linear_tuning/{opt.method}/{opt.dataset}/{opt.pretrained_tag}_seed_{opt.seed}'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 5
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate
    
    if (opt.task_labels is not None):
        task_labs = opt.task_labels.split(',')
        opt.task_labels = [int(lab) for lab in task_labs]
        opt.n_cls = len(task_labs)
        opt.task_name = '_'.join(task_labs)
    elif (opt.task_labels is None and opt.dataset == 'cifar10'):
        opt.task_labels = [i for i in range(10)]
        opt.n_cls = 10
        opt.task_name = "all"
    elif (opt.task_labels is None and opt.dataset == 'cifar100'):
        opt.task_labels = [i for i in range(100)]
        opt.n_cls = 100 
        opt.task_name = "all"  
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    opt.save_folder = f'{opt.save_folder}/task_{opt.task_name}'
    
    opt.wandb_folder = opt.save_folder
    if not os.path.isdir(opt.wandb_folder):
        os.makedirs(opt.wandb_folder)

    opt.save_folder = os.path.join(opt.save_folder, "models")
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def fetch_dataset(opt):
    if opt.dataset == "cifar10":
        dataset = Cifar10Dataset(opt.task_labels)
    elif opt.dataset == "cifar100":
        dataset = Cifar100Dataset(opt.task_labels)
    else:
        raise NotImplementedError
    return dataset

def set_model(opt):
    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if (opt.method == 'SupCon' or opt.method == 'SimCLR'):
        model = SupConResNet(name=opt.model, feat_dim = ckpt["model"]["head.2.bias"].shape[0])
    elif (opt.method == 'SupCE'):
        model = SupCEResNet(name=opt.model, num_classes= ckpt["model"]["fc.bias"].shape[0])
    
    criterion = torch.nn.CrossEntropyLoss()
    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)


    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), labels.shape[0])
        acc1 = accuracy(output, labels, topk=(1,))
        top1.update(acc1[0].item(), labels.shape[0])
        

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print(f'Train: [{epoch}][{idx + 1}/{len(train_loader)}]\t loss {losses.val:.3f} ({losses.avg:.3f}) Acc@1 {top1.val:.3f} ({top1.avg:.3f})')
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), labels.shape[0])
            acc1 = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0].item(), labels.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print(f'Test: [{idx}/{len(val_loader)}]\t Loss {losses.val:.4f} ({losses.avg:.4f})\t Acc@1 {top1.val:.3f} ({top1.avg:.3f})')
                sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()
    
    # set random seed
    set_seed(opt.seed)

    # build data loader
    datasets = fetch_dataset(opt)
    loaders = datasets.fetch_data_loaders(opt.batch_size, opt.num_workers)
    train_loader, val_loader = loaders[0], loaders[1]

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # wandb
    wandb.init(project=f"fine_tuning_{opt.method}", dir = opt.wandb_folder)
    wandb.run.name = f"{opt.task_labels}_seed_{opt.seed}"
    wandb.config.update(opt)

    
    val_loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
    if val_acc > best_acc:
        best_acc = val_acc

    wandb.log({"epoch": 0, 
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "val_loss": val_loss,
                    "val_acc": val_acc})

    save_file = os.path.join(
                opt.save_folder, 'classifier_0.pth')
    save_model(classifier, optimizer, opt, 0, save_file)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        train_loss, train_acc = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, train_acc))

        # eval for one epoch
        val_loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
        if val_acc > best_acc:
            best_acc = val_acc


        wandb.log({"epoch": epoch, 
                       "learning_rate": optimizer.param_groups[0]['lr'],
                       "train_loss": train_loss,
                       "val_loss": val_loss,
                       "train_acc": train_acc,
                       "val_acc": val_acc})

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'classifier_{epoch}.pth'.format(epoch=epoch))
            save_model(classifier, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, f'classifier_{opt.epochs}.pth')
    save_model(classifier, optimizer, opt, opt.epochs, save_file)

    print('best accuracy: {:.2f}'.format(best_acc))
    wandb.finish()


if __name__ == '__main__':
    main()
