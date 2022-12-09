import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader, Sampler
from typing import List
from itertools import combinations


def wif(id):
    """
    Used to fix randomization bug for pytorch dataloader + numpy
    Code from https://github.com/pytorch/pytorch/issues/5059
    """
    process_seed = torch.initial_seed()
    # Back out the base_seed so we can use all the bits.
    base_seed = process_seed - id
    ss = np.random.SeedSequence([id, base_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))


class MultitaskDataset():
    """
    Template class for a Multi-task data handler
    """
    def __init__(self) -> None:
        self.trainset: Dataset
        self.testset: Dataset

    def fetch_data_loaders(self, bs, workers=4, shuf=True) -> DataLoader:
        """
        Get the Dataloader for the entire dataset
        Args:
            - shuf       : Shuffle
            - wtd_loss   : Dataloader also has wts along with targets
            - wtd_sampler: Sample data from dataloader with weights
                         according to self.tr_wts
        """
        loaders = []
        for idx, data in enumerate([self.trainset, self.testset]):
            loaders.append(
                DataLoader(
                    data, batch_size=bs, shuffle=(idx==0) and shuf,
                    num_workers=workers, pin_memory=True,
                    worker_init_fn=wif))

        return loaders


class CifarDataset(MultitaskDataset):
    def split_dataset(self, task: List[int], permute: bool):
        """
        Use the "tasks" vector to split dataset
        """
        tr_ind, te_ind = [], []
        tr_lab, te_lab = [], []

        for lab_id, lab in enumerate(task):
            task_tr_ind = np.where(np.isin(self.trainset.targets,
                                           [lab]))[0]
            task_te_ind = np.where(np.isin(self.testset.targets,
                                           [lab]))[0]

            # Get indices and store labels
            tr_ind.append(task_tr_ind)
            te_ind.append(task_te_ind)

            tr_lab.append([lab_id for _ in range(len(task_tr_ind))])
            te_lab.append([lab_id for _ in range(len(task_te_ind))])

        tr_ind, te_ind = np.concatenate(tr_ind), np.concatenate(te_ind)
        tr_lab, te_lab = np.concatenate(tr_lab), np.concatenate(te_lab)

        self.trainset.data = self.trainset.data[tr_ind]
        self.testset.data = self.testset.data[te_ind]

        self.trainset.targets = list(tr_lab)
        self.testset.targets = list(te_lab)

        if permute:
            new_idx = np.random.permutation(len(self.trainset.targets))
            self.trainset.data = self.trainset.data[new_idx]
            self.trainset.targets = [self.trainset.targets[idx] for idx in new_idx]


class Cifar10Dataset(CifarDataset):
    """
    Load CIFAR10 and prepare dataset
    """
    def __init__(self,
                 tasks: List[int],
                 download: bool = True,
                 permute: bool = True) -> None:
        """
        Download dataset and define transforms
        Args:
            - tasks: A List of tasks. Each element in the list is another list
              describing the labels contained in the task
        """
        train_transform, test_transform = self.get_transforms()

        self.trainset = torchvision.datasets.CIFAR10(
            root='./datasets', train=True, download=download,
            transform=train_transform)
        self.testset = torchvision.datasets.CIFAR10(
            root='./datasets', train=False, download=download,
            transform=test_transform)
        self.split_dataset(tasks, permute)

    def get_transforms(self):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

        return train_transform, test_transform


class Cifar100Dataset(CifarDataset):
    """
    Load CIFAR10 and prepare dataset
    """
    def __init__(self,
                 tasks: List[int],
                 download: bool = True,
                 permute: bool = True) -> None:
        """
        Download dataset and define transforms
        Args:
            - tasks: A List of tasks. Each element in the list is another list
              describing the labels contained in the task
        """
        train_transform, test_transform = self.get_transforms()

        self.trainset = torchvision.datasets.CIFAR100(
            root='./datasets', train=True, download=download,
            transform=train_transform)
        self.testset = torchvision.datasets.CIFAR100(
            root='./datasets', train=False, download=download,
            transform=test_transform)
        self.split_dataset(tasks, permute)

    def get_transforms(self):
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

        return train_transform, test_transform

