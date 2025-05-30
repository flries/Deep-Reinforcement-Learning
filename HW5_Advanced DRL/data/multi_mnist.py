from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import numpy as np
import torch
import codecs
from torchvision.datasets import MNIST
from torchvision import transforms


class MultiMNIST(data.Dataset):
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    multi_training_file = 'multi_training.pt'
    multi_test_file = 'multi_test.pt'

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 multi=False):

        self._root = os.path.expanduser(root)
        self._transform = transform
        self._target_transform = target_transform
        self._train, self._multi = train, multi

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        if not self._check_multi_exists():
            raise RuntimeError('Multi Task extension not found. You can use download=True to download it')

        if multi:
            if self._train:
                self.train_data, self.train_labels_l, self.train_labels_r = torch.load(
                    os.path.join(self._root, self.processed_folder, self.multi_training_file))
            else:
                self.test_data, self.test_labels_l, self.test_labels_r = torch.load(
                    os.path.join(self._root, self.processed_folder, self.multi_test_file))
        else:
            if self._train:
                self.train_data, self.train_labels = torch.load(
                    os.path.join(self._root, self.processed_folder, self.training_file))
            else:
                self.test_data, self.test_labels = torch.load(
                    os.path.join(self._root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        if self._multi:
            if self._train:
                img, target_l, target_r = self.train_data[index], self.train_labels_l[index], self.train_labels_r[index]
            else:
                img, target_l, target_r = self.test_data[index], self.test_labels_l[index], self.test_labels_r[index]
        else:
            if self._train:
                img, target = self.train_data[index], self.train_labels[index]
            else:
                img, target = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img.numpy().astype(np.uint8), mode='L')
        if self._transform is not None:
            img = self._transform(img)

        if self._target_transform is not None:
            target = self._target_transform(target)

        if self._multi:
            return img, target_l, target_r
        else:
            return img, target

    def __len__(self):
        if self._train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        train_path = os.path.join(self._root, self.processed_folder, self.training_file)
        test_path = os.path.join(self._root, self.processed_folder, self.test_file)
        return os.path.exists(train_path) and os.path.exists(test_path)

    def _check_multi_exists(self):
        multi_train_path = os.path.join(self._root, self.processed_folder, self.multi_training_file)
        multi_test_path = os.path.join(self._root, self.processed_folder, self.multi_test_file)
        return os.path.exists(multi_train_path) and os.path.exists(multi_test_path)

    def download(self):
        if self._check_exists() and self._check_multi_exists():
            return

        print("Downloading and processing MNIST dataset using torchvision...")

        transform = transforms.ToTensor()
        train_dataset = MNIST(root=self._root, train=True, download=True, transform=transform)
        test_dataset = MNIST(root=self._root, train=False, download=True, transform=transform)

        train_data = train_dataset.data
        train_labels = train_dataset.targets
        test_data = test_dataset.data
        test_labels = test_dataset.targets

        os.makedirs(os.path.join(self._root, self.processed_folder), exist_ok=True)

        torch.save((train_data, train_labels), os.path.join(self._root, self.processed_folder, self.training_file))
        torch.save((test_data, test_labels), os.path.join(self._root, self.processed_folder, self.test_file))

        extension = np.random.permutation(len(train_data))
        multi_train_data, multi_train_labels_l, multi_train_labels_r = self.create_multi_mnist(train_data, train_labels, extension)

        extension = np.random.permutation(len(test_data))
        multi_test_data, multi_test_labels_l, multi_test_labels_r = self.create_multi_mnist(test_data, test_labels, extension)

        torch.save((multi_train_data, multi_train_labels_l, multi_train_labels_r),
                   os.path.join(self._root, self.processed_folder, self.multi_training_file))
        torch.save((multi_test_data, multi_test_labels_l, multi_test_labels_r),
                   os.path.join(self._root, self.processed_folder, self.multi_test_file))

        print("Download and processing complete.")

    def create_multi_mnist(self, data, labels, extension):
        length = len(data)
        multi_data = np.zeros((length, 28, 28), dtype=np.uint8)
        multi_labels_l = np.zeros(length, dtype=np.long)
        multi_labels_r = np.zeros(length, dtype=np.long)

        for i in range(length):
            left = data[i].numpy()
            right = data[extension[i]].numpy()

            new_im = np.zeros((36, 36))
            new_im[0:28, 0:28] = left
            new_im[6:34, 6:34] = right
            new_im[6:28, 6:28] = np.maximum(left[6:28, 6:28], right[0:22, 0:22])
            new_im_resized = np.array(Image.fromarray(new_im).resize((28, 28)))

            multi_data[i] = new_im_resized
            multi_labels_l[i] = labels[i]
            multi_labels_r[i] = labels[extension[i]]

        return torch.from_numpy(multi_data), torch.from_numpy(multi_labels_l), torch.from_numpy(multi_labels_r)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self._train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self._root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self._transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp, self._target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
