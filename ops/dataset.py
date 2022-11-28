# Code for TSM adapted from the original TSM repo:
# https://github.com/mit-han-lab/temporal-shift-module

import os
import os.path
import numpy as np
from numpy.random import randint
import torch
from torch.utils import data
from tqdm import tqdm
from PIL import Image

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def start_frames(self):
    	return int(self._data[1])

    @property
    def num_frames(self):
        return int(self._data[2])

    @property
    def label(self):
        return int(self._data[3])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='frame_{:010d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, 
                 dense_sample=False, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        if self.modality == 'RGBDiff':
            # Diff needs one more image to calculate diff
            self.new_length += 1

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            prefix = directory.split('/')[-1]
            list_dir = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 10500, 11000, 11500, 12000, 12500, 13000, 13500, 14000, 14500, 15000, 15500, 16000, 16500, 17000, 17500, 18000, 18500, 19000, 19500, 20000, 20500, 21000, 21500, 22000, 22500, 23000, 23500, 24000, 24500, 25000, 25500, 26000, 26500, 27000, 27500, 28000, 28500, 29000, 29500, 30000, 30500, 31000, 31500, 32000, 32500, 33000, 33500, 34000, 34500, 35000, 35500, 36000, 36500, 37000, 37500, 38000, 38500, 39000, 39500, 40000, 40500, 41000, 41500, 42000, 42500, 43000, 43500, 44000, 44500, 45000, 45500, 46000, 46500, 47000, 47500, 48000, 48500, 49000, 49500, 50000, 50500, 51000, 51500, 52000, 52500, 53000, 53500, 54000, 54500, 55000, 55500, 56000, 56500, 57000, 57500, 58000, 58500, 59000, 59500, 60000, 60500, 61000, 61500, 62000, 62500, 63000, 63500, 64000, 64500, 65000]
            lower, upper = self.get_range(idx, list_dir)
            directory += '/{}-{}/'.format(lower, upper-1)
            # print(os.path.join(directory, prefix + '_' + self.image_tmpl.format(idx)))
            result = Image.open(os.path.join(directory, prefix + '_' + self.image_tmpl.format(idx))).convert('RGB')
            return [result]
            #return [Image.new('RGB', (456, 256), (73, 109, 137))]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, directory.split('/')[-1] + '_' + self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, directory.split('/')[-1] + '_' + self.image_tmpl.format('y', idx))).convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        for x in tmp:
            x[0] = self.root_path + x[0]
        self.video_list = [VideoRecord(x) for x in tmp]


    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        print("num_frames: ", record.num_frames)
        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        
        return offsets + record.start_frames

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))

        return offsets + record.start_frames

    def _get_test_indices(self, record):
        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        return offsets + record.start_frames

    def __getitem__(self, index):
        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)

    def get_range(self, num, lst):
        l = len(lst)

        for i in range(1, l):
            if lst[i - 1] <= num < lst[i]:
                return lst[i-1], lst[i]


class PhonePackagingDataset(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='{:010d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, 
                 dense_sample=False, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB':
            if not os.path.isfile(os.path.join(directory,self.image_tmpl.format(idx))): 
                with open("/content/drive/MyDrive/Action Recognition/TSM MODEL + ASSEMBLY101 DATASET/TSM-action-recognition/log/TSM_PhonePackaging_RGB_resnet50_shift8_blockres_avg_segment8_e50/error.txt", 'a') as f:
                    f.write('Not found: ' + str(os.path.join(directory,self.image_tmpl.format(idx))) + '\n')
                    return []
            result = Image.open(os.path.join(directory,self.image_tmpl.format(idx))).convert('RGB')
            return [result]

    def _parse_list(self):
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        for x in tmp:
            x[0] = self.root_path + x[0]
        self.video_list = [VideoRecord(x) for x in tmp]


    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        
        return offsets + record.start_frames

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))

        return offsets + record.start_frames

    def _get_test_indices(self, record):
        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        return offsets + record.start_frames

    def __getitem__(self, index):
        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                if seg_imgs == []: 
                    seg_imgs = [images[-1]]
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)

    def get_range(self, num, lst):
        l = len(lst)

        for i in range(1, l):
            if lst[i - 1] <= num < lst[i]:
                return lst[i-1], lst[i]


class HWDataset(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='{:d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, 
                 dense_sample=False, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB':
            result = Image.open(os.path.join(directory,self.image_tmpl.format(idx))).convert('RGB')
            return [result]

    def _parse_list(self):
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        for x in tmp:
            x[0] = self.root_path + x[0]
            x[2] = 20 if int(x[2]) > 20 else x[2]
        self.video_list = [VideoRecord(x) for x in tmp]


    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        
        return offsets + record.start_frames

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))

        return offsets + record.start_frames

    def _get_test_indices(self, record):
        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        return offsets + record.start_frames

    def __getitem__(self, index):
        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)

    def get_range(self, num, lst):
        l = len(lst)

        for i in range(1, l):
            if lst[i - 1] <= num < lst[i]:
                return lst[i-1], lst[i]