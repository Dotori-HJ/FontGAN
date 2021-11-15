from __future__ import print_function
from __future__ import absolute_import
import pickle as pickle
import numpy as np
import random
import os
import torch
from utils import pad_seq, bytes_to_file, read_split_image
from utils import shift_and_resize_image, normalize_image, centering_image
from torch.utils.data import DataLoader


def process(img, augment=False):
    img = bytes_to_file(img)
    try:
        img_A, img_B = read_split_image(img)
        if augment:
            # augment the image by:
            # 1) enlarge the image
            # 2) random crop the image back to its original size
            # NOTE: image A and B needs to be in sync as how much
            # to be shifted
            w, h = img_A.shape[0], img_A.shape[1]
            multiplier = random.uniform(1.00, 1.20)
            # add an eps to prevent cropping issue
            nw = int(multiplier * w) + 1
            nh = int(multiplier * h) + 1
            shift_x = int(np.ceil(np.random.uniform(0.01, nw - w)))
            shift_y = int(np.ceil(np.random.uniform(0.01, nh - h)))
            img_A = shift_and_resize_image(img_A, shift_x, shift_y, nw, nh)
            img_B = shift_and_resize_image(img_B, shift_x, shift_y, nw, nh)
        
        img_A = centering_image(img_A, resize_fix=120, pad_value=255)
        img_A = normalize_image(img_A)
        img_A = img_A.reshape(1, len(img_A), len(img_A[0]))
        img_B = centering_image(img_B, resize_fix=120, pad_value=255)
        img_B = normalize_image(img_B)
        img_B = img_B.reshape(1, len(img_B), len(img_B[0]))
        return np.concatenate([img_A, img_B], axis=0)
    finally:
        img.close()
        
def get_batch_iter(examples, batch_size, augment, with_charid=False):
    # the transpose ops requires deterministic
    # batch size, thus comes the padding
    padded = pad_seq(examples, batch_size)

    def batch_iter(with_charid=with_charid):
        for i in range(0, len(padded), batch_size):
            batch = padded[i: i + batch_size]
            labels = [e[0] for e in batch]
            if with_charid:
                charid = [e[1] for e in batch]
                image = [process(e[2]) for e in batch]
                image = np.array(image).astype(np.float32)
                image = torch.from_numpy(image)
                # stack into tensor
                yield [labels, charid, image]
            else:
                image = [process(e[1]) for e in batch]
                image = np.array(image).astype(np.float32)
                image = torch.from_numpy(image)
                # stack into tensor
                yield [labels, image]

    return batch_iter(with_charid=with_charid)


class PickledImageProvider(object):
    def __init__(self, obj_path, verbose):
        self.obj_path = obj_path
        self.verbose = verbose
        self.examples = self.load_pickled_examples()

    def load_pickled_examples(self):
        with open(self.obj_path, "rb") as of:
            examples = list()
            while True:
                try:
                    e = pickle.load(of)
                    examples.append(e)
                except EOFError:
                    break
                except Exception:
                    pass
            if self.verbose:
                print("unpickled total %d examples" % len(examples))
            return examples


class DataProvider(object):
    def __init__(self, data_dir, obj_name="train.obj", filter_by_font=None, filter_by_charid=None, verbose=True, augment=True):
        self.data_dir = data_dir
        self.filter_by_font = filter_by_font
        self.filter_by_charid = filter_by_charid
        self.data_path = os.path.join(self.data_dir, obj_name)
        self.data = PickledImageProvider(self.data_path, verbose)
        self.augment = augment
        if self.filter_by_font:
            if verbose:
                print("filter by label ->", filter_by_font)
            self.data.examples = [e for e in self.data.examples if e[0] in self.filter_by_font]
        if self.filter_by_charid:
            if verbose:
                print("filter by char ->", filter_by_charid)
            self.data.examples = [e for e in self.data.examples if e[1] in filter_by_charid]
        if verbose:
            print(f"{obj_name.split('.')[0]} examples -> %d" % (len(self.data.examples)))

    def __getitem__(self, i):
        font_id = self.data.examples[i][0]
        img = process(self.data.examples[i][1], augment=self.augment)

        return font_id, img
    
    def __len__(self):
        return len(self.data.examples)

def save_fixed_sample(sample_size, img_size, data_dir, save_dir, \
                      val=False, verbose=True, with_charid=False, resize_fix=90):
    if not val:
        provider = DataProvider(data_dir, obj_name='train.obj', augment=False)
        
        source_name = 't_fixed_source.pkl'
        target_name = 't_fixed_target.pkl'
        label_name = 't_fixed_label.pkl'

    else:
        provider = DataProvider(data_dir, obj_name='val.obj', augment=False)
        
        source_name = 'fixed_source.pkl'
        target_name = 'fixed_target.pkl'
        label_name = 'fixed_label.pkl'

    loader = DataLoader(provider, batch_size=sample_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    font_ids, batch_images = batch

    fixed_batch = batch_images
    fixed_source = fixed_batch[:, [1], :, :]
    fixed_target = fixed_batch[:, [0], :, :]
    # fixed_target = (fixed_target + 1) * 127.5
    fixed_label = np.array(font_ids)
    torch.save(fixed_source, os.path.join(save_dir, source_name))
    torch.save(fixed_target, os.path.join(save_dir, target_name))
    torch.save(fixed_label, os.path.join(save_dir, label_name))

data_dir = 'dataset'
fixed_dir = 'fixed_set'
if __name__ == '__main__':
    os.makedirs(fixed_dir, exist_ok=True)
    save_fixed_sample(sample_size=64, img_size=128, data_dir=data_dir, save_dir=fixed_dir, val=True)
    save_fixed_sample(sample_size=16, img_size=128, data_dir=data_dir, save_dir=fixed_dir, val=False)