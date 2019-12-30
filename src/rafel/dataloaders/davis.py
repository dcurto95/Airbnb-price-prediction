import os

import numpy as np
from PIL import Image

import torch
from torch.utils import data


class DAVISCurrentAndFirst(data.Dataset):

    """
    Loads pairs of frame_0, frame_t
    Resizes frames so they all have same dimensions
    Useful for training as shuffles all frames
    Returns frames FloatTensor(n_frames=2, ch=3, h, w)
    Returns masks LongTensor(n_frames=2, ch=1, h, w)
    Returns info dict
    """

    def __init__(self, davis_root_dir, image_set, resize=(480, 864), multi_object=True, resolution='480p', transform=None):
        assert resolution == '480p', 'Other resolutions than 480p not implemented yet in dataloader'
        self.MO = multi_object
        _year = 2017 if self.MO else 2016
        self.image_dir = os.path.join(davis_root_dir, 'JPEGImages', resolution)
        self.mask_dir = os.path.join(davis_root_dir, 'Annotations', resolution)
        _image_set_file = os.path.join(davis_root_dir, 'ImageSets', str(_year), image_set + '.txt')
        self.transform = transform
        self.resize = resize

        self.mydict = {}            # Mapping to retrieve frames

        _global_offset = 0
        with open(os.path.join(_image_set_file), 'r') as lines:
            for line in lines:
                _video = line.strip()
                _n_frames = len(os.listdir(os.path.join(self.image_dir, _video)))

                for i in range(1, _n_frames):  # Avoid frame_0 as it will be loaded as frame_0 or frame_t-1
                    self.mydict[_global_offset + i - 1] = (_video, i)
                _global_offset += _n_frames - 1

    def __len__(self):
        return len(self.mydict)

    def __getitem__(self, index):
        video, frame = self.mydict[index]
        assert len(os.listdir(os.path.join(self.mask_dir, video))) > 1, 'No ground truth for training dataset'

        frames = torch.FloatTensor(2, 3, self.resize[0], self.resize[1])  # (ref_frame frame_t, RGB, H, W)
        masks = torch.LongTensor(2, 1, self.resize[0], self.resize[1])    # (ref_frame frame_t, Palette, H, W)

        frame_ref_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(0))
        frame_t_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(frame))
        frame_ref_img = Image.open(frame_ref_file).convert('RGB')
        frame_t_img = Image.open(frame_t_file).convert('RGB')

        mask_ref_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(0))
        mask_t_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(frame))
        mask_ref_img = Image.open(mask_ref_file).convert('P')
        mask_t_img = Image.open(mask_t_file).convert('P')

        if self.transform is not None:
            frame_ref_img = self.transform(frame_ref_img)
            frame_t_img = self.transform(frame_t_img)
            mask_ref_img = self.transform(mask_ref_img)
            mask_t_img = self.transform(mask_t_img)

        if frame_ref_img.size != (self.resize[1], self.resize[0]):
            frame_ref_img = frame_ref_img.resize((self.resize[1], self.resize[0]), Image.LANCZOS)  # AntiAliasing Filter
            frame_t_img = frame_t_img.resize((self.resize[1], self.resize[0]), Image.LANCZOS)
            mask_ref_img = mask_ref_img.resize((self.resize[1], self.resize[0]))  # NN Filter as we are in palette mode
            mask_t_img = mask_t_img.resize((self.resize[1], self.resize[0]))

        frame_ref_np = np.array(frame_ref_img).transpose((2, 0, 1))
        frame_t_np = np.array(frame_t_img).transpose((2, 0, 1))
        frame_ref_np = frame_ref_np / 255.  # RGB rescaled [0, 1]
        frame_t_np = frame_t_np / 255.

        frames[0] = torch.from_numpy(frame_ref_np)
        frames[1] = torch.from_numpy(frame_t_np)
        masks[0] = torch.from_numpy(np.array(mask_ref_img))
        masks[1] = torch.from_numpy(np.array(mask_t_img))

        n_objects = np.max(np.array(mask_ref_img)) + 1

        if not self.MO:
            masks = (masks != 0)
            n_objects = 2

        # Correct bug in dataset (appears new object to segment in frame != 0 with index 255)
        if video == 'tennis':
            masks = (masks != 255).long()*masks

        info = {
            'name': video,
            'frame': frame,
            'n_objects': n_objects
        }

        return frames, masks, info


class DAVISCurrentFirstAndPrevious(data.Dataset):

    """
    Loads triplets of frame_0, frame_t-1, frame_t;
    Resizes frames so they all have same dimensions
    Useful for training as shuffles all frames
    Returns frames FloatTensor(n_frames=3, ch=3, h, w)
    Returns masks LongTensor(n_frames=3, ch=1, h, w)
    Returns info dict
    """

    def __init__(self, davis_root_dir, image_set, resize=(480, 864), multi_object=True, resolution='480p', transform=None):
        assert resolution == '480p', 'Other resolutions than 480p not implemented yet in dataloader'
        self.MO = multi_object
        _year = 2017 if self.MO else 2016
        self.image_dir = os.path.join(davis_root_dir, 'JPEGImages', resolution)
        self.mask_dir = os.path.join(davis_root_dir, 'Annotations', resolution)
        _image_set_file = os.path.join(davis_root_dir, 'ImageSets', str(_year), image_set + '.txt')
        self.transform = transform
        self.resize = resize

        self.mydict = {}            # Mapping to retrieve frames

        _global_offset = 0
        with open(os.path.join(_image_set_file), 'r') as lines:
            for line in lines:
                _video = line.strip()
                _n_frames = len(os.listdir(os.path.join(self.image_dir, _video)))

                for i in range(1, _n_frames):  # Avoid frame_0 as it will be loaded as frame_0 or frame_t-1
                    self.mydict[_global_offset + i - 1] = (_video, i)
                _global_offset += _n_frames - 1

    def __len__(self):
        return len(self.mydict)

    def __getitem__(self, index):
        video, frame = self.mydict[index]
        assert len(os.listdir(os.path.join(self.mask_dir, video))) > 1, 'No ground truth for training dataset'

        frames = torch.FloatTensor(3, 3, self.resize[0], self.resize[1])  # (frame_0 frame_t-1 frame_t, RGB, H, W)
        masks = torch.LongTensor(3, 1, self.resize[0], self.resize[1])    # (frame_0 frame_t-1 frame_t, Palette, H, W)

        frame_0_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(0))
        frame_prev_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(frame - 1))
        frame_t_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(frame))
        frame_0_img = Image.open(frame_0_file).convert('RGB')
        frame_prev_img = Image.open(frame_prev_file).convert('RGB')
        frame_t_img = Image.open(frame_t_file).convert('RGB')

        mask_0_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(0))
        mask_prev_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(frame - 1))
        mask_t_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(frame))
        mask_0_img = Image.open(mask_0_file).convert('P')
        mask_prev_img = Image.open(mask_prev_file).convert('P')
        mask_t_img = Image.open(mask_t_file).convert('P')

        if self.transform is not None:
            frame_0_img = self.transform(frame_0_img)
            frame_prev_img = self.transform(frame_prev_img)
            frame_t_img = self.transform(frame_t_img)
            mask_0_img = self.transform(mask_0_img)
            mask_prev_img = self.transform(mask_prev_img)
            mask_t_img = self.transform(mask_t_img)

        if frame_0_img.size != (self.resize[1], self.resize[0]):
            frame_0_img = frame_0_img.resize((self.resize[1], self.resize[0]), Image.LANCZOS)       # AntiAliasing Filter
            frame_prev_img = frame_prev_img.resize((self.resize[1], self.resize[0]), Image.LANCZOS)
            frame_t_img = frame_t_img.resize((self.resize[1], self.resize[0]), Image.LANCZOS)
            mask_0_img = mask_0_img.resize((self.resize[1], self.resize[0]))                        # NN Filter as we are in palette mode
            mask_prev_img = mask_prev_img.resize((self.resize[1], self.resize[0]))
            mask_t_img = mask_t_img.resize((self.resize[1], self.resize[0]))

        frame_0_np = np.array(frame_0_img).transpose((2, 0, 1))
        frame_prev_np = np.array(frame_prev_img).transpose((2, 0, 1))
        frame_t_np = np.array(frame_t_img).transpose((2, 0, 1))
        frame_0_np = frame_0_np / 255.  # RGB rescaled [0, 1]
        frame_prev_np = frame_prev_np / 255.
        frame_t_np = frame_t_np / 255.

        frames[0] = torch.from_numpy(frame_0_np)
        frames[1] = torch.from_numpy(frame_prev_np)
        frames[2] = torch.from_numpy(frame_t_np)
        masks[0] = torch.from_numpy(np.array(mask_0_img))
        masks[1] = torch.from_numpy(np.array(mask_prev_img))
        masks[2] = torch.from_numpy(np.array(mask_t_img))

        n_objects = np.max(np.array(mask_0_img)) + 1

        if not self.MO:
            masks = (masks != 0).long()
            n_objects = 2

        # Correct bug in dataset (appears new object to segment in frame != 0 with index 255)
        if video == 'tennis':
            masks = (masks != 255).long()*masks

        info = {
            'name': video,
            'frame': frame,
            'n_objects': n_objects
        }

        return frames, masks, info


class DAVISAllSequence(data.Dataset):

    """
        Loads one entire sequence of frames
        Useful for validating and testing
        Returns frames FloatTensor(n_frames, ch=3, h, w)
        Returns masks LongTensor(n_frames, ch=1, h, w)
        Returns info dict
    """

    def __init__(self, davis_root_dir, image_set, resize=(480, 864), multi_object=True, resolution='480p', transform=None):
        assert resolution == '480p', 'Other resolutions than 480p not implemented yet in dataloader'
        self.MO = multi_object
        _year = 2017 if self.MO else 2016
        self.image_dir = os.path.join(davis_root_dir, 'JPEGImages', resolution)
        self.mask_dir = os.path.join(davis_root_dir, 'Annotations', resolution)
        _image_set_file = os.path.join(davis_root_dir, 'ImageSets', str(_year), image_set + '.txt')
        self.transform = transform
        self.resize = resize

        self.videos = []

        with open(_image_set_file, 'r') as lines:
            for line in lines:
                _video = line.strip()
                self.videos.append(_video)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video = self.videos[index]
        n_frames = len(os.listdir(os.path.join(self.image_dir, video)))
        has_gt = len(os.listdir(os.path.join(self.mask_dir, video))) > 1
        n_objects, palette, original_shape, final_shape = None, None, None, None

        if self.resize is None:
            _size = Image.open(os.path.join(self.mask_dir, video, '00000.png')).size
            self.resize = _size[1], _size[0]

        frames = torch.FloatTensor(n_frames, 3, self.resize[0], self.resize[1])
        if has_gt:
            masks = torch.LongTensor(n_frames, 1, self.resize[0], self.resize[1])
        else:
            masks = torch.LongTensor(1, 1, self.resize[0], self.resize[1])

        for frame in range(n_frames):
            # Frame loader
            image_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(frame))
            img = Image.open(image_file).convert('RGB')

            if self.transform is not None:
                img = self.transform(img)

            if img.size != (self.resize[1], self.resize[0]):
                img = img.resize((self.resize[1], self.resize[0]), Image.LANCZOS)  # AntiAliasing Filter

            img_np = np.array(img).transpose((2, 0, 1))
            img_np = img_np / 255.  # RGB rescaled [0, 1]

            frames[frame] = torch.from_numpy(img_np)

            # Mask loader
            if frame == 0 or has_gt:
                mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(frame))
                mask_img = Image.open(mask_file).convert('P')

                if mask_img.size != (self.resize[1], self.resize[0]):
                    mask_img_resized = mask_img.resize((self.resize[1], self.resize[0]))  # NN Filter
                else:
                    mask_img_resized = mask_img

                mask_np = np.array(mask_img_resized)

                # Loading Palette || Maximum number of objects always in frame 0
                if frame == 0:
                    palette = mask_img.getpalette()
                    n_objects = np.max(mask_np) + 1
                    original_shape = mask_img.size
                    # final_shape = np.shape(img_np)

                mask = torch.from_numpy(mask_np)

                if not self.MO:
                    mask = (mask != 0).long()
                    n_objects = 2

                masks[frame][0] = mask

        # Correct bug in dataset (appears new object to segment in frame != 0 with index 255)
        if video == 'tennis':
            masks = (masks != 255).long()*masks

        info = {
            'name': video,
            'n_frames': n_frames,
            'n_objects': n_objects,
            # 'final_shape': final_shape,
            'original_shape': original_shape,
            'has_gt': has_gt,
            'palette': torch.ByteTensor(palette)
        }

        assert torch.max(masks).item() + 1 == n_objects, 'Error preprocessing data'

        return frames, masks, info
