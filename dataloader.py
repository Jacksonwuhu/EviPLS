import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset


categories = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
              'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
              'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
              'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
              'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
              'Clapping']


def ids_to_multinomial(ids):
    """ label encoding

    Returns:
      1d array, multimonial representation, e.g. [1,0,1,0,0,...]
    """
    id_to_idx = {id: index for index, id in enumerate(categories)}

    y = np.zeros(len(categories))
    for id in ids:
        index = id_to_idx[id]
        y[index] = 1
    return y



class LLP_dataset(Dataset):

    def __init__(self, label, audio_dir, video_dir, st_dir, transform, v_pseudo_data_dir, a_pseudo_data_dir, mode=None):
        self.mode = mode
        self.df = pd.read_csv(label, header=0, sep='\t')
        self.filenames = self.df["filename"]
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.st_dir = st_dir
        self.num_of_data = len(self.filenames)
        self.transform = transform

        self.v_pseudo_data_dir = v_pseudo_data_dir
        self.a_pseudo_data_dir = a_pseudo_data_dir


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        row = self.df.iloc[idx, :]
        # Positional access must use ``iloc``.  ``Series[0]``/``Series[-1]``
        # changes meaning across pandas versions and fails with pandas 3.
        name = str(row.iloc[0])[:11]
        audio = np.load(os.path.join(self.audio_dir, name + '.npy'))
        video_s = np.load(os.path.join(self.video_dir, name + '.npy'))
        video_st = np.load(os.path.join(self.st_dir, name + '.npy'))
        ids = str(row.iloc[-1]).split(',')
        label = ids_to_multinomial(ids)


        audio_pseudo_labels = np.load(os.path.join(self.a_pseudo_data_dir, name + '.npy'))
        visual_pseudo_labels = np.load(os.path.join(self.v_pseudo_data_dir, name + '.npy'))

        Pa = np.sum(audio_pseudo_labels, axis=0)
        np.clip(Pa, 0, 1, out=Pa)
        Pv = np.sum(visual_pseudo_labels, axis=0)
        np.clip(Pv, 0, 1, out=Pv)

        sample = {
            'audio': audio,
            'video_s': video_s,
            'video_st': video_st,
            'label': label,
            'audio_pseudo_labels': audio_pseudo_labels,
            'visual_pseudo_labels': visual_pseudo_labels,
            'Pa': Pa,
            'Pv': Pv,
            # video_name is the 11-character feature key used by the
            # legacy loader. segment_name preserves the full
            # video_start_end identifier for temporal-neighbor experiments.
            'video_name': name,
            'segment_name': str(row.iloc[0]),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor:
    def __call__(self, sample):
        tensor = dict()
        for key in sample:
            if isinstance(sample[key], np.ndarray):
                tensor[key] = torch.from_numpy(sample[key])
            else:
                tensor[key] = sample[key]
        return tensor
