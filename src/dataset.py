import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

import vocab


class Dataset(TorchDataset):
    def __init__(self,path):
        self.data = np.load(path, allow_pickle=True)
        print("files", self.data.files)

        self.dict = {}

        for name in self.data.files:
            self.dict[name] = self.data[name]

        self.sorted_alphabets = vocab.build_vocab(self.dict['alphabet'])
        self.char_to_idx = vocab.build_char_to_idx(self.sorted_alphabets)

    def __len__(self):
        return int(self.dict['strokes'].shape[0])

    def __getitem__(self,idx):
        tensor_stroke = torch.tensor(self.dict['strokes'][idx])
        tensor_text = torch.tensor(vocab.encode_text(self.dict['texts'][idx],self.char_to_idx))

        return (tensor_stroke,tensor_text)
         

# adding padding to have each batch the same dimensions
def collate_fn(batch):
    strokes, texts = zip(*batch)
    
    # true length before padding
    stroke_lengths = torch.tensor([len(s) for s in strokes])
    text_lengths = torch.tensor([len(t) for t in texts])

    padded_strokes = pad_sequence(strokes, batch_first=True)
    padded_texts = pad_sequence(texts, batch_first=True)
    
    max_len = padded_strokes.shape[1]

    # checking whether the real length is less than paddded length or not, if it is then it means it is a padding so it it becomes 0 and true length becomes 1 making a mask
    mask = torch.arange(max_len).unsqueeze(0) < stroke_lengths.unsqueeze(1)
    text_mask = torch.arange(padded_texts.shape[1])..unsqueeze(0) < text_lengths.unsqueeze(1)

    return padded_strokes, padded_texts, mask, text_mask



