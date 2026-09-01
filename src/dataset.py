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

    padded_strokes = pad_sequence(strokes, batch_first=True)
    padded_texts = pad_sequence(texts, batch_first=True)

    return padded_strokes,padded_texts



if __name__ == '__main__':
    dataset = Dataset('../dataset/deepwriting_training.npz')
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    
    batch_strokes, batch_texts = next(iter(loader))
    print("Batch strokes shape:", batch_strokes.shape)
    print("Batch texts shape:", batch_texts.shape)
