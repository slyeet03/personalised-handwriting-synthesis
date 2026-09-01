import numpy as np
import torch
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
         



if __name__ == '__main__':
    dataset = Dataset('../dataset/deepwriting_training.npz')
    print("Dataset length:", len(dataset))

    stroke, text = dataset[0]
    print("Stroke tensor shape:", stroke.shape, "| dtype:", stroke.dtype)
    print("Text tensor shape:", text.shape, "| dtype:", text.dtype)
    print("Text tensor values:", text)
