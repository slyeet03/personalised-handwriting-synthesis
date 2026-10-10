import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as TorchDataset

import vocab


class Dataset(TorchDataset):
    def __init__(self, path, max_len=None):
        self.data = np.load(path, allow_pickle=True)
        print("files", self.data.files)

        self.dict = {}

        for name in self.data.files:
            self.dict[name] = self.data[name]

        self.sorted_alphabets = vocab.build_vocab(self.dict['alphabet'])
        self.char_to_idx = vocab.build_char_to_idx(self.sorted_alphabets)

        strokes = self.dict['strokes']
        texts = self.dict['texts']

        valid_chars = set(self.char_to_idx) | {' '}    # ' ' gets converted to '' inside encode_text

        def text_ok(text):
            return all(c in valid_chars for c in str(text))

        self.indices = [i for i in range(len(strokes))
                        if (max_len is None or len(strokes[i]) <= max_len)
                        and text_ok(texts[i])]        

        print(f"kept {len(self.indices)} / {len(strokes)} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        tensor_stroke = torch.tensor(self.dict['strokes'][i])
        tensor_text = torch.tensor(vocab.encode_text(self.dict['texts'][i], self.char_to_idx))
        return (tensor_stroke, tensor_text)
 

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
    text_mask = torch.arange(padded_texts.shape[1]).unsqueeze(0) < text_lengths.unsqueeze(1)

    return padded_strokes, padded_texts, mask, text_mask



