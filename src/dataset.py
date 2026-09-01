import numpy as np

import vocab


class Dataset:
    def __init__(self):
        self.train_path = '../dataset/deepwriting_training.npz'
        self.validation_path = '../dataset/deepwriting_validation.npz'
        
        self.train_data = np.load(self.train_path, allow_pickle=True)
        self.validation_data = np.load(self.validation_path, allow_pickle=True)

        print("Train files", self.train_data.files)
        print("Validation files", self.validation_data.files)

        self.train_dict = {}
        self.val_dict = {}

        for name in self.train_data.files:
            self.train_dict[name] = self.train_data[name]

        for name in self.validation_data.files:
            self.val_dict[name] = self.validation_data[name]

        self.train_sorted_alphabets = vocab.build_vocab(self.train_dict['alphabet'])
        self.char_to_idx = vocab.build_char_to_idx(self.train_sorted_alphabets)
        self.val_sorted_alphabets = vocab.build_vocab(self.val_dict['alphabet'])
        self.char_to_idx = vocab.build_char_to_idx(self.val_sorted_alphabets)

    def __len__(self):
        return (self.train_dict['strokes'].shape, self.val_dict['strokes'].shape)



if __name__ == '__main__':
    dataset = Dataset()

