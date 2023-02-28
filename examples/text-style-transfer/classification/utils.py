import torch
from torch.utils.data import Dataset
import numpy as np
import random

class SIOP_Dataset(Dataset):
    def __init__(self, split='train'):
        self.answer = []
        self.label = []
        for label in [0,1]:
            if label == 1:
                filepath = f'../data/siop/{split}_positive.txt'
            elif label == 0:
                filepath = f'../data/siop/{split}_negative.txt'
            else:
                raise
            with open(filepath, 'r+') as f:
                sentences = [line.strip() for line in f]
            self.answer.extend(sentences)
            self.label.extend([label]*len(sentences))
        assert len(self.answer) == len(self.label)

    def __len__(self):
        return len(self.answer)

    def __getitem__(self, idx):
        return self.answer[idx], torch.tensor(self.label[idx])

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    siop = SIOP_Dataset()
    print(siop.__getitem__(5))
    