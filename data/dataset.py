##Setting up future file for potential custom dataset function
from torch.utils.data import Dataset



class fMRIDataset(Dataset):

    def __init__(self):
        super(fMRIDataset, self).__init__()
        self.data = []

        return

    def __getitem__(self, item):

        return

    def __len__(self):

        return len(self.data)

