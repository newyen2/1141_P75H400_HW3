import torch

class TorchHelper:
    def __init__(self):
        self.device = torch.device("cpu")
    
    def f(self, x):
        return torch.tensor(x).float()
    
    def i(self, x):
        return torch.tensor(x).int()
    
    def l(self, x):
        return torch.tensor(x).long()