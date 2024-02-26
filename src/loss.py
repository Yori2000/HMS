import torch
import torch.nn as nn
import torch.nn.functional as F

class KLDivLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn      = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, y, target):
        
        target = target.to(torch.float64)
        y = y + 1e-12
        y = F.log_softmax(y, dim=1)
        loss           = self.loss_fn(y, target)
        
        return loss