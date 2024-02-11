import torch
import torch.nn as nn
import torch.nn.functional as F

class KLDivLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_vote      = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, y, vote_target):
        vote_probability    = F.normalize(vote_target.to(torch.float32), dim=1)
        vote_loss           = self.loss_vote(torch.log(y+1e-12), vote_probability)
        
        return vote_loss