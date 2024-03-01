import torch
import wandb
class AccuracyTable:
    def __init__(self):
        self.table = [[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],
                      [0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.]]
        self.num = [1,1,1,1,1,1]
    
    def add(self, out, consensus):
        B,_ = out.shape
        
        for b in range(B):
            _y = out[b]
            _p = torch.argmax(_y).item()
            _c = consensus[b].item()
            
            self.num[_c] += 1
            self.table[_c][_p] += 1
        
    def get_table(self):
        l = ["Seizure","LPD","GPD","LRDA","GRDA","Other"]
        ave = []
        for t, num in zip(self.table, self.num):
            _ave = [x / num for x in t]
            ave.append(_ave)
        return ave
    
    def log_wandb(self, fold, step):
        l = ["Seizure","LPD","GPD","LRDA","GRDA","Other"]
        ave = []
        for t, num in zip(self.table, self.num):
            _ave = [x / num for x in t]
            ave.append(_ave)
        for i, target in enumerate(l):
            for j, predict in enumerate(l):
                tag = "fold {}:{}_{}".format(fold, target ,predict)
                wandb.log({tag: ave[i][j]}, step=step)