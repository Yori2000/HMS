import torch

class AccuracyTable:
    def __init__(self):
        self.table = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],
                      [0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
        self.num = [0,0,0,0,0,0]
    
    def add(self, out, consensus):
        B,_ = out.shape
        
        for b in range(B):
            _y = out[b]
            _y = torch.argmax(_y).item()
            _c = consensus[b].item()
            
            self.num[_c] += 1
            self.table[_c][_y] += 1
        
    def write_tensorboard(self, writer, step):
        l = ["Seizure","LPD","GPD","LRDA","GRDA","Other"]
        ave = []
        for t in self.table:
            _ave = [x / sum(t) for x in t]
            ave.append(_ave)
            
        for i, consensus in enumerate(l):
            for j, predict in enumerate(l):
                tab = "Predict/{}/{}".format(consensus,predict)
                writer.add_scalar(tab, ave[i][j], step)