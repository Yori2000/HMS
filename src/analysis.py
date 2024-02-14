import torch

class AccuracyTable:
    def __init__(self):
        self.table = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],
                      [0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
        self.num = [1,1,1,1,1,1]
    
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
        for t, num in zip(self.table, self.num):
            _ave = [x / num for x in t]
            ave.append(_ave)
            
        for i, consensus in enumerate(l):
            tag = "Predict/{}".format(consensus)
            writer.add_scalars(tag, {"Seizure":ave[i][0], "LPD":ave[i][1],"GPD":ave[i][2],
                                        "LRDA":ave[i][3],"GRDA":ave[i][4],"Other":ave[i][5]}, step)