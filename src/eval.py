import torch
from dataset import HmsTrainDataset2
from test import SimpleModel

device = torch.device('cuda')
p =  "/work/abelab4/k_hiro/study/HMS/checkpoint/double_lstm/175000.pth"
model = SimpleModel().to(device)
model.load_state_dict(torch.load(p))

dataset = HmsTrainDataset2("./data", "./data/train.csv")

idx = 5000
eeg, consensus, vote = dataset[idx]
x = eeg.unsqueeze(0).to(device)

y = model(x)
print(y, consensus)
vote = torch.nn.functional.normalize(vote.unsqueeze(0).to(torch.float32), dim=1)
loss = torch.nn.functional.kl_div(torch.log(y), vote.to(device), reduction="batchmean")
print(loss)