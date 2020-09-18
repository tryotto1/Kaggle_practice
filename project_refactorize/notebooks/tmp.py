import torch 

weight_path = f'/home/sykim/Desktop/project_refactorize/seresnext50_fold_1.pt'
print(torch.load(weight_path))

import pretrainedmodels
from torch import nn, cuda

num_classes=196
model = pretrainedmodels.se_resnext50_32x4d(pretrained=None)
model.last_linear = nn.Linear(2048, num_classes)

model.load_state_dict(torch.load(weight_path))