import torch
from model import darknet53

checkpoint = torch.load("model_best.pth.tar", map_location='cpu')
model = darknet53(num_classes=1000)
model.load_state_dict(checkpoint['state_dict'])