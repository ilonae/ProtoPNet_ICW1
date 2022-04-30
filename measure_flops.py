
import argparse
import numpy as np
import torch
import os
from pthflops import count_ops
from thop import profile,clever_format

print("resuming model")
device = 'cuda:0'
save_loc = "./saved_models/resnet18/003/pruned_prototypes_epoch8_k6_pt3/8_99prune0.8957.pth"
ppnet = torch.load(save_loc)
ppnet = ppnet.cuda()
inp = torch.rand(1,3,224,224).to(device)
input = torch.randn(1, 3, 224, 224)
macs, params = profile(ppnet, inputs=(inp, ))
# Count the number of FLOPs
macs, params = clever_format([macs, params], "%.3f")
print(macs, params)
ppnet_multi = torch.nn.DataParallel(ppnet)