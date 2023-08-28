import torch
import numpy as np

feats_in_3d = ["clc", "temp", "pres", "qc", "qi", "qv"]
feats_in_2d = ["pres_sfc", "cosmu0", "qv_s", "albvisdir", "albnirdir", "tsfctrad", "albvisdif", "albnirdif", ]
feats_out_3d = ["lwflx_up", "lwflx_dn", "swflx_up", "swflx_dn"]
len_height = 70

num_in_2d = len(feats_in_2d)
num_in_3d = len(feats_in_3d)
num_out_3d = len(feats_out_3d)

class Net(torch.nn.Module):
    def __init__(self, num_out_3d):
        super().__init__()
        self.num_out_3d = num_out_3d
    def forward(self, x2d, x3d):
        #x2d, x3d = x[0], x[1]
        y = torch.zeros((x3d.shape[0], x3d.shape[1], self.num_out_3d), dtype=torch.float32)
        y[...,0] = torch.sum(x3d, -1)
        y[...,1] = torch.tile(torch.sum(x3d[:,:,0], 1, True), (1, x3d.shape[1],))
        y[...,2] = x3d[...,0]
        y[...,3] = torch.tile(x2d[...,3:4], (1,x3d.shape[1]))
        return y
            
model = Net(num_out_3d)

model.to("cuda")
model_jit = torch.jit.script(model)
model_jit.save('dummy_torch_model.pt')
