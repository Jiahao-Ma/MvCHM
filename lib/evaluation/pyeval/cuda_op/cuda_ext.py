import torch
from torch import nn
from torch.autograd import Function
import sys
sys.path.append('/home/dzc/Projects/MvCHM')
import sort_vertices

class SortVertices(Function):
    @staticmethod
    def forward(ctx, vertices, mask, num_valid):
        idx = sort_vertices.sort_vertices_forward(vertices, mask, num_valid)
        ctx.mark_non_differentiable(idx)
        return idx
    
    @staticmethod
    def backward(ctx, gradout):
        return ()

sort_v = SortVertices.apply

if __name__ == "__main__":
    import time
    v = torch.rand([8, 1024, 24, 2]).float().cuda()
    mean = torch.mean(v, dim=2, keepdim=True)
    v = v - mean
    m = (torch.rand([8, 1024, 24]) > 0.8).cuda()
    nv = torch.sum(m.int(), dim=-1).int().cuda()
    start = time.time()
    result = sort_v(v, m, nv)
    torch.cuda.synchronize()
    print("time: %.2f ms"%((time.time() - start)*1000))
    print(result.size())
    print(result[0,0,:])

    # adjust
    v = torch.rand([8, 1024, 24, 2]).float().cuda()
    mask = (torch.randn([8, 1024, 24]) > 0.8).cuda()
    nv = torch.sum(mask.int(), dim=-1).int().cuda()
    mean = torch.sum(v * mask.float().unsqueeze(-1), dim=2, keepdim=True) / nv.unsqueeze(-1).unsqueeze(-1)
    v = v - mean
    result = sort_v(v, mask, nv)
    print(result.shape)
    print(result[0, 0, :])