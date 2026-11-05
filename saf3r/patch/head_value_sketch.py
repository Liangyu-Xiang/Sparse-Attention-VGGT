"""Fixed orthogonal sketches within each head; heads are never averaged together."""
import math
import torch

def projection(heads,dimension,sketch_dimension,device):
    if dimension&(dimension-1) or not 1<=sketch_dimension<=dimension:
        raise ValueError('Head sketch requires a power-of-two head dimension and a valid sketch size')
    basis=torch.ones(1,1)
    while len(basis)<dimension:basis=torch.cat((torch.cat((basis,basis),1),torch.cat((basis,-basis),1)),0)
    generator=torch.Generator(device='cpu').manual_seed(7351)
    matrices=[]
    for _ in range(heads):
        signs=torch.randint(0,2,(dimension,),generator=generator).float()*2-1
        rows=torch.randperm(dimension,generator=generator)[:sketch_dimension]
        matrices.append((basis[rows]*signs).T/math.sqrt(sketch_dimension*heads))
    return torch.stack(matrices).to(device=device).contiguous()

def features(v,matrix):
    projected=torch.bmm(v.float(),matrix)
    return projected.permute(1,0,2).reshape(v.shape[1],-1).contiguous()
