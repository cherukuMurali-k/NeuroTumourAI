"""Differentiable NeuroTumourAI networks. Tensor conventions: B,C,D,H,W."""
import torch
from torch import nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
 def __init__(self,ci,co):
  super().__init__(); self.net=nn.Sequential(nn.Conv3d(ci,co,3,padding=1,bias=False),nn.GroupNorm(min(8,co),co),nn.LeakyReLU(.1,inplace=True),nn.Conv3d(co,co,3,padding=1,bias=False),nn.GroupNorm(min(8,co),co),nn.LeakyReLU(.1,inplace=True))
 def forward(self,x):return self.net(x)
class UNet3D(nn.Module):
 def __init__(self,channels=4,base=16):
  super().__init__();b=base;self.e1=ConvBlock(channels,b);self.e2=ConvBlock(b,b*2);self.e3=ConvBlock(b*2,b*4);self.bot=ConvBlock(b*4,b*8);self.d3=ConvBlock(b*12,b*4);self.d2=ConvBlock(b*6,b*2);self.d1=ConvBlock(b*3,b);self.out=nn.Conv3d(b,3,1)
 def forward(self,x):
  a=self.e1(x);b=self.e2(F.max_pool3d(a,2));c=self.e3(F.max_pool3d(b,2));d=self.bot(F.max_pool3d(c,2));u=self.d3(torch.cat([F.interpolate(d,size=c.shape[-3:],mode='trilinear',align_corners=False),c],1));u=self.d2(torch.cat([F.interpolate(u,size=b.shape[-3:],mode='trilinear',align_corners=False),b],1));u=self.d1(torch.cat([F.interpolate(u,size=a.shape[-3:],mode='trilinear',align_corners=False),a],1));return self.out(u)

def regions_from_logits(logits,threshold=.5):
 p=(logits.sigmoid()>threshold);wt=p[:,0];tc=p[:,1]&wt;et=p[:,2]&tc
 return torch.stack((wt,tc,et),1)
class SE(nn.Module):
 def __init__(self,c):super().__init__();self.net=nn.Sequential(nn.AdaptiveAvgPool3d(1),nn.Conv3d(c,max(1,c//8),1),nn.ReLU(),nn.Conv3d(max(1,c//8),c,1),nn.Sigmoid())
 def forward(self,x):return x*self.net(x)
class CBAM3D(nn.Module):
 def __init__(self,c):
  super().__init__();h=max(1,c//8);self.channel=nn.Sequential(nn.Linear(c,h),nn.ReLU(),nn.Linear(h,c));self.spatial=nn.Conv3d(2,1,7,padding=3)
 def forward(self,x):
  avg=F.adaptive_avg_pool3d(x,1).flatten(1);mx=F.adaptive_max_pool3d(x,1).flatten(1);x=x*torch.sigmoid(self.channel(avg)+self.channel(mx))[:,:,None,None,None];return x*torch.sigmoid(self.spatial(torch.cat([x.mean(1,keepdim=True),x.amax(1,keepdim=True)],1)))
class Encoder3D(nn.Module):
 def __init__(self,channels=4,base=16,out_dim=512):
  super().__init__();cs=[base,base*2,base*4,base*8];self.stages=nn.ModuleList();prev=channels
  for c in cs:self.stages.append(nn.Sequential(nn.Conv3d(prev,c,3,stride=2,padding=1),nn.GroupNorm(min(8,c),c),nn.GELU(),ConvBlock(c,c),SE(c),CBAM3D(c)));prev=c
  self.proj=nn.Linear(sum(cs),out_dim);self.last_activation=None
 def forward(self,x):
  pool=[]
  for s in self.stages:x=s(x);pool.append(F.adaptive_avg_pool3d(x,1).flatten(1))
  self.last_activation=x
  return self.proj(torch.cat(pool,1))
class SageLayer(nn.Module):
 def __init__(self,ci,co):super().__init__();self.self_fc=nn.Linear(ci,co);self.neigh_fc=nn.Linear(ci,co,bias=False)
 def forward(self,x,edge_index):
  n=x.shape[0];out=torch.zeros_like(x);deg=x.new_zeros(n,1)
  if edge_index.numel():
   src,dst=edge_index;out.index_add_(0,dst,x[src]);deg.index_add_(0,dst,torch.ones((dst.numel(),1),device=x.device,dtype=x.dtype))
  return F.relu(self.self_fc(x)+self.neigh_fc(out/deg.clamp_min(1)))
class GraphSAGE(nn.Module):
 def __init__(self,input_dim=8,hidden=64,out_dim=128):super().__init__();self.a=SageLayer(input_dim,hidden);self.b=SageLayer(hidden,out_dim)
 def forward(self,graphs):
  pooled=[]
  for g in graphs:
   x=g['x'];e=g['edge_index'];h=self.b(self.a(x,e),e);pooled.append(h.mean(0) if h.numel() else x.new_zeros(self.b.self_fc.out_features))
  return torch.stack(pooled)
class GCATLayer(nn.Module):
 def __init__(self,d=256,heads=8,ff=512,drop=.1):
  super().__init__();self.attn=nn.MultiheadAttention(d,heads,dropout=drop,batch_first=True);self.gate=nn.Linear(d*2,d);self.norm1=nn.LayerNorm(d);self.norm2=nn.LayerNorm(d);self.ff=nn.Sequential(nn.Linear(d,ff),nn.GELU(),nn.Dropout(drop),nn.Linear(ff,d));self.drop=nn.Dropout(drop)
 def forward(self,x):
  a,_=self.attn(x,x,x,need_weights=False);g=torch.sigmoid(self.gate(torch.cat((x,a),-1)));x=self.norm1(x+self.drop(g*a));return self.norm2(x+self.drop(self.ff(x)))
class GMoE(nn.Module):
 def __init__(self,d=256,n=4):
  super().__init__();self.experts=nn.ModuleList([nn.Sequential(nn.Linear(d,512),nn.GELU(),nn.Linear(512,d)) for _ in range(n)]);self.gate=nn.Linear(d,n)
 def forward(self,x):
  weights=torch.sigmoid(self.gate(x));ex=torch.stack([e(x) for e in self.experts],1);z=(ex*weights.unsqueeze(-1)).sum(1)/weights.sum(1,keepdim=True).clamp_min(1e-6);return z,weights
class HybridTumourNet(nn.Module):
 def __init__(self,deep_dim=512,rad_dim=128,morph_dim=14,graph_dim=128,d=256,layers=3,heads=8,experts=4,encoder_base=16,graph_input=8):
  super().__init__();self.encoder=Encoder3D(base=encoder_base,out_dim=deep_dim);self.graphsage=GraphSAGE(graph_input,out_dim=graph_dim);self.projections=nn.ModuleList([nn.Linear(k,d) for k in (deep_dim,rad_dim,morph_dim,graph_dim)]);self.types=nn.Parameter(torch.randn(4,d)*.02);self.gcat=nn.Sequential(*[GCATLayer(d,heads) for _ in range(layers)]);self.moe=GMoE(d,experts);self.head=nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.Linear(128,2))
 def forward(self,images,radiomics,morphometrics,graphs,return_aux=False):
  deep=self.encoder(images);graph=self.graphsage(graphs);raw=[deep,radiomics,morphometrics,graph];tokens=torch.stack([p(z)+self.types[i] for i,(p,z) in enumerate(zip(self.projections,raw))],1);att=self.gcat(tokens);fused,gates=self.moe(att.mean(1));logits=self.head(fused)
  return {'logits':logits,'embedding':fused,'tokens':tokens,'gates':gates,'deep':deep,'graph':graph} if return_aux else logits
