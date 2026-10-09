import torch
from torch import nn
import torch.nn.functional as F

def dice_loss(logits,target,eps=1e-6):
 p=logits.sigmoid();dims=tuple(range(2,p.ndim));return (1-(2*(p*target).sum(dims)+eps)/(p.sum(dims)+target.sum(dims)+eps)).mean()
def focal_loss(logits,target,gamma=1.5,smoothing=.05,class_weights=None):
 logp=F.log_softmax(logits,dim=1);p=logp.exp();n=logits.shape[1];q=F.one_hot(target,n).float()*(1-smoothing)+smoothing/n;loss=-(q*(1-p).pow(gamma)*logp)
 if class_weights is not None:loss=loss*class_weights.to(logits.device)[None,:]
 return loss.sum(1).mean()
class CompositeLoss(nn.Module):
 def __init__(self,weights=(1.,.3,.1,.05,.001),num_classes=2):super().__init__();self.weights=weights;self.centers=nn.Parameter(torch.zeros(num_classes,256))
 def forward(self,out,target,seg_logits=None,seg_target=None,class_weights=None):
  cls=focal_loss(out['logits'],target,class_weights=class_weights);d=dice_loss(seg_logits,seg_target) if seg_logits is not None and seg_target is not None else cls.new_zeros(());t=F.normalize(out['tokens'],dim=-1);sim=torch.matmul(t,t.transpose(1,2));eye=torch.eye(t.shape[1],device=t.device);orth=((sim-eye)**2).mean();cent=(out['embedding']-self.centers[target]).square().mean();sparse=out['gates'].mean();w=self.weights
  total=w[0]*cls+(w[1]*d if seg_logits is not None else 0)+w[2]*orth+w[3]*cent+w[4]*sparse
  return total,{'total':total.detach().item(),'cls':cls.detach().item(),'dice':d.detach().item(),'orth':orth.detach().item(),'center':cent.detach().item(),'sparse':sparse.detach().item()}
