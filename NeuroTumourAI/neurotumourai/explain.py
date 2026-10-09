"""Gradient and perturbation based attributions. Exploratory, not clinical validation."""
import numpy as np
import torch

def gradcam3d(model,images,rad,morph,graphs,target_class=1):
 """First-order Grad-CAM. For true Grad-CAM++ use gradcam_plusplus3d below."""
 activations=[];gradients=[]
 def forward(_,inputs,out):activations.append(out)
 def backward(_,g_in,g_out):gradients.append(g_out[0])
 layer=model.encoder.stages[-1];h1=layer.register_forward_hook(forward);h2=layer.register_full_backward_hook(backward)
 try:
  model.zero_grad(set_to_none=True);logits=model(images,rad,morph,graphs);logits[:,target_class].sum().backward();a=activations[0];g=gradients[0];w=g.mean((2,3,4),keepdim=True);heat=torch.relu((a*w).sum(1,keepdim=True));heat=torch.nn.functional.interpolate(heat,size=images.shape[-3:],mode='trilinear',align_corners=False);heat=heat/(heat.amax(dim=(2,3,4),keepdim=True)+1e-8);return heat.detach()
 finally:h1.remove();h2.remove()
def gradcam_plusplus3d(model,images,rad,morph,graphs,target_class=1):
 # Grad-CAM++ alpha weighting, computed using analytical higher-order derivatives of the softmax-exponential score.
 activation=[]
 def hook(_,i,o):activation.append(o)
 h=model.encoder.stages[-1].register_forward_hook(hook)
 try:
  logits=model(images,rad,morph,graphs);score=logits[:,target_class].exp().sum();g=torch.autograd.grad(score,activation[0],retain_graph=False)[0];a=activation[0];g2=g.square();g3=g2*g;denom=2*g2+(a*g3).sum(dim=(2,3,4),keepdim=True);alpha=g2/(denom+1e-8);weights=(alpha*torch.relu(g)).sum((2,3,4),keepdim=True);cam=torch.relu((weights*a).sum(1,keepdim=True));cam=torch.nn.functional.interpolate(cam,size=images.shape[-3:],mode='trilinear',align_corners=False);return (cam/(cam.amax(dim=(2,3,4),keepdim=True)+1e-8)).detach()
 finally:h.remove()
def perturb_graph(model,images,rad,morph,graphs,target=1):
 model.eval();scores=[]
 with torch.no_grad():
  original=model(images,rad,morph,graphs).softmax(1)[:,target]
  for i,g in enumerate(graphs):
   imp=[]
   for j in range(len(g['x'])):
    altered=[{'x':v['x'].clone(),'edge_index':v['edge_index']} for v in graphs];altered[i]['x'][j]=0;new=model(images,rad,morph,altered).softmax(1)[i,target];imp.append(float(original[i]-new))
   scores.append(imp)
 return scores
def feature_occlusion(model,images,rad,morph,graphs,kind='radiomics',target=1):
 """Model-faithful zero-reference feature occlusion for qualitative comparisons."""
 base=rad if kind=='radiomics' else morph;importance=[]
 with torch.no_grad():
  ref=model(images,rad,morph,graphs).softmax(1)[:,target]
  for k in range(base.shape[1]):
   altered=base.clone();altered[:,k]=0;prob=model(images,altered if kind=='radiomics' else rad,altered if kind=='morphometrics' else morph,graphs).softmax(1)[:,target];importance.append((ref-prob).cpu().numpy())
 return np.stack(importance,axis=1)
def shap_kernel(model,images,rad,morph,graphs,background,target=1,nsamples=200):
 import shap
 assert images.shape[0]==1
 def predict(x):
  result=[]
  with torch.no_grad():
   for row in x:
    r=torch.tensor(row,dtype=rad.dtype,device=rad.device)[None];result.append(float(model(images,r,morph,graphs).softmax(1)[0,target]))
  return np.array(result)
 explainer=shap.KernelExplainer(predict,background);return explainer.shap_values(rad.detach().cpu().numpy(),nsamples=nsamples)
