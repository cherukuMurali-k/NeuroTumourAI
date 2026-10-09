import copy,json,random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .models import HybridTumourNet,UNet3D
from .losses import CompositeLoss,dice_loss
from .data import FeatureDataset,batch_collate
from .evaluation import subject_aggregate,metrics

def seed_all(seed=42):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def transfer(b,device):return (b['images'].to(device),b['radiomics'].to(device),b['morphometrics'].to(device),[{'x':g['x'].to(device),'edge_index':g['edge_index'].to(device)} for g in b['graphs']],b['labels'].to(device))
def predict(model,records,batch_size=1,device='cpu'):
 loader=DataLoader(FeatureDataset(records),batch_size=batch_size,shuffle=False,collate_fn=batch_collate,num_workers=0);model.eval();logits=[];ys=[];ids=[]
 with torch.no_grad():
  for b in loader:
   im,rad,morph,graphs,y=transfer(b,device);out=model(im,rad,morph,graphs);logits.extend(out.cpu().numpy());ys.extend(y.cpu().numpy());ids.extend(b['subject_ids'])
 return np.array(logits),np.array(ys),ids

def train_classifier(train_records,val_records,out_dir,epochs=200,patience=20,batch_size=1,grad_accum=4,lr=3e-4,device=None,seed=42,model_options=None):
 seed_all(seed);device=device or ('cuda' if torch.cuda.is_available() else 'cpu');out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
 model=HybridTumourNet(**(model_options or {})).to(device);criterion=CompositeLoss().to(device);opt=torch.optim.AdamW(list(model.parameters())+list(criterion.parameters()),lr=lr,weight_decay=1e-4);loader=DataLoader(FeatureDataset(train_records),batch_size=batch_size,shuffle=True,collate_fn=batch_collate,num_workers=0);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(epochs-10,1));scaler=torch.amp.GradScaler('cuda',enabled=str(device).startswith('cuda'));best=-1.;wait=0;history=[];ema=copy.deepcopy(model).eval();decay=.999
 for epoch in range(epochs):
  model.train();opt.zero_grad(set_to_none=True);losses=[]
  for step,b in enumerate(loader):
   im,rad,morph,graphs,y=transfer(b,device)
   with torch.autocast(device_type='cuda',enabled=str(device).startswith('cuda')):
    pred=model(im,rad,morph,graphs,True);loss,details=criterion(pred,y);scaled_loss=loss/grad_accum
   scaler.scale(scaled_loss).backward();losses.append(loss.item())
   if (step+1)%grad_accum==0 or step+1==len(loader):
    scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
    with torch.no_grad():
     for a,beta in zip(ema.parameters(),model.parameters()):a.mul_(decay).add_(beta,alpha=1-decay)
  if epoch<10:
   for group in opt.param_groups:group['lr']=lr*(epoch+1)/10
  else:scheduler.step()
  v_logits,vy,vid=predict(ema,val_records,batch_size,device);sub=subject_aggregate(v_logits,vy,vid);met=metrics([v['label'] for v in sub.values()],[v['prob'] for v in sub.values()]);history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'validation':met});print(f'epoch={epoch+1} train_loss={np.mean(losses):.4f} val_macro_f1={met["macro_f1"]:.4f}',flush=True)
  if met['macro_f1']>best+1e-6:
   best=met['macro_f1'];wait=0;torch.save({'model':ema.state_dict(),'model_options':model_options or {},'epoch':epoch+1,'validation_metrics':met},out/'best.pt')
  else:wait+=1
  (out/'history.json').write_text(json.dumps(history,indent=2))
  if wait>=patience:break
 return model

def load_classifier(path,device='cpu'):
 ck=torch.load(path,map_location=device,weights_only=False);model=HybridTumourNet(**ck.get('model_options',{}));model.load_state_dict(ck['model']);return model.to(device).eval()

def train_segmentation(train_loader,val_loader,out_dir,epochs=100,device=None):
 device=device or ('cuda' if torch.cuda.is_available() else 'cpu');model=UNet3D().to(device);opt=torch.optim.AdamW(model.parameters(),lr=3e-4);out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);best=1e9
 for ep in range(epochs):
  model.train()
  for im,mask in train_loader:
   im=im.to(device);mask=mask.to(device);opt.zero_grad();pred=model(im);loss=torch.nn.functional.binary_cross_entropy_with_logits(pred,mask)+dice_loss(pred,mask);loss.backward();opt.step()
  model.eval();losses=[]
  with torch.no_grad():
   for im,mask in val_loader:
    pred=model(im.to(device));losses.append(dice_loss(pred,mask.to(device)).item())
  avg=float(np.mean(losses));print('seg_epoch',ep+1,'dice_loss',avg)
  if avg<best:best=avg;torch.save({'model':model.state_dict(),'epoch':ep+1},out/'segmenter.pt')
 return model
