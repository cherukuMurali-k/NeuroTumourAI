"""MRI data IO, cohort manifests, fold-fitted processing and patch generation."""
import csv,json,os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
MODALITIES=('t1','t1ce','t2','flair')
def require_nii():
 try:import nibabel as nib;return nib
 except ImportError as e:raise ImportError('Install nibabel to process NIfTI volumes: pip install nibabel') from e
def load_nifti(path):
 nib=require_nii();obj=nib.load(str(path));return np.asarray(obj.get_fdata(dtype=np.float32)),obj.affine,obj.header.get_zooms()[:3]
def read_manifest(path):
 with open(path,newline='') as f:rows=list(csv.DictReader(f))
 ids=[r['subject_id'] for r in rows]
 if len(ids)!=len(set(ids)):raise ValueError('Duplicate subject IDs')
 for r in rows:
  if r['label'] not in ('LGG','HGG'):raise ValueError('Missing or invalid verified LGG/HGG label: '+r['subject_id'])
  for m in MODALITIES:
   if not r.get(m) or not Path(r[m]).exists():raise FileNotFoundError(f'{r["subject_id"]}: missing {m}')
 return rows
def create_folds(rows,n_splits=5,seed=42):
 y=np.array([int(r['label']=='HGG') for r in rows]);idx=np.arange(len(rows));split=StratifiedKFold(n_splits,shuffle=True,random_state=seed);return [(idx[a].tolist(),idx[b].tolist()) for a,b in split.split(idx,y)]
def ensure_disjoint(a,b):
 ids1={r['subject_id'] for r in a};ids2={r['subject_id'] for r in b};shared=ids1&ids2
 if shared:raise ValueError('Overlapping cohort subjects: '+str(sorted(shared)[:10]))
def modality_zscore(x,mask=None):
 mask=(x!=0) if mask is None else mask.astype(bool);v=x[mask];out=np.zeros_like(x,dtype=np.float32)
 if v.size:out[mask]=(v-v.mean())/max(float(v.std()),1e-6)
 return out
class NylStandardizer:
 def __init__(self,percentiles=(1,10,25,50,75,90,99)):self.percentiles=percentiles;self.landmarks=None
 def fit(self,volumes):
  # volumes iterable of four-channel ndarrays from TRAINING subjects only
  arrays=[[] for _ in MODALITIES]
  for volume in volumes:
   for c in range(4):
    nonzero=volume[c][volume[c]!=0]
    if len(nonzero):arrays[c].append(np.percentile(nonzero,self.percentiles))
  self.landmarks=np.stack([np.mean(a,axis=0) if a else np.arange(len(self.percentiles)) for a in arrays]);return self
 def transform(self,volume):
  if self.landmarks is None:raise ValueError('Fit on training cohort first')
  output=volume.copy()
  for c in range(4):
   mask=volume[c]!=0
   if mask.sum()>1:src=np.percentile(volume[c][mask],self.percentiles);src=np.maximum.accumulate(src+np.arange(len(src))*1e-7);output[c][mask]=np.interp(volume[c][mask],src,self.landmarks[c]).astype(np.float32)
  return output
 def save(self,path):np.savez(path,percentiles=self.percentiles,landmarks=self.landmarks)
 @classmethod
 def load(cls,path):
  v=np.load(path);obj=cls(tuple(v['percentiles']));obj.landmarks=v['landmarks'];return obj

def prep_volumes(row,n4=False,register=False,brain_mask=True):
 import scipy.ndimage as ndi
 data=[];affine=None;spacing=None
 for m in MODALITIES:
  v,a,s=load_nifti(row[m]);data.append(v);affine=a if affine is None else affine;spacing=s if spacing is None else spacing
 if len(set(tuple(v.shape) for v in data))!=1 or any(not np.allclose(load_nifti(row[m])[1],affine,atol=1e-3) for m in MODALITIES):
  if not register:raise ValueError('Volumes are not aligned. Set register=True after installing SimpleITK.')
  try:import SimpleITK as sitk
  except ImportError as e:raise ImportError('SimpleITK required for registration') from e
  ref=sitk.ReadImage(row['t1']);new=[]
  for m in MODALITIES:
   moving=sitk.ReadImage(row[m]);tx=sitk.CenteredTransformInitializer(ref,moving,sitk.Euler3DTransform());reg=sitk.ImageRegistrationMethod();reg.SetMetricAsMattesMutualInformation(32);reg.SetOptimizerAsRegularStepGradientDescent(1,0.001,100);reg.SetInterpolator(sitk.sitkLinear);tx=reg.Execute(ref,moving) if m!='t1' else sitk.Transform(3,sitk.sitkIdentity);aligned=sitk.Resample(moving,ref,tx,sitk.sitkLinear,0,moving.GetPixelID());new.append(sitk.GetArrayFromImage(aligned).transpose(2,1,0))
  data=new
 if n4:
  try:import SimpleITK as sitk
  except ImportError as e:raise ImportError('SimpleITK required for N4 bias correction') from e
  corrected=[]
  for v in data:
   im=sitk.GetImageFromArray(v.transpose(2,1,0));mask=sitk.OtsuThreshold(im,0,1,100);out=sitk.N4BiasFieldCorrectionImageFilter().Execute(im,mask);corrected.append(sitk.GetArrayFromImage(out).transpose(2,1,0))
  data=corrected
 data=np.stack(data).astype(np.float32)
 if brain_mask:
  # Tissue mask is an APPROXIMATION, not a learned skull-stripper.
  mask=np.any(data!=0,axis=0);mask=ndi.binary_fill_holes(mask);data*=mask[None]
 data=np.stack([modality_zscore(v) for v in data]);return data,affine,spacing

def mask_to_regions(raw):
 # BraTS conventional codes 1 necrotic/nonenhancing, 2 edema, 4 enhancing; some releases use 3 enhancing.
 vals=set(np.unique(raw).astype(int));enhancing=4 if 4 in vals else 3
 wt=raw>0;tc=(raw==1)|(raw==enhancing);et=raw==enhancing
 return np.stack([wt,tc,et]).astype(np.float32)
def patches(image,regions,size=128,stride=64,min_fraction=.1):
 from itertools import product
 mask=regions[0]>0;coordinates=np.argwhere(mask)
 if len(coordinates)==0:return []
 lo=np.maximum(coordinates.min(0)-16,0);hi=np.minimum(coordinates.max(0)+17,mask.shape);shape=np.asarray(image.shape[1:]);lo=np.minimum(lo,np.maximum(hi-size,0));hi=np.maximum(hi,np.minimum(lo+size,shape))
 starts=[]
 for a,b,s in zip(lo,hi,[size]*3):
  end=max(int(b)-s,int(a));choices=list(range(int(a),end+1,stride));choices.append(end);starts.append(sorted(set(choices)))
 output=[]
 for pos in product(*starts):
  sl=tuple(slice(k,k+size) for k in pos);im=image[(slice(None),)+sl];m=regions[(slice(None),)+sl];pad=[(0,0)]+[(0,max(0,size-v)) for v in im.shape[1:]];im=np.pad(im,pad);m=np.pad(m,pad)
  # Fraction is with respect to actual (unpadded) patch volume.
  if m[0].mean()<min_fraction:continue
  output.append((im.astype(np.float32),m.astype(np.float32),pos))
 # tiny tumours would otherwise have zero valid patches; document and retain a centered ROI patch
 if not output:
  center=coordinates.mean(0).astype(int);pos=np.maximum(0,np.minimum(center-size//2,np.maximum(shape-size,0)));sl=tuple(slice(int(k),int(k)+size) for k in pos);im=image[(slice(None),)+sl];m=regions[(slice(None),)+sl];pad=[(0,0)]+[(0,max(0,size-v)) for v in im.shape[1:]];output=[(np.pad(im,pad),np.pad(m,pad),tuple(pos))]
 return output
class FeatureDataset(Dataset):
 def __init__(self,records):self.records=records
 def __len__(self):return len(self.records)
 def __getitem__(self,i):
  r=self.records[i];z=np.load(r['npz'],allow_pickle=False);return {'image':torch.tensor(z['image'],dtype=torch.float32),'radiomics':torch.tensor(z['radiomics'],dtype=torch.float32),'morphometrics':torch.tensor(z['morphometrics'],dtype=torch.float32),'graph_x':torch.tensor(z['graph_x'],dtype=torch.float32),'graph_edge':torch.tensor(z['graph_edge'],dtype=torch.long),'label':int(r['label']=='HGG'),'subject_id':r['subject_id']}
def batch_collate(batch):
 return {'images':torch.stack([b['image'] for b in batch]),'radiomics':torch.stack([b['radiomics'] for b in batch]),'morphometrics':torch.stack([b['morphometrics'] for b in batch]),'graphs':[{'x':b['graph_x'],'edge_index':b['graph_edge']} for b in batch],'labels':torch.tensor([b['label'] for b in batch],dtype=torch.long),'subject_ids':[b['subject_id'] for b in batch]}
