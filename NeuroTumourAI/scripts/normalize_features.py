#!/usr/bin/env python3
"""Train-only feature scaling and zero-imputation, applied identically to held-out caches.
Usage: fit --features train/features.csv --output scalers.npz
       transform --features val/features.csv --scalers scalers.npz --output val_scaled/
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler

def rows(path):
 with open(path,newline='') as f:return list(csv.DictReader(f))
def matrix(records,key):return np.stack([np.load(r['npz'],allow_pickle=False)[key] for r in records])
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['fit','transform']);p.add_argument('--features',required=True);p.add_argument('--output',required=True);p.add_argument('--scalers');a=p.parse_args();r=rows(a.features)
 if a.action=='fit':
  settings={}
  for k in ['radiomics','morphometrics']:
   x=matrix(r,k);x[~np.isfinite(x)]=np.nan;med=np.nanmedian(x,axis=0);med=np.where(np.isfinite(med),med,0);x=np.where(np.isfinite(x),x,med);s=StandardScaler().fit(x);settings[k+'_median']=med;settings[k+'_mean']=s.mean_;settings[k+'_scale']=np.maximum(s.scale_,1e-8)
  np.savez(a.output,**settings);print('Fitted training-only scalers:',a.output)
 else:
  if not a.scalers:raise ValueError('--scalers is required for transform')
  v=np.load(a.scalers);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);new=[]
  for i,row in enumerate(r):
   with np.load(row['npz'],allow_pickle=False) as z:data={k:z[k] for k in z.files}
   for k in ['radiomics','morphometrics']:
    x=data[k];med=v[k+'_median'];x=np.where(np.isfinite(x),x,med);data[k]=((x-v[k+'_mean'])/v[k+'_scale']).astype(np.float32)
   path=out/f'patch_{i:07d}.npz';np.savez_compressed(path,**data);new.append({**row,'npz':str(path.resolve())})
  with open(out/'features.csv','w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=new[0].keys());w.writeheader();w.writerows(new)
  print('Wrote standardized cache:',out/'features.csv')
if __name__=='__main__':main()
