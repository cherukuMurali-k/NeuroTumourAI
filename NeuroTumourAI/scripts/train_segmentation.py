#!/usr/bin/env python3
"""Fold-isolated 3D U-Net training. Uses training-subject reference masks only."""
import argparse,csv,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from neurotumourai.data import read_manifest,prep_volumes,mask_to_regions,load_nifti,patches
from neurotumourai.train import train_segmentation
class SegDataset(Dataset):
 def __init__(self,manifest,patch_size=128):
  self.data=[]
  for row in read_manifest(manifest):
   if not row.get('seg'):raise ValueError('Training segmentation requires segmentation labels')
   v,_,_=prep_volumes(row);m=mask_to_regions(load_nifti(row['seg'])[0]);self.data.extend((torch.from_numpy(x.copy()),torch.from_numpy(y.copy())) for x,y,_ in patches(v,m,size=patch_size,stride=patch_size,min_fraction=0))
 def __len__(self):return len(self.data)
 def __getitem__(self,i):return self.data[i]
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--val',required=True);p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=100);p.add_argument('--patch-size',type=int,default=128);p.add_argument('--batch',type=int,default=1);a=p.parse_args()
 train=SegDataset(a.train,a.patch_size);val=SegDataset(a.val,a.patch_size);train_segmentation(DataLoader(train,batch_size=a.batch,shuffle=True),DataLoader(val,batch_size=a.batch),a.output,a.epochs)
