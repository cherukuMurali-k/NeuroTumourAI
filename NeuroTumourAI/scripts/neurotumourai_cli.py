#!/usr/bin/env python3
"""Run with: python scripts/neurotumourai_cli.py <command> [options]"""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from neurotumourai.data import read_manifest,create_folds,ensure_disjoint,prep_volumes,mask_to_regions,patches,load_nifti,NylStandardizer
from neurotumourai.features import radiomic_features,morphometrics,build_graph
from neurotumourai.train import train_classifier,predict,load_classifier,train_segmentation
from neurotumourai.evaluation import subject_aggregate,metrics,ece,TemperatureScaler,bootstrap_ci,save_figures

def save_json(obj,path):Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(obj,indent=2))
def command_folds(a):
 rows=read_manifest(a.manifest);external=read_manifest(a.external) if a.external else [];ensure_disjoint(rows,external)
 folds=create_folds(rows,a.folds,a.seed);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 for i,(t,v) in enumerate(folds,1):
  for name,ind in [('train',t),('val',v)]:
   with open(out/f'fold{i}_{name}.csv','w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows([rows[j] for j in ind])
 print('Created',len(folds),'stratified subject-wise folds. Cohort overlap checked.')
def command_prepare(a):
 rows=read_manifest(a.manifest);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 # First pre-normalize training volumes and fit Nyul landmarks ONLY on supplied training manifest.
 prepared=[]
 for row in rows:
  v,aff,sp=prep_volumes(row,n4=a.n4,register=a.register);prepared.append((row,v,sp))
 nyul=NylStandardizer().fit(v for _,v,_ in prepared) if a.fit_nyul else NylStandardizer.load(a.nyul)
 if a.fit_nyul:nyul.save(out/'nyul.npz')
 records=[];feature_names=json.loads(Path(a.feature_schema).read_text()) if a.feature_schema else None
 for row,v,sp in prepared:
  v=nyul.transform(v);raw,_,_=load_nifti(row['seg']) if row.get('seg') else (None,None,None)
  if raw is None:
   if not a.segmenter:raise ValueError('Provide trained --segmenter for held-out/external records (never use reference masks).')
   from neurotumourai.models import UNet3D
   segmenter=UNet3D();ck=torch.load(a.segmenter,map_location='cpu',weights_only=False);segmenter.load_state_dict(ck['model']);segmenter.eval()
   with torch.no_grad():r=segmenter(torch.tensor(v[None])).sigmoid().numpy()[0];reg=(r>.5).astype(np.float32)
  else:
   if not a.allow_training_masks:raise ValueError('Reference masks are prohibited for held-out cases. Use --segmenter.')
   reg=mask_to_regions(raw)
  for j,(p,m,pos) in enumerate(patches(v,reg,size=a.patch_size,stride=a.stride,min_fraction=a.min_fraction)):
   radi,fn=radiomic_features(p,m[0],feature_names)
   if feature_names is None:
    feature_names=fn[:a.rad_dim]
    radi,fn=radiomic_features(p,m[0],feature_names)
   radi=radi[:a.rad_dim];radi=np.pad(radi,(0,max(0,a.rad_dim-len(radi))))
   mor=morphometrics(m[0],sp);graph=build_graph(p,m[0]);key=f'{row["subject_id"]}_p{j:04d}';np.savez_compressed(out/(key+'.npz'),image=p,radiomics=radi,morphometrics=mor,graph_x=graph['x'],graph_edge=graph['edge_index']);records.append({'subject_id':row['subject_id'],'label':row['label'],'npz':str((out/(key+'.npz')).resolve())})
 if feature_names is not None:(out/'radiomics_features.json').write_text(json.dumps(feature_names,indent=2))
 with open(out/'features.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['subject_id','label','npz']);w.writeheader();w.writerows(records)
 print('Prepared',len(records),'patches from',len(rows),'subjects')
def feature_records(path):
 with open(path,newline='') as f:return list(csv.DictReader(f))
def command_train(a):
 train_classifier(feature_records(a.train),feature_records(a.val),a.output,epochs=a.epochs,patience=a.patience,batch_size=a.batch,grad_accum=a.accum,device=a.device)
def command_evaluate(a):
 records=feature_records(a.features);device=a.device or ('cuda' if torch.cuda.is_available() else 'cpu');model=load_classifier(a.model,device);logits,labels,ids=predict(model,records,a.batch,device);agg=subject_aggregate(logits,labels,ids);y=np.array([v['label'] for v in agg.values()]);p=np.stack([v['prob'] for v in agg.values()]);m=metrics(y,p);m['ece']=ece(y,p)
 if a.temperature:
  t=float(json.loads(Path(a.temperature).read_text())['temperature']);from scipy.special import softmax
  # temperature MUST act on aggregated subject logits, not per-patch confidence.
  sublog=np.log(np.maximum(p,1e-9));p=softmax(sublog/t,axis=1);m['calibrated']=metrics(y,p);m['calibrated_ece']=ece(y,p)
 m['ece_bootstrap_95ci']=bootstrap_ci(y,p,'ece',a.bootstrap);save_json(m,Path(a.output)/'metrics.json');save_figures(y,p,a.output);print(json.dumps(m,indent=2))
def command_calibrate(a):
 import scipy.special
 records=feature_records(a.features);device=a.device or ('cuda' if torch.cuda.is_available() else 'cpu');model=load_classifier(a.model,device);logits,y,ids=predict(model,records,a.batch,device);agg=subject_aggregate(logits,y,ids);labels=np.array([v['label'] for v in agg.values()]);probs=np.stack([v['prob'] for v in agg.values()]);cal=TemperatureScaler().fit(np.log(np.maximum(probs,1e-9)),labels);save_json({'temperature':cal.temperature,'source':'held-out development predictions, subject-aggregated'},a.output);print('Temperature',cal.temperature)
def command_smoke(a):
 from neurotumourai.models import HybridTumourNet,UNet3D
 from neurotumourai.losses import CompositeLoss
 n=2;im=torch.randn(n,4,16,16,16);graphs=[{'x':torch.randn(4,8),'edge_index':torch.tensor([[0,1,2,3],[1,2,3,0]])} for _ in range(n)];model=HybridTumourNet(encoder_base=8);out=model(im,torch.randn(n,128),torch.randn(n,14),graphs,True);loss,_=CompositeLoss()(out,torch.tensor([0,1]));loss.backward();assert out['logits'].shape==(2,2);assert UNet3D(base=4)(im).shape==(2,3,16,16,16);print('PASS model forward/backward and segmentation smoke')
def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
 def common(name):return sub.add_parser(name)
 x=common('folds');x.add_argument('--manifest',required=True);x.add_argument('--external');x.add_argument('--output',required=True);x.add_argument('--folds',type=int,default=5);x.add_argument('--seed',type=int,default=42);x.set_defaults(func=command_folds)
 x=common('prepare');x.add_argument('--manifest',required=True);x.add_argument('--output',required=True);x.add_argument('--fit-nyul',action='store_true');x.add_argument('--nyul');x.add_argument('--n4',action='store_true');x.add_argument('--register',action='store_true');x.add_argument('--segmenter');x.add_argument('--allow-training-masks',action='store_true');x.add_argument('--patch-size',type=int,default=128);x.add_argument('--stride',type=int,default=64);x.add_argument('--min-fraction',type=float,default=.1);x.add_argument('--rad-dim',type=int,default=128);x.add_argument('--feature-schema',help='JSON feature names from fold training cache, required for validation and TCGA');x.set_defaults(func=command_prepare)
 x=common('train');x.add_argument('--train',required=True);x.add_argument('--val',required=True);x.add_argument('--output',required=True);x.add_argument('--epochs',type=int,default=200);x.add_argument('--patience',type=int,default=20);x.add_argument('--batch',type=int,default=1);x.add_argument('--accum',type=int,default=4);x.add_argument('--device');x.set_defaults(func=command_train)
 for name,func in [('evaluate',command_evaluate),('calibrate',command_calibrate)]:
  x=common(name);x.add_argument('--model',required=True);x.add_argument('--features',required=True);x.add_argument('--output',required=True);x.add_argument('--batch',type=int,default=1);x.add_argument('--device');x.set_defaults(func=func)
  if name=='evaluate':x.add_argument('--temperature');x.add_argument('--bootstrap',type=int,default=1000)
 x=common('smoke');x.set_defaults(func=command_smoke)
 a=p.parse_args();a.func(a)
if __name__=='__main__':main()
