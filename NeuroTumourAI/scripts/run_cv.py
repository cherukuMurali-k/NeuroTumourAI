#!/usr/bin/env python3
"""Run full five-fold classifier training on precomputed, fold-isolated caches."""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from neurotumourai.train import train_classifier,load_classifier,predict
from neurotumourai.evaluation import subject_aggregate,metrics
from sklearn.model_selection import StratifiedKFold

def read(path):
 with open(path,newline='') as f:return list(csv.DictReader(f))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--features',required=True,help='All development patches, prepared without leakage; recommended to train distinct segmenter for each fold');p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=200);p.add_argument('--batch',type=int,default=1);a=p.parse_args();r=read(a.features);ids=sorted(set(x['subject_id'] for x in r));lookup={s:next(x['label'] for x in r if x['subject_id']==s) for s in ids};labels=[lookup[s] for s in ids];out=Path(a.output);out.mkdir(parents=True,exist_ok=True);results=[]
 for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(ids,labels),1):
  train_ids={ids[i] for i in tr};val_ids={ids[i] for i in va};train=[v for v in r if v['subject_id'] in train_ids];val=[v for v in r if v['subject_id'] in val_ids];dest=out/f'fold_{fold}';train_classifier(train,val,dest,epochs=a.epochs,batch_size=a.batch);model=load_classifier(dest/'best.pt');log,y,subjects=predict(model,val,a.batch);agg=subject_aggregate(log,y,subjects);result=metrics([v['label'] for v in agg.values()],[v['prob'] for v in agg.values()]);results.append(result);print('fold',fold,result)
 summary={'folds':results,'mean_sd':{k:{'mean':float(np.mean([v[k] for v in results])),'sd':float(np.std([v[k] for v in results],ddof=1))} for k in ['accuracy','macro_f1','auroc']}};(out/'summary.json').write_text(json.dumps(summary,indent=2))
