#!/usr/bin/env python3
"""Paired subject-level model comparison from columns subject_id,label,prob_hgg."""
import argparse,csv,json
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

def records(path):
 with open(path,newline='') as f:return {r['subject_id']:r for r in csv.DictReader(f)}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--comparison',required=True);p.add_argument('--output',required=True);p.add_argument('--bootstrap',type=int,default=1000);a=p.parse_args();r=records(a.reference);c=records(a.comparison);ids=sorted(set(r)&set(c))
 if set(r)!=set(c):raise ValueError('Paired tests require identical subjects')
 y=np.array([int(r[s]['label']=='HGG') for s in ids]);p1=np.array([float(r[s]['prob_hgg']) for s in ids]);p2=np.array([float(c[s]['prob_hgg']) for s in ids]);assert all(r[s]['label']==c[s]['label'] for s in ids)
 hit1=(p1>=.5)==y;hit2=(p2>=.5)==y;b=int(np.sum(hit1 & ~hit2));d=int(np.sum(~hit1 & hit2));pv=binomtest(min(b,d),b+d,.5).pvalue if b+d else 1.
 rng=np.random.default_rng(42);diff=[]
 for _ in range(a.bootstrap):
  ix=rng.integers(0,len(ids),len(ids));diff.append(float(np.mean(hit1[ix])-np.mean(hit2[ix])))
 result={'n_paired_subjects':len(ids),'reference_accuracy':float(hit1.mean()),'comparison_accuracy':float(hit2.mean()),'accuracy_difference':float(hit1.mean()-hit2.mean()),'difference_bootstrap_95ci':np.percentile(diff,[2.5,97.5]).tolist(),'exact_mcnemar_p':pv,'discordant_counts':[b,d]}
 with open(a.output,'w') as f:json.dump(result,f,indent=2)
 print(json.dumps(result,indent=2))
