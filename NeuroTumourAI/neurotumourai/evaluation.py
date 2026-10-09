import json
import numpy as np
import torch
from scipy.special import softmax
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,roc_auc_score,confusion_matrix,brier_score_loss,roc_curve,precision_recall_curve

def subject_aggregate(logits,labels,ids):
 grouped={}
 for logit,y,s in zip(logits,labels,ids):
  if s in grouped and grouped[s]['label']!=int(y):raise ValueError('Inconsistent subject labels')
  grouped.setdefault(s,{'probabilities':[],'label':int(y)})['probabilities'].append(softmax(logit))
 return {s:{'prob':np.mean(v['probabilities'],axis=0),'label':v['label']} for s,v in grouped.items()}
def metrics(y,probs):
 y=np.asarray(y,dtype=int);p=np.asarray(probs,dtype=float);pred=p.argmax(1);precision,recall,f1,_=precision_recall_fscore_support(y,pred,labels=[0,1],average='macro',zero_division=0);cm=confusion_matrix(y,pred,labels=[0,1]);tn,fp,fn,tp=cm.ravel();d={'accuracy':float(accuracy_score(y,pred)),'macro_precision':float(precision),'macro_recall':float(recall),'macro_f1':float(f1),'sensitivity':float(tp/max(tp+fn,1)),'specificity':float(tn/max(tn+fp,1)),'brier':float(brier_score_loss(y,p[:,1])),'confusion_matrix':cm.tolist(),'n_subjects':len(y)}
 d['auroc']=float(roc_auc_score(y,p[:,1])) if len(np.unique(y))==2 else None;return d
def ece(y,p,bins=15):
 y=np.asarray(y);p=np.asarray(p);conf=p.max(1);correct=(p.argmax(1)==y);edges=np.linspace(0,1,bins+1);value=0.
 for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
  mask=(conf>=lo)&(conf<=hi if i==bins-1 else conf<hi)
  if mask.any():value+=float(mask.mean())*abs(float(correct[mask].mean())-float(conf[mask].mean()))
 return float(value)
class TemperatureScaler:
 def __init__(self):self.temperature=1.
 def fit(self,logits,y):
  logits=np.asarray(logits);y=np.asarray(y,dtype=int)
  def nll(logt):
   p=softmax(logits/np.exp(logt),axis=1);return -np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()
  result=minimize_scalar(nll,bounds=(-3,3),method='bounded');self.temperature=float(np.exp(result.x));return self
 def predict(self,logits):return softmax(np.asarray(logits)/self.temperature,axis=1)
def bootstrap_ci(y,probs,metric='accuracy',n=1000,seed=42):
 rng=np.random.default_rng(seed);y=np.asarray(y);probs=np.asarray(probs);scores=[]
 for _ in range(n):
  ind=rng.integers(0,len(y),len(y));scores.append(ece(y[ind],probs[ind]) if metric=='ece' else metrics(y[ind],probs[ind]).get(metric))
 a=np.array([s for s in scores if s is not None]);return np.percentile(a,[2.5,97.5]).tolist() if a.size else [None,None]
def save_figures(y,p,out):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from pathlib import Path
 out=Path(out);out.mkdir(parents=True,exist_ok=True);y=np.asarray(y);p=np.asarray(p)
 if len(np.unique(y))==2:
  fpr,tpr,_=roc_curve(y,p[:,1]);fig,ax=plt.subplots();ax.plot(fpr,tpr,label=f'AUROC {roc_auc_score(y,p[:,1]):.3f}');ax.plot([0,1],[0,1],'--');ax.set(xlabel='False Positive Rate',ylabel='True Positive Rate',title='Subject-Level ROC');ax.legend();fig.tight_layout();fig.savefig(out/'roc.png',dpi=300);plt.close(fig)
  precision,recall,_=precision_recall_curve(y,p[:,1]);fig,ax=plt.subplots();ax.plot(recall,precision);ax.set(xlabel='Recall',ylabel='Precision',title='Precision–Recall');fig.tight_layout();fig.savefig(out/'pr.png',dpi=300);plt.close(fig)
 cm=confusion_matrix(y,p.argmax(1),labels=[0,1]);fig,ax=plt.subplots();ax.imshow(cm);ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['LGG','HGG'],yticklabels=['LGG','HGG'],xlabel='Predicted',ylabel='True',title='Confusion Matrix')
 for i in range(2):
  for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center')
 fig.tight_layout();fig.savefig(out/'confusion.png',dpi=300);plt.close(fig)
 bins=np.linspace(0,1,16);confidence=p.max(1);acc=[];mean=[]
 for a,b in zip(bins[:-1],bins[1:]):
  subset=(confidence>=a)&(confidence<=b if b==1 else confidence<b)
  if subset.any():acc.append(np.mean(p.argmax(1)[subset]==y[subset]));mean.append(np.mean(confidence[subset]))
 fig,ax=plt.subplots();ax.plot([0,1],[0,1],'--');ax.plot(mean,acc,'o-');ax.set(xlabel='Mean Confidence',ylabel='Observed Accuracy',title='Reliability Diagram',xlim=(0,1),ylim=(0,1));fig.tight_layout();fig.savefig(out/'reliability.png',dpi=300);plt.close(fig)
