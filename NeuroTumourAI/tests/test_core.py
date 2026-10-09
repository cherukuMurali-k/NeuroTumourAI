import numpy as np
import torch
from neurotumourai.models import UNet3D,HybridTumourNet,regions_from_logits
from neurotumourai.losses import CompositeLoss
from neurotumourai.data import NylStandardizer,patches,create_folds,ensure_disjoint
from neurotumourai.features import morphometrics,build_graph
from neurotumourai.evaluation import subject_aggregate,metrics,ece,TemperatureScaler

def test_models():
 x=torch.randn(2,4,16,16,16);s=UNet3D(base=4)(x);assert s.shape==(2,3,16,16,16)
 graphs=[{'x':torch.randn(3,8),'edge_index':torch.tensor([[0,1],[1,2]])} for _ in range(2)];m=HybridTumourNet(encoder_base=8);o=m(x,torch.randn(2,128),torch.randn(2,14),graphs,True);loss,_=CompositeLoss()(o,torch.tensor([0,1]));loss.backward();assert m.head[-1].weight.grad is not None

def test_feature_graph():
 a=np.zeros((16,16,16),bool);a[3:12,3:12,3:12]=True;f=morphometrics(a);assert f.shape==(14,) and f[0]>0
 g=build_graph(np.random.default_rng(0).normal(size=(4,16,16,16)).astype('float32'),a,segments=8);assert g['x'].shape[1]==8 and g['edge_index'].shape[0]==2

def test_patch_and_nyul():
 a=np.ones((4,12,12,12),np.float32);m=np.ones((3,12,12,12),np.float32);assert len(patches(a,m,16,8))>=1
 n=NylStandardizer().fit([a]);assert n.transform(a).shape==a.shape

def test_fold_isolation():
 rows=[{'subject_id':str(i),'label':'HGG' if i%2 else 'LGG'} for i in range(20)];splits=create_folds(rows);assert len(splits)==5
 for tr,va in splits:assert not(set(tr)&set(va))

def test_eval():
 a=subject_aggregate(np.array([[2.,1.],[1.,2.],[2.,1.],[1.,2.]]),[0,1,0,1],['a','b','a','b']);y=[v['label'] for v in a.values()];p=np.stack([v['prob'] for v in a.values()]);assert metrics(y,p)['accuracy']==1;assert np.isfinite(ece(y,p));assert TemperatureScaler().fit(np.log(p),y).temperature>0
