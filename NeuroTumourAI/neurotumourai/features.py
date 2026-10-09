"""Handcrafted features and 3D SLIC graph construction. Train-only scaling is external."""
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import slic
from skimage.measure import marching_cubes,mesh_surface_area
from scipy.spatial import ConvexHull

def morphometrics(mask,spacing=(1,1,1)):
 m=mask.astype(bool);coords=np.argwhere(m)*np.array(spacing)[None]
 if len(coords)<4:return np.zeros(14,np.float32)
 vol=float(len(coords)*np.prod(spacing));mesh=m.astype(np.uint8)
 try:
  verts,faces,_,_=marching_cubes(np.pad(mesh,1),.5,spacing=spacing);area=float(mesh_surface_area(verts,faces));surface=verts[::max(1,len(verts)//300)]
 except ValueError:area=0.;surface=coords[::max(1,len(coords)//300)]
 eigen=np.maximum(np.linalg.eigvalsh(np.cov(coords.T)),1e-8)[::-1];axes=4*np.sqrt(eigen);diam=float(np.linalg.norm(surface.max(0)-surface.min(0))) if len(surface)>1 else 0.
 radius=(3*vol/(4*np.pi))**(1/3);sphericity=float(np.pi**(1/3)*(6*vol)**(2/3)/area) if area else 0.;compact=float(vol/(area**1.5)) if area else 0.
 try:convex=float(ConvexHull(coords[::max(1,len(coords)//10000)]).volume);solid=min(vol/convex,1.) if convex else 0.
 except Exception:solid=0.
 sizes=np.array([2,4,8,16]);counts=[]
 for s in sizes:
  sh=tuple((np.array(m.shape)+s-1)//s);occupied=np.zeros(sh,dtype=bool);inds=np.argwhere(m)//s;occupied[tuple(inds.T)]=True;counts.append(max(occupied.sum(),1))
 fract=float(-np.polyfit(np.log(sizes),np.log(counts),1)[0]);return np.nan_to_num(np.array([vol,area,area/vol if vol else 0,sphericity,compact,np.sqrt(eigen[1]/eigen[0]),np.sqrt(eigen[2]/eigen[0]),diam,*axes,2*radius,fract,solid],np.float32))

def radiomic_features(image,mask,feature_names=None):
 """Deterministic PyRadiomics extraction; caller supplies selection order saved during fitting."""
 try:import radiomics.featureextractor as pyr;import SimpleITK as sitk
 except ImportError as e:raise ImportError('Install pyradiomics and SimpleITK for real radiomics') from e
 ext=pyr.RadiomicsFeatureExtractor(binWidth=25,normalize=False);ext.disableAllFeatures()
 for key in ['firstorder','glcm','glrlm','glszm','ngtdm']:ext.enableFeatureClassByName(key)
 result={}
 for c in range(image.shape[0]):
  sitim=sitk.GetImageFromArray(image[c].astype(np.float32));sitmask=sitk.GetImageFromArray(mask.astype(np.uint8));sitmask.CopyInformation(sitim)
  if mask.sum()<10:continue
  features=ext.execute(sitim,sitmask,label=1)
  for name,val in features.items():
   if name.startswith('original_') and np.isscalar(val):
    try:result[f'm{c}_{name}']=float(val)
    except (TypeError,ValueError):pass
 if feature_names is None:feature_names=sorted(result)
 return np.nan_to_num(np.array([result.get(k,np.nan) for k in feature_names],np.float32),nan=0.,posinf=0.,neginf=0.),feature_names

def build_graph(image,mask,segments=96,compactness=.1,node_dim=8):
 """3D SLIC, undirected neighboring supervoxels, local intensity+geometry properties."""
 mask=mask.astype(bool)
 if mask.sum()<8:return {'x':np.zeros((1,node_dim),np.float32),'edge_index':np.empty((2,0),np.int64)}
 # SLIC on downsampled ROI is necessary for practical CPU processing at 128^3.
 coords=np.argwhere(mask);lo=np.maximum(coords.min(0)-2,0);hi=np.minimum(coords.max(0)+3,mask.shape);sl=tuple(slice(a,b) for a,b in zip(lo,hi));roi=image[(slice(None),)+sl];roi_mask=mask[sl]
 fac=max(1,int(np.ceil(max(roi_mask.shape)/64)));vol=roi[:,::fac,::fac,::fac];mm=roi_mask[::fac,::fac,::fac]
 if not mm.any():return {'x':np.zeros((1,node_dim),np.float32),'edge_index':np.empty((2,0),np.int64)}
 labels=slic(np.moveaxis(vol,0,-1),n_segments=min(segments,int(mm.sum())),compactness=compactness,start_label=1,mask=mm,channel_axis=-1,spacing=(1,1,1))
 ids=np.unique(labels[mm]);ids=ids[ids>0];mapping={int(v):i for i,v in enumerate(ids)};nodes=[]
 for v in ids:
  coords=np.argwhere(labels==v);vals=vol[:,labels==v];mean=vals.mean(1);std=vals.std(1);props=np.r_[mean,std,coords.mean(0)/np.maximum(np.array(labels.shape)-1,1),len(coords)/max(mm.sum(),1)];nodes.append(np.pad(props[:node_dim],(0,max(0,node_dim-len(props)))) )
 edges=set()
 for axis in range(3):
  a=np.take(labels,range(labels.shape[axis]-1),axis=axis);b=np.take(labels,range(1,labels.shape[axis]),axis=axis);different=(a!=b)&(a>0)&(b>0)
  for u,v in zip(a[different],b[different]):
   if int(u) in mapping and int(v) in mapping:edges.add((mapping[int(u)],mapping[int(v)]));edges.add((mapping[int(v)],mapping[int(u)]))
 return {'x':np.array(nodes,np.float32),'edge_index':np.array(sorted(edges),dtype=np.int64).reshape(-1,2).T.copy() if edges else np.empty((2,0),np.int64)}
