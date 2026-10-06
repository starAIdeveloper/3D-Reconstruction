from dataclasses import dataclass
from pathlib import Path
import json
import cv2
import numpy as np
from .geometry import intrinsics,recover

@dataclass
class Result:
    points:np.ndarray
    colors:np.ndarray
    R:np.ndarray
    t:np.ndarray
    K:np.ndarray
    matches_image:np.ndarray
    feature_image:np.ndarray
    report:dict

def read_image(path):
    raw=np.fromfile(path,dtype=np.uint8);image=cv2.imdecode(raw,cv2.IMREAD_COLOR)
    if image is None:raise ValueError(f'Cannot read image: {Path(path).name}')
    return image

def reconstruct(first,second,focal,progress=lambda x:None):
    progress('Loading selected image pair');a=read_image(first);b=read_image(second)
    if a.shape!=b.shape:raise ValueError('Images must share dimensions and calibration; resize them consistently before import')
    h,w=a.shape[:2]
    if max(h,w)>4000:raise ValueError('Resize images to at most 4000 pixels on the longest side and scale focal length accordingly')
    K=intrinsics(w,h,focal);progress('Extracting SIFT features')
    sift=cv2.SIFT_create(nfeatures=8000);ka,da=sift.detectAndCompute(cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),None);kb,db=sift.detectAndCompute(cv2.cvtColor(b,cv2.COLOR_BGR2GRAY),None)
    if da is None or db is None or len(db)<2:raise ValueError('Not enough features in the image pair')
    progress('Matching descriptors and estimating camera pose')
    pairs=cv2.BFMatcher().knnMatch(da,db,k=2);matches=[p[0] for p in pairs if len(p)==2 and p[0].distance<.72*p[1].distance]
    # Keep one best match per destination descriptor.
    best={}
    for m in sorted(matches,key=lambda m:m.distance):best.setdefault(m.trainIdx,m)
    matches=list(best.values());pa=np.array([ka[m.queryIdx].pt for m in matches]);pb=np.array([kb[m.trainIdx].pt for m in matches])
    points,R,t,indices,errors=recover(pa,pb,K);progress('Filtering triangulated cloud')
    pixels=np.rint(pa[indices]).astype(int);pixels[:,0]=np.clip(pixels[:,0],0,w-1);pixels[:,1]=np.clip(pixels[:,1],0,h-1)
    colors=a[pixels[:,1],pixels[:,0],::-1].copy()
    preview=cv2.drawMatches(a,ka,b,kb,[matches[i] for i in indices[:150]],None,flags=2)
    features=cv2.drawKeypoints(a,ka,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    report={'mode':'calibrated sparse two-view reconstruction','images':2,'features_first':len(ka),'features_second':len(kb),'ratio_matches':len(matches),'accepted_points':len(points),'median_reprojection_error_px':float(np.median(errors)),'scale':'arbitrary; translation normalized to one','distortion':'assumed zero; use undistorted input','dense_mesh':False}
    return Result(points,colors,R,t,K,preview,features,report)

def export(result,path):
    path=Path(path)
    lines=['ply','format ascii 1.0',f'element vertex {len(result.points)}','property float x','property float y','property float z','property uchar red','property uchar green','property uchar blue','end_header']
    lines += [' '.join([*(f'{v:.8g}' for v in xyz),*(str(int(c)) for c in rgb)]) for xyz,rgb in zip(result.points,result.colors)]
    path.write_text('\n'.join(lines)+'\n')
    path.with_suffix('.json').write_text(json.dumps({**result.report,'intrinsics':result.K.tolist(),'world_to_second_camera':{'R':result.R.tolist(),'t':result.t.tolist()},'second_camera_center':(-result.R.T@result.t).ravel().tolist()},indent=2))
