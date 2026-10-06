import numpy as np
import cv2

def intrinsics(width,height,focal):
    if width<=0 or height<=0 or not np.isfinite(focal) or focal<=0:raise ValueError('Positive image dimensions and focal length required')
    return np.array([[focal,0,width/2],[0,focal,height/2],[0,0,1]],dtype=float)

def triangulate(a,b,K,R,t,max_error=2.):
    a=np.asarray(a,dtype=float).reshape(-1,2);b=np.asarray(b,dtype=float).reshape(-1,2)
    if len(a)!=len(b):raise ValueError('Correspondence count mismatch')
    if len(a)==0:return np.empty((0,3)),np.zeros(0,dtype=bool),np.empty(0)
    P=K@np.c_[np.eye(3),np.zeros(3)];Q=K@np.c_[R,np.asarray(t).reshape(3)]
    h=cv2.triangulatePoints(P,Q,a.T,b.T).T
    valid=np.abs(h[:,3])>1e-10;points=np.full((len(a),3),np.nan);points[valid]=h[valid,:3]/h[valid,3,None]
    other=(R@points.T).T+np.asarray(t).reshape(3)
    def project(points):
        q=(K@points.T).T
        with np.errstate(divide='ignore',invalid='ignore'):return q[:,:2]/q[:,2,None]
    errors=np.maximum(np.linalg.norm(project(points)-a,axis=1),np.linalg.norm(project(other)-b,axis=1))
    mask=valid&np.isfinite(points).all(axis=1)&(points[:,2]>0)&(other[:,2]>0)&(errors<=max_error)
    return points[mask],mask,errors[mask]

def recover(a,b,K):
    if len(a)<12:raise ValueError('At least 12 matched features are required')
    E,mask=cv2.findEssentialMat(np.asarray(a,dtype=float),np.asarray(b,dtype=float),K,method=cv2.RANSAC,prob=.999,threshold=1.)
    if E is None or E.shape!=(3,3):raise ValueError('Pose estimation failed; use overlapping images with lateral camera motion')
    count,R,t,mask=cv2.recoverPose(E,np.asarray(a,dtype=float),np.asarray(b,dtype=float),K,mask=mask)
    keep=mask.ravel()!=0
    if count<12:raise ValueError('Insufficient pose inliers; avoid pure rotation and flat or textureless subjects')
    points,good,error=triangulate(np.asarray(a)[keep],np.asarray(b)[keep],K,R,t)
    indices=np.flatnonzero(keep)[good]
    if len(points)<12:raise ValueError('Too few points passed positive-depth and reprojection checks')
    return points,R,t,indices,error
