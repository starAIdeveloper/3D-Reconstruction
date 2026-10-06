from pathlib import Path
import cv2,numpy as np

def make_pair(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(44);a=np.full((600,800,3),25,dtype=np.uint8);b=a.copy()
    for y in range(70,540,55):
        for x in range(110,750,55):
            depth=rng.uniform(5,12);shift=round(700*.6/depth);patch=rng.integers(35,245,(23,23,3),dtype=np.uint8);patch=cv2.GaussianBlur(patch,(3,3),.5)
            a[y-11:y+12,x-11:x+12]=patch;b[y-11:y+12,x-shift-11:x-shift+12]=patch
    paths=[folder/'synthetic-left.png',folder/'synthetic-right.png']
    for p,img in zip(paths,[a,b]):cv2.imwrite(str(p),img)
    return paths
