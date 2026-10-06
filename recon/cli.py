import argparse,json
from .pipeline import reconstruct,export

def main():
    p=argparse.ArgumentParser(description='Calibrated sparse two-view reconstruction (arbitrary scale)');p.add_argument('first');p.add_argument('second');p.add_argument('--focal',type=float,required=True);p.add_argument('--output',default='reconstruction.ply');a=p.parse_args()
    try:r=reconstruct(a.first,a.second,a.focal,print);export(r,a.output);print(json.dumps(r.report,indent=2))
    except Exception as e:p.exit(1,f'Failed: {e}\n')
if __name__=='__main__':main()
