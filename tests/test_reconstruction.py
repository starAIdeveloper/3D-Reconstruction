import unittest,tempfile,json
from pathlib import Path
import cv2,numpy as np
from recon.geometry import intrinsics,triangulate,recover
from recon.pipeline import reconstruct,read_image,export
from tests.fixtures import make_pair

class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.K=intrinsics(800,600,700);rng=np.random.default_rng(2);self.points=rng.uniform([-2,-1,5],[2,1,12],(100,3));self.R=np.eye(3);self.t=np.array([-.6,0,0])
        def project(p):q=p@self.K.T;return q[:,:2]/q[:,2,None]
        self.a=project(self.points);self.b=project(self.points+self.t)
    def test_triangulation_exact(self):
        p,mask,e=triangulate(self.a,self.b,self.K,self.R,self.t);np.testing.assert_allclose(p,self.points,atol=1e-8);self.assertTrue(mask.all());self.assertLess(e.max(),1e-7)
    def test_recovered_camera(self):
        p,R,t,ids,e=recover(self.a,self.b,self.K);self.assertGreater(len(p),80);np.testing.assert_allclose(R,np.eye(3),atol=1e-4);self.assertLess(t[0,0],-.99)
    def test_reject_negative_depth(self):
        p,mask,e=triangulate(self.a,self.b,self.K,self.R,-self.t);self.assertEqual(len(p),0)
    def test_intrinsics_validation(self):
        for f in [0,-1,float('nan')]:
            with self.assertRaises(ValueError):intrinsics(800,600,f)
    def test_insufficient_matches(self):
        with self.assertRaises(ValueError):recover(self.a[:5],self.b[:5],self.K)
    def test_mismatched_points(self):
        with self.assertRaises(ValueError):triangulate(self.a,self.b[:4],self.K,self.R,self.t)
    def test_image_pipeline_export(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=make_pair(d);r=reconstruct(a,b,700);self.assertGreater(len(r.points),30);self.assertLess(r.report['median_reprojection_error_px'],2);out=Path(d)/'cloud.ply';export(r,out);self.assertIn(f'element vertex {len(r.points)}',out.read_text());report=json.loads(out.with_suffix('.json').read_text());self.assertEqual(report['accepted_points'],len(r.points));self.assertEqual(r.colors.shape,r.points.shape)
    def test_bad_file(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.png';p.write_text('bad');
            with self.assertRaises(ValueError):read_image(p)
    def test_textureless_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'blank.png';cv2.imwrite(str(p),np.zeros((60,80,3),dtype=np.uint8))
            with self.assertRaises(ValueError):reconstruct(p,p,70)
if __name__=='__main__':unittest.main()
