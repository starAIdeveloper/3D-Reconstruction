# 3D Reconstruction Workbench
Native Python/Qt application inspired by the supplied photogrammetry workflow image. It performs real **calibrated sparse two-view reconstruction**: SIFT features, descriptor matching, RANSAC essential matrix, relative camera pose, triangulation, positive-depth filtering, reprojection checks, image colors, and point-cloud export.

## Install and run
Python 3.11 or 3.12 recommended. Create a virtual environment, then `pip install -r requirements.txt`. Run `python run.py` for the desktop application.

Import photos, select exactly two using Ctrl-click, enter their known focal length in pixels, and click Reconstruct selected pair. Tabs show input, features, verified matches, an orbitable colored cloud, and a measured quality report. Drag the cloud to orbit and scroll to zoom. Export writes an ASCII PLY point cloud plus camera/calibration JSON.

CLI: `python -m recon.cli left.png right.png --focal 1000 --output cloud.ply`.

## Input and calibration
Use two overlapping photos of a static textured subject with lateral camera movement and consistent lighting. Both images must share dimensions and calibration. Supply undistorted images. The current calibration model assumes fx=fy, principal point at the image center, and no distortion. Focal length must be calibrated at the input resolution; resizing requires scaling focal length. Images larger than 4000 pixels on their longest side are rejected to bound feature processing.

Camera 1 defines the world frame. Camera 2 is represented by world-to-camera R,t. The estimated translation has unit length, so distances are **arbitrary scale**, not meters. The JSON includes the second camera center, -R.T @ t. Reprojection errors do not establish absolute geometric accuracy. Flat scenes, pure rotation, repeated patterns, small baselines, and wrong calibration can produce failures or unreliable shapes.

## Scope and limitations
This is a two-view sparse workbench, not a full SLAM or dense photogrammetry engine. It does not register an entire image collection, perform bundle adjustment, estimate dense depth, generate meshes, or build texture atlases. Multiple imported files allow choosing pairs; their results are not merged. The supplied reference artwork is not included as a reconstructed model. No metric accuracy or production-quality reconstruction is claimed.

## Tests
`python -m unittest discover -v` runs nine algorithm tests, including known-geometry camera recovery and an end-to-end generated stereo pair. `QT_QPA_PLATFORM=offscreen python -m tests.gui_check` checks the actual Qt UI, background processing, orbit interaction, and exports. On Windows, use the normal Qt platform or set the environment variable with the syntax for your shell.

The generated test pair reconstructs 817 sparse points in the validation run. Screenshots depict that generated fixture and are labeled through the report; they are not results from real photographs. Real-scene accuracy and Windows packaging remain untested. See docs/validation.json.

## History
Separate implementation commits record the work as completed, without backdated dates. artifacts/3D-Reconstruction.bundle preserves the original local history. GitHub commits import those stages in order.
