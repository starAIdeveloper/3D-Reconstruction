import json,math
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt,QThread,Signal,QPointF
from PySide6.QtGui import QPainter,QColor,QImage,QPixmap,QPen
from PySide6.QtWidgets import QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QListWidget,QDoubleSpinBox,QFileDialog,QSplitter,QTabWidget,QTextEdit,QMessageBox
from .pipeline import reconstruct,export,read_image

class CloudView(QWidget):
    def __init__(self):
        super().__init__();self.points=np.empty((0,3));self.colors=np.empty((0,3));self.cameras=[];self.yaw=.5;self.pitch=-.25;self.zoom=1.;self.last=None;self.setMinimumSize(300,240)
    def set_result(self,result):
        self.center=np.median(result.points,axis=0);self.scale=max(np.percentile(np.linalg.norm(result.points-self.center,axis=1),90),.01)
        self.points=(result.points-self.center)/self.scale;self.colors=result.colors
        self.cameras=[-self.center/self.scale,(-result.R.T@result.t).ravel()/self.scale-self.center/self.scale];self.update()
    def project(self,points):
        cy,sy=math.cos(self.yaw),math.sin(self.yaw);cp,sp=math.cos(self.pitch),math.sin(self.pitch)
        R=np.array([[cy,0,sy],[sp*sy,cp,-sp*cy],[-cp*sy,sp,cp*cy]])
        q=np.asarray(points)@R.T;factor=min(self.width(),self.height())*.35*self.zoom
        return np.c_[self.width()/2+q[:,0]*factor,self.height()/2-q[:,1]*factor],q[:,2]
    def paintEvent(self,event):
        p=QPainter(self);p.setRenderHint(QPainter.Antialiasing);p.fillRect(self.rect(),QColor('#071723'));p.setPen(QColor('#153247'))
        for x in range(0,self.width(),40):p.drawLine(x,0,x,self.height())
        for y in range(0,self.height(),40):p.drawLine(0,y,self.width(),y)
        if not len(self.points):p.setPen(QColor('#88a7ba'));p.drawText(self.rect(),Qt.AlignCenter,'Reconstruct a pair to view the sparse cloud\nDrag to orbit. Scroll to zoom.');return
        xy,z=self.project(self.points)
        for i in np.argsort(z):
            if np.isfinite(xy[i]).all():p.setPen(QPen(QColor(*map(int,self.colors[i])),3));p.drawPoint(QPointF(*xy[i]))
        xy,_=self.project(self.cameras);p.setPen(QPen(QColor('#32caff'),2));p.drawLine(QPointF(*xy[0]),QPointF(*xy[1]))
        for i,c in enumerate(xy):p.drawEllipse(QPointF(*c),7,7);p.drawText(QPointF(c[0]+10,c[1]),f'C{i+1}')
        p.setPen(QColor('#b6d3e0'));p.drawText(15,24,f'{len(self.points):,} points | arbitrary scale | C1/C2 camera centers')
    def mousePressEvent(self,e):self.last=e.position()
    def mouseMoveEvent(self,e):
        if self.last is not None:
            d=e.position()-self.last;self.yaw+=d.x()*.01;self.pitch=max(-1.5,min(1.5,self.pitch+d.y()*.01));self.last=e.position();self.update()
    def mouseReleaseEvent(self,e):self.last=None
    def wheelEvent(self,e):self.zoom=max(.1,min(10,self.zoom*math.exp(e.angleDelta().y()/1200)));self.update()

class Worker(QThread):
    progress=Signal(str);done=Signal(object);failed=Signal(str)
    def __init__(self,a,b,f):super().__init__();self.args=(a,b,f)
    def run(self):
        try:self.done.emit(reconstruct(*self.args,progress=self.progress.emit))
        except Exception as e:self.failed.emit(str(e))

class Window(QMainWindow):
    def __init__(self):
        super().__init__();self.setWindowTitle('3D Reconstruction | Sparse Photogrammetry Workbench');self.resize(1400,850);self.paths=[];self.result=None;self.worker=None
        root=QWidget();self.setCentralWidget(root);layout=QVBoxLayout(root)
        head=QHBoxLayout();title=QLabel('3D RECONSTRUCTION  /  IMAGE TO POINT CLOUD');title.setStyleSheet('font-size:22px;font-weight:700;color:#42caff');head.addWidget(title);head.addStretch();layout.addLayout(head)
        controls=QHBoxLayout();self.import_button=QPushButton('Import photos');self.run_button=QPushButton('Reconstruct selected pair');self.export_button=QPushButton('Export PLY + camera JSON');self.export_button.setEnabled(False)
        self.focal=QDoubleSpinBox();self.focal.setRange(1,50000);self.focal.setValue(1000);self.focal.setSuffix(' px');self.focal.setToolTip('Calibrated focal length for these image dimensions. Assumes fx=fy, centered principal point, zero distortion.')
        for w in [self.import_button,QLabel('Focal length'),self.focal,self.run_button,self.export_button]:controls.addWidget(w)
        layout.addLayout(controls);self.note=QLabel('Select exactly two overlapping, undistorted photos. Use known calibration. Sparse output only; absolute scale is unknown.');layout.addWidget(self.note)
        split=QSplitter();layout.addWidget(split,1);self.files=QListWidget();self.files.setSelectionMode(QListWidget.ExtendedSelection);split.addWidget(self.files)
        self.tabs=QTabWidget();self.original=QLabel('Import images to begin');self.features=QLabel();self.matches=QLabel();self.cloud=CloudView();self.stats=QTextEdit();self.stats.setReadOnly(True)
        for label in [self.original,self.features,self.matches]:label.setAlignment(Qt.AlignCenter);label.setMinimumSize(300,220)
        for name,w in [('Input preview',self.original),('Features',self.features),('Verified matches',self.matches),('3D cloud',self.cloud),('Quality report',self.stats)]:self.tabs.addTab(w,name)
        split.addWidget(self.tabs);split.setSizes([250,1100]);self.status=QLabel('Ready');layout.addWidget(self.status)
        self.import_button.clicked.connect(self.import_photos);self.run_button.clicked.connect(self.start);self.export_button.clicked.connect(self.save);self.files.itemSelectionChanged.connect(self.preview)
        self.setStyleSheet('QWidget{background:#0c2030;color:#d5e8f2;font-size:14px}QPushButton{background:#134665;border:1px solid #3186af;padding:9px;border-radius:4px}QPushButton:disabled{color:#607888;background:#142835}QListWidget,QTextEdit{background:#071723;border:1px solid #214258}QTabBar::tab{padding:10px;background:#173347}QTabBar::tab:selected{background:#07567c}QDoubleSpinBox{padding:6px;background:#173347}')
    def load_paths(self,paths):
        self.result=None;self.export_button.setEnabled(False);self.paths=list(paths);self.files.clear();self.status.setText("Images loaded. Reconstruct the selected pair before export.")
        for p in paths:self.files.addItem(Path(p).name)
        for i in range(min(2,len(paths))):self.files.item(i).setSelected(True)
        self.preview()
    def import_photos(self):
        paths,_=QFileDialog.getOpenFileNames(self,'Import photos','','Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)')
        if paths:self.load_paths(paths)
    def image(self,label,array):
        rgb=np.ascontiguousarray(array[:,:,::-1]);h,w=rgb.shape[:2];q=QImage(rgb.data,w,h,rgb.strides[0],QImage.Format_RGB888).copy();label.setPixmap(QPixmap.fromImage(q).scaled(1000,580,Qt.KeepAspectRatio,Qt.SmoothTransformation))
    def preview(self):
        selected=self.files.selectedIndexes()
        if selected:
            try:self.image(self.original,read_image(self.paths[selected[0].row()]))
            except Exception as e:self.status.setText(str(e))
    def start(self):
        indices=self.files.selectedIndexes()
        if len(indices)!=2:self.status.setText('Select exactly two images, using Ctrl-click if needed.');return
        self.run_button.setEnabled(False);self.import_button.setEnabled(False);self.export_button.setEnabled(False);self.result=None
        self.worker=Worker(*(self.paths[i.row()] for i in indices),self.focal.value());self.worker.progress.connect(self.status.setText);self.worker.done.connect(self.completed);self.worker.failed.connect(self.failed);self.worker.finished.connect(self.finished);self.worker.start()
    def completed(self,result):
        self.result=result;self.image(self.features,result.feature_image);self.image(self.matches,result.matches_image);self.cloud.set_result(result);self.stats.setText(json.dumps(result.report,indent=2));self.tabs.setCurrentWidget(self.cloud);self.export_button.setEnabled(True);self.status.setText(f'Complete: {len(result.points)} sparse points. Metric scale and dense surfaces are unavailable.')
    def failed(self,message):self.status.setText('Reconstruction failed: '+message)
    def finished(self):self.run_button.setEnabled(True);self.import_button.setEnabled(True)
    def save(self):
        path,_=QFileDialog.getSaveFileName(self,'Export sparse reconstruction','reconstruction.ply','PLY (*.ply)')
        if path:
            try:export(self.result,path);self.status.setText('Saved point cloud and camera report')
            except Exception as e:self.status.setText('Export failed: '+str(e))
    def closeEvent(self,event):
        if self.worker and self.worker.isRunning():self.status.setText('Wait for reconstruction to finish before closing.');event.ignore()
        else:event.accept()

def main():
    app=QApplication([]);w=Window();w.show();app.exec()
