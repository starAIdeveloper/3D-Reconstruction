import tempfile,json,time
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from PySide6.QtCore import Qt,QPoint
from recon.gui import Window
from recon.pipeline import export
from tests.fixtures import make_pair

app=QApplication([]);w=Window();w.show();root=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as d:
    paths=make_pair(d);w.load_paths(paths);w.focal.setValue(700);w.start()
    deadline=time.time()+15
    while (w.worker.isRunning() or w.result is None) and time.time()<deadline:app.processEvents();QTest.qWait(20)
    assert w.result is not None,w.status.text();assert len(w.result.points)>30;app.processEvents()
    assert w.run_button.isEnabled() and w.export_button.isEnabled()
    previous=w.cloud.yaw;QTest.mousePress(w.cloud,Qt.LeftButton,pos=QPoint(120,120));QTest.mouseMove(w.cloud,QPoint(170,140));QTest.mouseRelease(w.cloud,Qt.LeftButton,pos=QPoint(170,140));assert w.cloud.yaw!=previous
    app.processEvents();w.grab().save(str(root/'docs/desktop.png'));w.tabs.setCurrentWidget(w.matches);app.processEvents();w.grab().save(str(root/'docs/matches.png'))
    out=Path(d)/'test.ply';export(w.result,out);assert out.exists() and out.with_suffix('.json').exists()
    report={'passed':['Qt runtime','threaded reconstruction','selected pair','feature and match previews','cloud orbit','PLY and camera JSON export'],'synthetic_result':w.result.report,'real_photo_accuracy':'not evaluated'}
    (root/'docs/validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
w.close()
