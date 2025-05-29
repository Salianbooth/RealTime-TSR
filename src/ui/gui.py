import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSplitter,
    QVBoxLayout, QHBoxLayout, QTabWidget, QTextEdit, QListWidget,
    QListWidgetItem, QFileDialog, QStatusBar, QSizePolicy, QSlider,
    QAction
)
from PyQt5.QtCore import Qt, QTimer, QTime
from PyQt5.QtGui import QImage, QPixmap, QKeySequence, QFont

class TrafficSignUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("交通标志检测与识别")
        self.resize(1600, 900)
        self.camera = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        self.total_frames = 0
        self.fps = 30
        self.overlay_enabled = True
        self._setup_ui()
        self._create_actions()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # 左侧视频与控制
        left_widget = QWidget()
        v_layout = QVBoxLayout(left_widget)
        v_layout.setContentsMargins(10, 10, 10, 10)
        self.video_label = QLabel("请点击“打开视频”选择文件")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("border:2px dashed #999; background:#f5f5f5; color:#666;")
        font = QFont(); font.setPointSize(20)
        self.video_label.setFont(font)
        v_layout.addWidget(self.video_label)

        ctrl = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderMoved.connect(self._seek_frame)
        ctrl.addWidget(self.slider)
        self.time_label = QLabel("00:00:00 / 00:00:00")
        self.time_label.setFixedWidth(160)
        ctrl.addWidget(self.time_label)
        v_layout.addLayout(ctrl)

        btns = QHBoxLayout(); btns.setSpacing(10)
        for text, slot, tip in [
            ("打开视频", self._open_video, "Ctrl+O"),
            ("开始/暂停", self._toggle_play, "Space"),
            ("截屏保存", self._save_frame, "Ctrl+S"),
            ("退出", self.close, "Ctrl+Q")
        ]:
            btn = QPushButton(text)
            btn.setToolTip(tip)
            btn.setMinimumHeight(40)
            btn.clicked.connect(slot)
            btns.addWidget(btn)
        v_layout.addLayout(btns)
        splitter.addWidget(left_widget)

        # 右侧选项卡
        tabs = QTabWidget()
        # 检测结果
        result_tab = QWidget(); res_layout = QVBoxLayout(result_tab)
        self.sign_img = QLabel("无检测结果")
        self.sign_img.setFixedSize(200,200)
        self.sign_img.setAlignment(Qt.AlignCenter)
        res_layout.addWidget(self.sign_img)
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        res_layout.addWidget(self.info_text)
        tabs.addTab(result_tab, "检测结果")
        # 历史记录
        hist_tab = QWidget(); h_layout = QVBoxLayout(hist_tab)
        self.hist_list = QListWidget(); h_layout.addWidget(self.hist_list)
        tabs.addTab(hist_tab, "历史记录")
        splitter.addWidget(tabs)
        splitter.setStretchFactor(0,3); splitter.setStretchFactor(1,1)

        status = QStatusBar(); self.setStatusBar(status)
        status.showMessage("就绪")

    def _create_actions(self):
        # 文件菜单
        file_menu = self.menuBar().addMenu("文件")
        file_menu.addAction(QAction("打开(&O)", self, shortcut=QKeySequence("Ctrl+O"), triggered=self._open_video))
        file_menu.addAction(QAction("退出(&Q)", self, shortcut=QKeySequence("Ctrl+Q"), triggered=self.close))
        # 视图菜单
        view_menu = self.menuBar().addMenu("视图")
        self.act_overlay = QAction("检测叠加层", self, checkable=True, checked=True)
        self.act_overlay.triggered.connect(self._toggle_overlay)
        view_menu.addAction(self.act_overlay)

    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频 (*.mp4 *.avi)")
        if not path: return
        if self.camera: self.camera.release()
        self.camera = cv2.VideoCapture(path)
        self.total_frames = int(self.camera.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.camera.get(cv2.CAP_PROP_FPS) or 30
        self.slider.setMaximum(self.total_frames)
        self.slider.setEnabled(True)
        dur = QTime(0,0,0).addMSecs(int(self.total_frames/self.fps*1000))
        self.time_label.setText(f"00:00:00 / {dur.toString('hh:mm:ss')}" )

    def _toggle_play(self):
        if not self.camera: return
        if self.timer.isActive():
            self.timer.stop()
            self.statusBar().showMessage("已暂停")
        else:
            self.timer.start(int(1000/self.fps))
            self.statusBar().showMessage("检测中...")

    def _seek_frame(self, f):
        if not self.camera: return
        self.camera.set(cv2.CAP_PROP_POS_FRAMES, f)

    def _save_frame(self):
        if not self.camera: return
        ret, frame = self.camera.read()
        if not ret: return
        p, _ = QFileDialog.getSaveFileName(self, "保存图片", "", "PNG (*.png)")
        if p:
            cv2.imwrite(p, frame)
            self.statusBar().showMessage(f"已保存 {p}")

    def _update_frame(self):
        ret, frame = self.camera.read()
        if not ret:
            self.timer.stop()
            self.statusBar().showMessage("播放结束")
            return
        # 绘制叠加层
        if self.overlay_enabled and hasattr(self, 'detector'):
            dets = self.detector.detect(frame)
            self.info_text.clear()
            self.hist_list.clear()
            for x1,y1,x2,y2,cls,conf in dets:
                cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame, f"{cls}:{conf:.2f}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
                self.hist_list.addItem(f"帧{int(self.camera.get(cv2.CAP_PROP_POS_FRAMES))}: {cls} {conf:.2f}")
        # 显示视频
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h,w,_ = rgb.shape
        img = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        self.video_label.setPixmap(pix)
        pos = int(self.camera.get(cv2.CAP_PROP_POS_FRAMES))
        self.slider.blockSignals(True)
        self.slider.setValue(pos)
        self.slider.blockSignals(False)
        cur = QTime(0,0,0).addMSecs(int(pos/self.fps*1000))
        tot = QTime(0,0,0).addMSecs(int(self.total_frames/self.fps*1000))
        self.time_label.setText(f"{cur.toString('hh:mm:ss')} / {tot.toString('hh:mm:ss')}" )

    def _toggle_overlay(self):
        self.overlay_enabled = not self.overlay_enabled
        self.statusBar().showMessage("叠加层" + ("开启" if self.overlay_enabled else "关闭"), 2000)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TrafficSignUI()
    window.show()
    sys.exit(app.exec_())
