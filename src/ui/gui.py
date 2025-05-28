import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSplitter,
    QVBoxLayout, QHBoxLayout, QTabWidget, QTextEdit, QFileDialog, QStatusBar,
    QSizePolicy, QSlider, QAction, QMessageBox
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
        self._setup_ui()
        self._create_actions()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setCollapsible(1, True)
        layout.addWidget(splitter)

        # 左侧视频与控制
        left_widget = QWidget()
        v_layout = QVBoxLayout(left_widget)
        v_layout.setSpacing(10)
        v_layout.setContentsMargins(10, 10, 10, 10)

        # 视频展示区：初始显示卡片提示
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet(
            "border:2px dashed #999; background:#f5f5f5; color:#666;"
        )
        font = QFont()
        font.setPointSize(20)
        self.video_label.setFont(font)
        self.video_label.setText("请点击“打开视频”选择文件")
        v_layout.addWidget(self.video_label)

        # 进度条与时间
        control_hbox = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderMoved.connect(self._seek_frame)
        control_hbox.addWidget(self.slider)
        self.time_label = QLabel("00:00:00 / 00:00:00")
        self.time_label.setFixedWidth(160)
        control_hbox.addWidget(self.time_label)
        v_layout.addLayout(control_hbox)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)
        self.btn_open = QPushButton("打开视频")
        self.btn_toggle = QPushButton("开始/暂停")
        self.btn_capture = QPushButton("截屏保存")
        self.btn_exit = QPushButton("退出")
        for btn, tip in [
            (self.btn_open, "打开视频 (Ctrl+O)"),
            (self.btn_toggle, "开始/暂停 (Space)"),
            (self.btn_capture, "截屏保存 (Ctrl+S)"),
            (self.btn_exit, "退出 (Ctrl+Q)")
        ]:
            btn.setMinimumHeight(50)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setStyleSheet(
                "QPushButton{background:#1976D2;color:white;border-radius:6px;font-size:16px;}"
                "QPushButton:hover{background:#1565C0;}"
            )
            btn.setToolTip(tip)
            btn_layout.addWidget(btn)
        v_layout.addLayout(btn_layout)

        splitter.addWidget(left_widget)

        # 右侧选项卡（检测结果 & 历史）
        tabs = QTabWidget()
        tabs.setStyleSheet(
            "QTabBar::tab { min-width: 160px; min-height: 60px; font-size: 18px; font-weight: bold; }"
        )
        result_tab = QWidget()
        res_layout = QVBoxLayout(result_tab)
        self.sign_img = QLabel("无检测结果")
        self.sign_img.setAlignment(Qt.AlignCenter)
        self.sign_img.setFixedSize(300, 300)
        self.sign_img.setStyleSheet("border:1px solid #ccc;")
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        res_layout.addWidget(self.sign_img, alignment=Qt.AlignCenter)
        res_layout.addWidget(self.info_text)
        tabs.addTab(result_tab, "检测结果")

        hist_tab = QWidget()
        hist_layout = QVBoxLayout(hist_tab)
        self.hist_text = QTextEdit()
        self.hist_text.setReadOnly(True)
        hist_layout.addWidget(self.hist_text)
        tabs.addTab(hist_tab, "历史记录")

        splitter.addWidget(tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        status = QStatusBar()
        self.setStatusBar(status)
        status.showMessage("就绪")

        # 信号连接
        self.btn_open.clicked.connect(self._open_video)
        self.btn_toggle.clicked.connect(self._toggle_play)
        self.btn_capture.clicked.connect(self._save_frame)
        self.btn_exit.clicked.connect(self.close)

    def _create_actions(self):
        self.addAction(QAction("打开(&O)", self, shortcut=QKeySequence("Ctrl+O"), triggered=self._open_video))
        self.addAction(QAction("开始/暂停(&P)", self, shortcut=QKeySequence("Space"), triggered=self._toggle_play))
        self.addAction(QAction("截屏(&S)", self, shortcut=QKeySequence("Ctrl+S"), triggered=self._save_frame))
        self.addAction(QAction("退出(&Q)", self, shortcut=QKeySequence("Ctrl+Q"), triggered=self.close))

    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频文件 (*.mp4 *.avi);;所有文件 (*.*)")
        if not path:
            return
        # 释放旧资源
        if self.camera:
            self.camera.release()
        self.camera = cv2.VideoCapture(path)
        # 获取基本信息
        self.total_frames = int(self.camera.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.camera.get(cv2.CAP_PROP_FPS) or 30
        self.slider.setMaximum(self.total_frames)
        self.slider.setEnabled(True)
        duration = QTime(0,0,0).addMSecs(int(self.total_frames/self.fps*1000))
        self.time_label.setText(f"00:00:00 / {duration.toString('hh:mm:ss')}")
        # 读取首帧作为封面
        ret, frame = self.camera.read()
        if ret:
            pix = self._frame_to_pixmap(frame)
            self.video_label.setPixmap(pix)
            # 重置到首帧
            self.camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.statusBar().showMessage(f"已加载：{path}")

    def _toggle_play(self):
        if self.camera is None:
            return
        if self.timer.isActive():
            self.timer.stop()
            self.statusBar().showMessage("已暂停")
        else:
            self.timer.start(int(1000/self.fps))
            self.statusBar().showMessage("检测中...")

    def _update_frame(self):
        ret, frame = self.camera.read()
        if not ret:
            self.timer.stop()
            self.statusBar().showMessage("播放结束")
            return
            # —— 在这里加识别 ——
            # 比如：
            # detections = self.detector.detect(frame)
            # 在 frame 上画框、在 info_text 里输出信息、
            # 或者把识别出的子图用 self.sign_img.setPixmap(...) 展示

            # 然后再把（可能画过框的）frame 显示到界面上
        pos = int(self.camera.get(cv2.CAP_PROP_POS_FRAMES))
        self.slider.blockSignals(True)
        self.slider.setValue(pos)
        self.slider.blockSignals(False)
        current = QTime(0,0,0).addMSecs(int(pos/self.fps*1000))
        total = QTime(0,0,0).addMSecs(int(self.total_frames/self.fps*1000))
        self.time_label.setText(f"{current.toString('hh:mm:ss')} / {total.toString('hh:mm:ss')}")
        pix = self._frame_to_pixmap(frame)
        self.video_label.setPixmap(pix)

    def _seek_frame(self, frame_no):
        if not self.camera:
            return
        self.camera.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        if not self.timer.isActive():
            ret, frame = self.camera.read()
            if ret:
                self.video_label.setPixmap(self._frame_to_pixmap(frame))

    def _save_frame(self):
        if not self.camera:
            return
        ret, frame = self.camera.read()
        if not ret:
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存图片", "", "PNG 文件 (*.png)")
        if path:
            cv2.imwrite(path, frame)
            self.statusBar().showMessage(f"已保存：{path}")

    def _frame_to_pixmap(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        img = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
        return QPixmap.fromImage(img)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TrafficSignUI()
    window.show()
    sys.exit(app.exec_())
