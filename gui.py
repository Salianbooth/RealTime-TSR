# gui.py
import sys
import cv2
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QTextEdit, QSplitter, QFileDialog, QStatusBar, QSizePolicy, QSlider, QAction, QTabWidget, QListWidget
)
from PyQt5.QtCore import Qt, QTimer, QTime
from PyQt5.QtGui import QImage, QPixmap, QKeySequence, QFont

from detector import Detector  # 自定义的检测逻辑

class TrafficSignUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("交通标志检测与识别")
        self.resize(1600, 900)

        # 视频相关
        self.processor = None          # 将在 _open_video 中赋值
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # 检测器
        self.detector = Detector()     # 在 detector.py 里实现

        self._setup_ui()
        self._create_actions()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # 左侧：视频显示 + 控制按钮
        left_widget = QWidget()
        v_layout = QVBoxLayout(left_widget)
        v_layout.setContentsMargins(10, 10, 10, 10)

        self.video_label = QLabel('请点击“打开视频”选择文件')
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("border:2px dashed #999; background:#f5f5f5; color:#666;")
        font = QFont()
        font.setPointSize(20)
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

        btns = QHBoxLayout()
        btns.setSpacing(10)
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

        # 右侧：选项卡（检测结果 + 历史记录）
        tabs = QTabWidget()

        result_tab = QWidget()
        res_layout = QVBoxLayout(result_tab)
        self.sign_img = QLabel("无检测结果预览")
        self.sign_img.setFixedSize(200, 200)
        self.sign_img.setAlignment(Qt.AlignCenter)
        res_layout.addWidget(self.sign_img)

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        res_layout.addWidget(self.info_text)
        tabs.addTab(result_tab, "检测结果")

        hist_tab = QWidget()
        h_layout = QVBoxLayout(hist_tab)
        self.hist_list = QListWidget()
        h_layout.addWidget(self.hist_list)
        tabs.addTab(hist_tab, "历史记录")

        splitter.addWidget(tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        status = QStatusBar()
        self.setStatusBar(status)
        status.showMessage("就绪")

    def _create_actions(self):
        file_menu = self.menuBar().addMenu("文件")
        file_menu.addAction(QAction("打开(&O)", self, shortcut=QKeySequence("Ctrl+O"), triggered=self._open_video))
        file_menu.addAction(QAction("退出(&Q)", self, shortcut=QKeySequence("Ctrl+Q"), triggered=self.close))

        view_menu = self.menuBar().addMenu("视图")
        self.act_overlay = QAction("检测叠加层", self, checkable=True, checked=True)
        view_menu.addAction(self.act_overlay)
        self.act_overlay.triggered.connect(self._toggle_overlay)

    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频 (*.mp4 *.avi)")
        if not path:
            return

        # 用 OpenCV 打开视频
        self.processor = cv2.VideoCapture(path)
        if not self.processor.isOpened():
            self.statusBar().showMessage("无法打开视频文件")
            return

        self.total_frames = int(self.processor.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.processor.get(cv2.CAP_PROP_FPS) or 30
        self.slider.setMaximum(self.total_frames)
        self.slider.setEnabled(True)

        dur = QTime(0, 0, 0).addMSecs(int(self.total_frames / self.fps * 1000))
        self.time_label.setText(f"00:00:00 / {dur.toString('hh:mm:ss')}")

        self.info_text.clear()
        self.statusBar().showMessage("视频已打开，正在播放并实时检测…")
        self.timer.start(int(1000 / self.fps))

    def _toggle_play(self):
        if not self.processor:
            return
        if self.timer.isActive():
            self.timer.stop()
            self.statusBar().showMessage("已暂停")
        else:
            self.timer.start(int(1000 / self.fps))
            self.statusBar().showMessage("播放中…")

    def _seek_frame(self, f):
        if not self.processor:
            return
        self.processor.set(cv2.CAP_PROP_POS_FRAMES, f)

    def _save_frame(self):
        if not self.processor:
            return
        ret, frame = self.processor.read()
        if not ret:
            return
        p, _ = QFileDialog.getSaveFileName(self, "保存图片", "", "PNG (*.png)")
        if p:
            cv2.imwrite(p, frame)
            self.statusBar().showMessage(f"已保存：{p}")

    def update_frame(self):
        if not self.processor:
            return

        ret, frame = self.processor.read()
        if not ret:
            # 播放到末尾后，循环从头开始
            self.processor.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.processor.read()
            if not ret:
                print("无法重新读取第一帧")
                return

        # 预处理（如果有）
        # frame = self.preprocessor.preprocess(frame)

        # 调用检测逻辑，得到一个带框的 frame 以及文字描述 text
        frame_with_boxes, text = self.detector.detect(frame)

        # 拼接当前时间
        from datetime import datetime
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        text_with_time = f"[{current_time}] {text}"

        # 把文字显示到右侧“检测结果”面板
        self.info_text.append(text_with_time)

        # 在 QLabel 上显示带框的图像
        self.show_frame_on_label(frame_with_boxes)

        # 同步更新滑动条和时间标签
        pos = int(self.processor.get(cv2.CAP_PROP_POS_FRAMES))
        self.slider.blockSignals(True)
        self.slider.setValue(pos)
        self.slider.blockSignals(False)
        cur = QTime(0, 0, 0).addMSecs(int(pos / self.fps * 1000))
        tot = QTime(0, 0, 0).addMSecs(int(self.total_frames / self.fps * 1000))
        self.time_label.setText(f"{cur.toString('hh:mm:ss')} / {tot.toString('hh:mm:ss')}")

    def show_frame_on_label(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled_image = image.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(scaled_image))

    def _toggle_overlay(self):
        self.overlay_enabled = not getattr(self, "overlay_enabled", True)
        self.statusBar().showMessage("叠加层" + ("开启" if self.overlay_enabled else "关闭"), 2000)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TrafficSignUI()
    window.show()
    sys.exit(app.exec_())
