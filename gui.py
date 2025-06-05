# gui.py
import sys
import cv2
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QTextEdit, QSplitter, QFileDialog, QStatusBar, QSizePolicy, QSlider, QTabWidget, QListWidget
)
from PyQt5.QtCore import Qt, QTimer, QTime
from PyQt5.QtGui import QImage, QPixmap, QFont

from detector import Detector  # 自定义的检测逻辑


class TrafficSignUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("交通标志检测与识别")
        self.resize(1600, 900)

        # 视频相关
        self.processor = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # 检测器
        self.detector = Detector()

        self._setup_ui()

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
            ("退出", self.close, "Ctrl+Q")
        ]:
            btn = QPushButton(text)
            btn.setToolTip(tip)
            btn.setMinimumHeight(40)
            btn.clicked.connect(slot)
            btns.addWidget(btn)
        v_layout.addLayout(btns)

        splitter.addWidget(left_widget)

        # 右侧：选项卡（只保留“检测结果”与“历史记录”）
        tabs = QTabWidget()

        # “检测结果” 页：只保留文本框，并限制最大行数
        result_tab = QWidget()
        res_layout = QVBoxLayout(result_tab)

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        # 关键：限制文档最多保留 100 个 block（即大概 100 行）
        self.info_text.document().setMaximumBlockCount(100)
        res_layout.addWidget(self.info_text)
        tabs.addTab(result_tab, "检测结果")

        # “历史记录” 页
        hist_tab = QWidget()
        h_layout = QVBoxLayout(hist_tab)
        self.hist_list = QListWidget()
        h_layout.addWidget(self.hist_list)
        tabs.addTab(hist_tab, "历史记录")

        splitter.addWidget(tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        # 状态栏
        status = QStatusBar()
        self.setStatusBar(status)
        status.showMessage("就绪")

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

        # 清空 QTextEdit
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

        # 调用检测逻辑，得到带框的 frame 和文字描述 text
        frame_with_boxes, text = self.detector.detect(frame)

        # 拼接当前时间
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        text_with_time = f"[{current_time}] {text}"

        # 通过 append() 追加日志，QTextEdit 会自动维护“最多保留 100 行”
        self.info_text.append(text_with_time)
        # 保证滚动条滚动到最底部
        self.info_text.verticalScrollBar().setValue(self.info_text.verticalScrollBar().maximum())

        # 在 QLabel 上显示带框的图像
        self.show_frame_on_label(frame_with_boxes)

        # 同步更新滑动条与时间标签
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
        # 如果日后需要用到叠加层开关，可在此添加逻辑
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TrafficSignUI()
    window.show()
    sys.exit(app.exec_())