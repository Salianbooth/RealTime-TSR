import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSplitter,
    QVBoxLayout, QHBoxLayout, QTextEdit, QFileDialog, QStatusBar,
    QSizePolicy, QSlider
)
from PyQt5.QtCore import Qt, QTimer, QTime
from PyQt5.QtGui import QImage, QPixmap, QFont


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

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # === 左侧：竖直排列的按钮区，按钮尺寸增大，字体增大 ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        # 边距让按钮不贴边
        left_layout.setContentsMargins(10, 10, 10, 10)
        # 按钮之间的竖直间隔
        left_layout.setSpacing(20)

        btn_font = QFont()
        btn_font.setPointSize(14)  # 按钮字体设为 16pt

        btn_open = QPushButton("打开视频")
        btn_open.setToolTip("Ctrl+O")
        btn_open.setFixedSize(150, 80)
        btn_open.setFont(btn_font)
        btn_open.clicked.connect(self._open_video)

        btn_toggle = QPushButton("开始/暂停")
        btn_toggle.setToolTip("Space")
        btn_toggle.setFixedSize(150, 80)
        btn_toggle.setFont(btn_font)
        btn_toggle.clicked.connect(self._toggle_play)

        btn_exit = QPushButton("退出")
        btn_exit.setToolTip("Ctrl+Q")
        btn_exit.setFixedSize(150, 80)
        btn_exit.setFont(btn_font)
        btn_exit.clicked.connect(self.close)

        left_layout.addStretch()
        left_layout.addWidget(btn_open, alignment=Qt.AlignHCenter)
        left_layout.addWidget(btn_toggle, alignment=Qt.AlignHCenter)
        left_layout.addWidget(btn_exit, alignment=Qt.AlignHCenter)
        left_layout.addStretch()

        splitter.addWidget(left_panel)

        # === 中间：视频预览 + 进度条 ===
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(10, 10, 10, 10)
        center_layout.setSpacing(15)

        # 视频显示区
        self.video_label = QLabel("请点击“打开视频”选择文件")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet(
            "border:2px dashed #999; background:#f5f5f5; color:#666;"
        )
        video_font = QFont()
        video_font.setPointSize(24)  # 视频提示文字设为 24pt
        self.video_label.setFont(video_font)
        center_layout.addWidget(self.video_label, stretch=3)

        # 进度条 + 时间显示
        slider_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderMoved.connect(self._seek_frame)
        slider_layout.addWidget(self.slider)

        self.time_label = QLabel("00:00:00 / 00:00:00")
        time_font = QFont()
        time_font.setPointSize(14)  # 时间标签字体设为 14pt
        self.time_label.setFont(time_font)
        self.time_label.setFixedWidth(180)
        slider_layout.addWidget(self.time_label)
        center_layout.addLayout(slider_layout)

        splitter.addWidget(center_panel)

        # === 右侧：仅保留检测结果 文本区，字体增大 ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)

        label_result_title = QLabel("检测结果")
        title_font = QFont()
        title_font.setPointSize(14)  # 检测结果标题设为 16pt
        title_font.setBold(True)
        label_result_title.setFont(title_font)
        right_layout.addWidget(label_result_title)

        self.info_text = QTextEdit()
        info_font = QFont()
        info_font.setPointSize(14)  # 识别结果文本区字体设为 14pt
        self.info_text.setFont(info_font)
        self.info_text.setReadOnly(True)
        right_layout.addWidget(self.info_text)

        splitter.addWidget(right_panel)

        # 设置三个区域的宽度权重：左侧按钮最窄、中间视频最大、右侧结果次之
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 5)
        splitter.setStretchFactor(2, 2)

        # 底部状态栏
        status = QStatusBar()
        self.setStatusBar(status)
        status.showMessage("就绪")

    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频 (*.mp4 *.avi)")
        if not path:
            return
        if self.camera:
            self.camera.release()
        self.camera = cv2.VideoCapture(path)
        self.total_frames = int(self.camera.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.camera.get(cv2.CAP_PROP_FPS) or 30
        self.slider.setMaximum(self.total_frames)
        self.slider.setEnabled(True)
        duration = QTime(0, 0, 0).addMSecs(int(self.total_frames / self.fps * 1000))
        self.time_label.setText(f"00:00:00 / {duration.toString('hh:mm:ss')}")

    def _toggle_play(self):
        if not self.camera:
            return
        if self.timer.isActive():
            self.timer.stop()
            self.statusBar().showMessage("已暂停")
        else:
            self.timer.start(int(1000 / self.fps))
            self.statusBar().showMessage("检测中...")

    def _seek_frame(self, frame_index):
        if not self.camera:
            return
        self.camera.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    def _update_frame(self):
        ret, frame = self.camera.read()
        if not ret:
            self.timer.stop()
            self.statusBar().showMessage("播放结束")
            return

        # 如果 overlay 开启且 detector 已设置，则只输出标签名称
        if self.overlay_enabled and hasattr(self, 'detector'):
            dets = self.detector.detect(frame)  # 假设返回 [ (x1, y1, x2, y2, cls, conf), ... ]
            self.info_text.clear()
            for _, _, _, _, cls, _ in dets:
                self.info_text.append(f"{cls}")

        # 将 BGR 转为 RGB 显示
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        self.video_label.setPixmap(pix)

        # 更新滑块位置和当前/总时长
        pos = int(self.camera.get(cv2.CAP_PROP_POS_FRAMES))
        self.slider.blockSignals(True)
        self.slider.setValue(pos)
        self.slider.blockSignals(False)

        current_time = QTime(0, 0, 0).addMSecs(int(pos / self.fps * 1000))
        total_time = QTime(0, 0, 0).addMSecs(int(self.total_frames / self.fps * 1000))
        self.time_label.setText(
            f"{current_time.toString('hh:mm:ss')} / {total_time.toString('hh:mm:ss')}"
        )

    def _toggle_overlay(self):
        self.overlay_enabled = not self.overlay_enabled
        self.statusBar().showMessage(
            "叠加层" + ("开启" if self.overlay_enabled else "关闭"), 2000
        )


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TrafficSignUI()
    window.show()
    sys.exit(app.exec_())
