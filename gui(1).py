#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于PyQt的用户界面模块
负责系统的图形用户界面实现
"""
import sys
import cv2
from datetime import datetime
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QSlider, QSpinBox,
    QCheckBox, QPlainTextEdit, QApplication, QFileDialog,
    QFrame, QStatusBar
)
from PyQt5.QtPrintSupport import QPrinter, QPrintDialog
from PyQt5.QtCore import Qt, QTimer, QTime
from PyQt5.QtGui import QImage, QPixmap

from detector import Detector  # 自定义的检测逻辑，需实现 detect(frame) -> (frame_with_boxes, text)


class StyleSheet:
    """样式表类"""

    MAIN_STYLE = """
        QMainWindow {
            background-color: #f0f0f0;
        }

        QPushButton {
            background-color: #2196F3;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
            min-width: 100px;
        }

        QPushButton:hover {
            background-color: #1976D2;
        }

        QPushButton:pressed {
            background-color: #0D47A1;
        }

        QLabel {
            color: #333333;
            font-size: 14px;
        }

        QSlider::groove:horizontal {
            border: 1px solid #999999;
            height: 8px;
            background: #ffffff;
            margin: 2px 0;
            border-radius: 4px;
        }

        QSlider::handle:horizontal {
            background: #2196F3;
            border: none;
            width: 18px;
            margin: -6px 0;
            border-radius: 9px;
        }

        QSlider::handle:horizontal:hover {
            background: #1976D2;
        }

        QSpinBox {
            padding: 5px;
            border: 1px solid #999999;
            border-radius: 4px;
            background: white;
        }

        QCheckBox {
            spacing: 8px;
        }

        QCheckBox::indicator {
            width: 18px;
            height: 18px;
        }

        QPlainTextEdit {
            border: 1px solid #999999;
            border-radius: 4px;
            background-color: white;
            padding: 5px;
            font-family: "Consolas", monospace;
        }
    """


class MainWindow(QMainWindow):
    """主窗口类"""

    def __init__(self):
        super().__init__()
        self.video_source = None
        self.is_playing = False
        self.current_frame = 0

        # 检测器
        self.detector = Detector()

        self.initUI()

    def initUI(self):
        """初始化用户界面"""
        # 设置窗口标题和大小
        self.setWindowTitle('交通标志检测与识别系统')
        self.setGeometry(100, 100, 1600, 900)

        # 应用样式表
        self.setStyleSheet(StyleSheet.MAIN_STYLE)

        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # 左侧控制面板
        left_panel = QFrame()
        left_panel.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        left_panel.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)
        main_layout.addWidget(left_panel)

        # 控制面板标题
        control_title = QLabel("控制面板")
        control_title.setAlignment(Qt.AlignCenter)
        control_title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 15px;")
        left_layout.addWidget(control_title)

        # 控制按钮
        button_layout = QVBoxLayout()
        button_layout.setSpacing(10)

        self.upload_button = QPushButton('打开视频')
        self.start_button = QPushButton('开始/暂停')
        self.pause_button = QPushButton('暂停')
        self.save_button = QPushButton('保存')
        self.exit_button = QPushButton('退出')

        for button in [
            self.upload_button, self.start_button, self.pause_button,
            self.save_button, self.exit_button
        ]:
            button.setMinimumHeight(40)
            button_layout.addWidget(button)

        self.exit_button.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)

        left_layout.addLayout(button_layout)

        # 参数控制区域标题
        params_title = QLabel("参数设置")
        params_title.setAlignment(Qt.AlignCenter)
        params_title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 20px 0 10px 0;")
        left_layout.addWidget(params_title)

        # 参数控制区域
        params_container = QFrame()
        params_container.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        params_layout = QVBoxLayout(params_container)

        self.exposure_slider = self.create_slider_group("曝光", -13, 0)
        params_layout.addLayout(self.exposure_slider)

        self.gain_slider = self.create_slider_group("增益", 0, 100)
        params_layout.addLayout(self.gain_slider)

        self.brightness_slider = self.create_slider_group("亮度", 0, 100)
        params_layout.addLayout(self.brightness_slider)

        self.contrast_slider = self.create_slider_group("对比度", 0, 100)
        params_layout.addLayout(self.contrast_slider)

        left_layout.addWidget(params_container)

        # 灰度显示复选框
        self.gray_checkbox = QCheckBox("灰度显示")
        self.gray_checkbox.setStyleSheet("margin-top: 10px;")
        left_layout.addWidget(self.gray_checkbox)

        # 添加弹性空间
        left_layout.addStretch()

        # 日志显示区域
        log_title = QLabel("运行日志")
        log_title.setAlignment(Qt.AlignCenter)
        log_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        left_layout.addWidget(log_title)

        self.log_text = QPlainTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QPlainTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #e0e0e0;
                font-family: "Consolas", monospace;
            }
        """)
        left_layout.addWidget(self.log_text)

        # 中间视频显示区域
        video_container = QFrame()
        video_container.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        video_container.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        video_layout = QVBoxLayout(video_container)

        video_title = QLabel("视频预览")
        video_title.setAlignment(Qt.AlignCenter)
        video_title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 15px;")
        video_layout.addWidget(video_title)

        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("""
            border: 2px solid #e0e0e0;
            border-radius: 5px;
            background-color: #f8f9fa;
        """)
        self.video_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.video_label)

        # 滑动条 + 时间标签
        ctrl = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderMoved.connect(self._seek_frame)
        ctrl.addWidget(self.slider)

        self.time_label = QLabel("00:00:00 / 00:00:00")
        self.time_label.setFixedWidth(160)
        ctrl.addWidget(self.time_label)
        video_layout.addLayout(ctrl)

        video_layout.addStretch()
        main_layout.addWidget(video_container, stretch=2)

        # 右侧“识别结果”显示面板（简化版，只剩标题和一个文本框用于打印输出/检测结果）
        right_panel = QFrame()
        right_panel.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        right_panel.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        right_panel.setFixedWidth(300)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(15)
        main_layout.addWidget(right_panel)

        result_title = QLabel("识别结果")
        result_title.setAlignment(Qt.AlignCenter)
        result_title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 15px;")
        right_layout.addWidget(result_title)

        # 用于输出检测结果的文本框
        self.output_text = QPlainTextEdit()
        self.output_text.setReadOnly(True)
        # 限制最多保留 100 条记录
        self.output_text.document().setMaximumBlockCount(100)
        self.output_text.setStyleSheet("""
            QPlainTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #e0e0e0;
                font-family: "Consolas", monospace;
            }
        """)
        right_layout.addWidget(self.output_text)

        # “打印日志”按钮
        self.print_button = QPushButton("打印日志")
        self.print_button.setMinimumHeight(40)
        self.print_button.clicked.connect(self.print_logs)
        right_layout.addWidget(self.print_button)

        # 设置信号连接
        self.setup_connections()

        # 初始化摄像头 & 定时器
        self.processor = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # 状态栏
        status = QStatusBar()
        self.setStatusBar(status)
        status.showMessage("就绪")

    def create_slider_group(self, name, min_val, max_val):
        """创建滑块组（标签+滑块+数值框）"""
        layout = QHBoxLayout()

        label = QLabel(name)
        label.setMinimumWidth(60)
        layout.addWidget(label)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        layout.addWidget(slider)

        spinbox = QSpinBox()
        spinbox.setMinimum(min_val)
        spinbox.setMaximum(max_val)
        spinbox.setMinimumWidth(60)
        layout.addWidget(spinbox)

        slider.valueChanged.connect(spinbox.setValue)
        spinbox.valueChanged.connect(slider.setValue)

        return layout

    def setup_connections(self):
        """设置信号和槽的连接"""
        self.upload_button.clicked.connect(self._open_video)
        self.start_button.clicked.connect(self._toggle_play)
        self.pause_button.clicked.connect(self._toggle_play)
        self.save_button.clicked.connect(self.save_frame)
        self.exit_button.clicked.connect(self.close)

    def _open_video(self):
        """打开本地视频文件"""
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)")
        if not path:
            return

        # 如果已有捕获对象，先释放
        if self.processor:
            self.processor.release()
            self.processor = None

        # 用 OpenCV 打开视频
        self.processor = cv2.VideoCapture(path)
        if not self.processor.isOpened():
            self.statusBar().showMessage("无法打开视频文件")
            return

        self.video_source = path
        self.is_playing = True
        self.current_frame = 0

        # 视频总帧数与 FPS
        self.total_frames = int(self.processor.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.processor.get(cv2.CAP_PROP_FPS) or 30
        self.slider.setMaximum(self.total_frames)
        self.slider.setEnabled(True)

        dur = QTime(0, 0, 0).addMSecs(int(self.total_frames / self.fps * 1000))
        self.time_label.setText(f"00:00:00 / {dur.toString('hh:mm:ss')}")

        # 清空输出框
        self.output_text.clear()
        self.statusBar().showMessage("视频已打开，播放并实时检测…")

        # 启动定时器
        self.timer.start(int(1000 / self.fps))

    def _toggle_play(self):
        """开始/暂停播放或实时监测"""
        if not self.processor:
            # 如果没有视频源，则打开摄像头进行实时监测
            self.processor = cv2.VideoCapture(0)
            if not self.processor.isOpened():
                self.log_message("无法打开摄像头")
                self.processor = None
                return
            self.video_source = None
            self.is_playing = True
            self.slider.setEnabled(False)
            self.time_label.setText("实时 / 实时")
            self.output_text.clear()
            self.statusBar().showMessage("开始摄像头实时监测")
            self.timer.start(int(1000 / 30))
            return

        # 如果已有视频或摄像头正在播放/检测，则切换暂停/继续
        if self.timer.isActive():
            self.timer.stop()
            self.is_playing = False
            self.statusBar().showMessage("已暂停")
        else:
            self.is_playing = True
            fps = self.fps if self.video_source else 30
            self.timer.start(int(1000 / fps))
            self.statusBar().showMessage("播放/检测中…")

    def _seek_frame(self, f):
        """本地视频跳转到指定帧"""
        if not self.processor or not self.video_source:
            return
        self.processor.set(cv2.CAP_PROP_POS_FRAMES, f)

    def update_frame(self):
        """定时回调：读取一帧，调用检测，并更新界面与输出"""
        if not self.processor:
            return

        ret, frame = self.processor.read()
        if not ret:
            # 如果是本地视频播放到末尾，循环播放
            if self.video_source:
                self.processor.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame = 0
                ret, frame = self.processor.read()
                if not ret:
                    return
            else:
                return

        # 灰度显示
        if self.gray_checkbox.isChecked():
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # 检测并获得带框图像与结果文本
        frame_with_boxes, text = self.detector.detect(frame)

        # 转为 Qt 格式并显示
        rgb_frame = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image).scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))

        # 追加检测结果到右侧文本框
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.output_text.appendPlainText(f"[{timestamp}] {text}")
        self.output_text.verticalScrollBar().setValue(self.output_text.verticalScrollBar().maximum())

        # 本地视频同步更新滑动条和时间标签
        if self.video_source:
            pos = int(self.processor.get(cv2.CAP_PROP_POS_FRAMES))
            self.slider.blockSignals(True)
            self.slider.setValue(pos)
            self.slider.blockSignals(False)
            cur = QTime(0, 0, 0).addMSecs(int(pos / self.fps * 1000))
            tot = QTime(0, 0, 0).addMSecs(int(self.total_frames / self.fps * 1000))
            self.time_label.setText(f"{cur.toString('hh:mm:ss')} / {tot.toString('hh:mm:ss')}")

    def save_frame(self):
        """保存当前帧"""
        if self.processor is not None:
            ret, frame = self.processor.read()
            if ret:
                file_name, _ = QFileDialog.getSaveFileName(
                    self,
                    "保存图像",
                    "",
                    "图像文件 (*.png *.jpg);;所有文件 (*.*)"
                )
                if file_name:
                    cv2.imwrite(file_name, frame)
                    self.log_message(f"图像已保存至: {file_name}")
        else:
            self.log_message("没有可用的视频源")

    def log_message(self, message):
        """添加日志消息到左侧运行日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{timestamp}] {message}")

    def print_logs(self):
        """将右侧“识别结果”文本框内容打印到打印机或导出为 PDF"""
        printer = QPrinter(QPrinter.HighResolution)
        dialog = QPrintDialog(printer, self)
        dialog.setWindowTitle("打印检测日志")
        if dialog.exec_() == QPrintDialog.Accepted:
            self.output_text.print_(printer)
            self.log_message("日志已发送到打印机")
        else:
            self.log_message("取消打印")

    def closeEvent(self, event):
        """窗口关闭事件：释放摄像头/视频"""
        if self.processor is not None:
            self.processor.release()
            self.processor = None
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
