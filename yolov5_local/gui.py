import sys
import cv2
import os
import subprocess
import traceback
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSplitter,
    QVBoxLayout, QHBoxLayout, QTabWidget, QTextEdit, QListWidget,
    QFileDialog, QStatusBar, QSizePolicy, QSlider,
    QAction, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer, QTime, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QKeySequence, QFont


class VideoProcessThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    log = pyqtSignal(str)  # 用于将子进程输出传递到 GUI

    def __init__(self, video_path, detect_script):
        super().__init__()
        self.video_path = video_path
        self.detect_script = detect_script

    def run(self):
        try:
            cmd = [
                sys.executable, self.detect_script,
                '--weights', 'best_1.pt',
                '--source', self.video_path,
                '--img', '416',
                '--conf', '0.25',
                '--device', 'cpu',
                '--data', 'D:\\Project\\PyCharm\\Traffic_Sign_Recognition\\RealTime-TSR\\yolov5_local\\data\\tt100k.yaml'
            ]

            self.log.emit(f"开始处理视频: {self.video_path}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='ignore',
                bufsize=1
            )

            for line in process.stdout:
                line = line.rstrip()
                # 将子进程输出发给 GUI
                self.log.emit(line)

                # 解析进度信息
                if 'video' in line and '/' in line:
                    try:
                        parts = line.split('(')[1].split(')')[0].split('/')
                        current = int(parts[0])
                        total = int(parts[1])
                        progress = int((current / total) * 100)
                        self.progress.emit(progress)
                        if progress % 10 == 0:
                            self.log.emit(f"处理进度: {progress}%")
                    except Exception:
                        pass

            process.wait()

            if process.returncode == 0:
                video_name = Path(self.video_path).stem
                output_dir = Path('runs/detect')

                found = False
                exp_dirs = list(output_dir.glob('exp*'))
                if exp_dirs:
                    latest_exp = max(exp_dirs, key=lambda x: x.stat().st_mtime)
                    self.log.emit(f"使用最新的输出目录: {latest_exp}")

                    for file in latest_exp.glob(f"{video_name}*.mp4"):
                        self.log.emit(f"找到处理后的视频: {file}")
                        self.finished.emit(str(file))
                        found = True
                        break
                else:
                    self.log.emit("未找到 exp 目录")

                if not found:
                    error_msg = (
                        f"未找到处理后的视频文件。\n"
                        f"视频名称: {video_name}\n"
                        f"输出目录: {output_dir.absolute()}\n"
                        f"已检查的 exp 目录: {[str(d) for d in exp_dirs]}"
                    )
                    self.log.emit(error_msg)
                    self.error.emit(error_msg)
            else:
                err = f"视频处理脚本返回非零状态码: {process.returncode}"
                self.log.emit(err)
                self.error.emit(err)

        except Exception:
            traceback.print_exc()
            error_msg = "处理视频时出现异常，请查看控制台输出以获得详细信息。"
            self.log.emit(error_msg)
            self.error.emit(error_msg)


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
        self.processed_video_path = None
        self._setup_ui()
        self._create_actions()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # 左侧：视频显示区和控制按钮
        left_widget = QWidget()
        v_layout = QVBoxLayout(left_widget)
        v_layout.setContentsMargins(10, 10, 10, 10)
        self.video_label = QLabel('请点击"打开视频"选择文件')
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("border:2px dashed #999; background:#f5f5f5; color:#666;")
        font = QFont()
        font.setPointSize(20)
        self.video_label.setFont(font)
        v_layout.addWidget(self.video_label)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        v_layout.addWidget(self.progress_bar)

        # 滑动条和时间标签
        ctrl = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderMoved.connect(self._seek_frame)
        ctrl.addWidget(self.slider)
        self.time_label = QLabel("00:00:00 / 00:00:00")
        self.time_label.setFixedWidth(160)
        ctrl.addWidget(self.time_label)
        v_layout.addLayout(ctrl)

        # 按钮行
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

        # 右侧：选项卡，包括检测结果和历史记录
        tabs = QTabWidget()
        # 检测结果页
        result_tab = QWidget()
        res_layout = QVBoxLayout(result_tab)
        self.sign_img = QLabel("无检测结果")
        self.sign_img.setFixedSize(200, 200)
        self.sign_img.setAlignment(Qt.AlignCenter)
        res_layout.addWidget(self.sign_img)
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        res_layout.addWidget(self.info_text)
        tabs.addTab(result_tab, "检测结果")

        # 历史记录页
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

    def _create_actions(self):
        file_menu = self.menuBar().addMenu("文件")
        file_menu.addAction(QAction("打开(&O)", self, shortcut=QKeySequence("Ctrl+O"), triggered=self._open_video))
        file_menu.addAction(QAction("退出(&Q)", self, shortcut=QKeySequence("Ctrl+Q"), triggered=self.close))

        view_menu = self.menuBar().addMenu("视图")
        self.act_overlay = QAction("检测叠加层", self, checkable=True, checked=True)
        self.act_overlay.triggered.connect(self._toggle_overlay)
        view_menu.addAction(self.act_overlay)

    def _open_video(self):
        try:
            path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频 (*.mp4 *.avi)")
            if not path:
                return

            # 清空“检测结果”文本框
            self.info_text.clear()
            self.statusBar().showMessage("正在处理视频...")

            current_dir = os.path.dirname(os.path.abspath(__file__))
            detect_script = os.path.join(current_dir, 'detect.py')

            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            self.process_thread = VideoProcessThread(path, detect_script)
            self.process_thread.progress.connect(self._update_progress)
            self.process_thread.finished.connect(self._on_processing_finished)
            self.process_thread.error.connect(self._on_processing_error)
            # 连接 log 信号，把子进程输出追加到 info_text
            self.process_thread.log.connect(self._append_log)
            self.process_thread.start()
        except Exception:
            traceback.print_exc()
            self.statusBar().showMessage("打开视频时出现异常，请查看控制台。")

    def _append_log(self, text):
        """将子进程输出追加到“检测结果”QTextEdit中"""
        self.info_text.append(text)

    def _update_progress(self, value):
        self.progress_bar.setValue(value)
        self.statusBar().showMessage(f"正在处理视频... {value}%")

    def _on_processing_finished(self, output_path):
        self.progress_bar.setVisible(False)
        self.processed_video_path = output_path

        try:
            if self.camera:
                self.camera.release()
            self.camera = cv2.VideoCapture(self.processed_video_path)
            if not self.camera.isOpened():
                err_msg = "无法打开处理后的视频"
                self.info_text.append(err_msg)
                self.statusBar().showMessage(err_msg)
                return

            self.total_frames = int(self.camera.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.camera.get(cv2.CAP_PROP_FPS) or 30
            self.slider.setMaximum(self.total_frames)
            self.slider.setEnabled(True)
            dur = QTime(0, 0, 0).addMSecs(int(self.total_frames / self.fps * 1000))
            self.time_label.setText(f"00:00:00 / {dur.toString('hh:mm:ss')}")
            self.statusBar().showMessage("视频处理完成")

            self.timer.start(int(1000 / self.fps))
            self.statusBar().showMessage("正在播放...")
        except Exception:
            traceback.print_exc()
            self.statusBar().showMessage("打开处理后的视频时出现异常，请查看控制台。")

    def _on_processing_error(self, error_msg):
        self.info_text.append(f"错误: {error_msg}")
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage(f"处理视频时出错: {error_msg}")

    def _toggle_play(self):
        try:
            if not self.camera:
                return
            if self.timer.isActive():
                self.timer.stop()
                self.statusBar().showMessage("已暂停")
            else:
                self.timer.start(int(1000 / self.fps))
                self.statusBar().showMessage("播放中...")
        except Exception:
            traceback.print_exc()
            self.statusBar().showMessage("播放切换时出现异常，请查看控制台。")

    def _seek_frame(self, f):
        try:
            if not self.camera:
                return
            self.camera.set(cv2.CAP_PROP_POS_FRAMES, f)
        except Exception:
            traceback.print_exc()
            self.statusBar().showMessage("跳转帧时出现异常，请查看控制台。")

    def _save_frame(self):
        try:
            if not self.camera:
                return
            ret, frame = self.camera.read()
            if not ret:
                return
            p, _ = QFileDialog.getSaveFileName(self, "保存图片", "", "PNG (*.png)")
            if p:
                cv2.imwrite(p, frame)
                self.statusBar().showMessage(f"已保存 {p}")
        except Exception:
            traceback.print_exc()
            self.statusBar().showMessage("保存截图时出现异常，请查看控制台。")

    def _update_frame(self):
        try:
            ret, frame = self.camera.read()
            if not ret:
                self.timer.stop()
                self.statusBar().showMessage("播放结束")
                return

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(img)
            self.video_label.setPixmap(pix)

            pos = int(self.camera.get(cv2.CAP_PROP_POS_FRAMES))
            self.slider.blockSignals(True)
            self.slider.setValue(pos)
            self.slider.blockSignals(False)

            cur = QTime(0, 0, 0).addMSecs(int(pos / self.fps * 1000))
            tot = QTime(0, 0, 0).addMSecs(int(self.total_frames / self.fps * 1000))
            self.time_label.setText(f"{cur.toString('hh:mm:ss')} / {tot.toString('hh:mm:ss')}")

        except Exception:
            traceback.print_exc()
            self.timer.stop()
            self.statusBar().showMessage("播放时出错，请查看控制台。")

    def _toggle_overlay(self):
        try:
            self.overlay_enabled = not self.overlay_enabled
            self.statusBar().showMessage("叠加层" + ("开启" if self.overlay_enabled else "关闭"), 2000)
        except Exception:
            traceback.print_exc()
            self.statusBar().showMessage("切换叠加层时出现异常，请查看控制台。")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TrafficSignUI()
    window.show()
    sys.exit(app.exec_())
