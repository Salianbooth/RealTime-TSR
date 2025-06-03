import sys
import os
import cv2
import queue
import torch
import traceback
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSplitter,
    QVBoxLayout, QHBoxLayout, QTabWidget, QTextEdit, QListWidget,
    QFileDialog, QStatusBar, QSizePolicy, QSlider, QAction, QProgressBar, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, QTime, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QKeySequence, QFont

# --------------------------------------------------------------------------
# RealTimeDetectThread：从队列读取原始帧，用本地 yolo 模型做推理并发射渲染后的帧
# --------------------------------------------------------------------------
class RealTimeDetectThread(QThread):
    processed_frame = pyqtSignal(object)   # 发射处理后的 BGR ndarray
    error_occurred = pyqtSignal(str)       # 发射错误信息（字符串）

    def __init__(self, frame_queue: queue.Queue, device: str = 'cpu'):
        super().__init__()
        self.frame_queue = frame_queue
        self.device = device
        self._running = True

    def run(self):
        try:
            # ------------------------------------------------------------------
            # 1) 获取脚本所在目录（假设本脚本位于 yolov5_local 根目录）
            repo_dir = Path(__file__).parent.resolve()

            # 2) 模型文件 best_1.pt 也在同一目录
            model_path = repo_dir / "best_1.pt"
            if not model_path.exists():
                raise FileNotFoundError(f"找不到模型文件：{model_path}")

            # 3) 调用 torch.hub.load 加载本地仓库（由 repo_dir 提供）
            #    source='local' 保证直接读取本地 hubconf.py
            model = torch.hub.load(
                str(repo_dir),
                'custom',
                path=str(model_path),
                source='local'
            )
            model.to(self.device)
            # 降低推理阈值，可根据需求调整
            model.conf = 0.25
            model.iou = 0.45

        except Exception as e:
            # 加载模型失败时，把异常堆栈发给主线程
            err_msg = "检测线程：模型加载失败！\n" + "".join(traceback.format_exception_only(type(e), e))
            self.error_occurred.emit(err_msg)
            return

        # ------------------------------------------------------------------
        # 4) 模型加载成功后，循环从队列读取帧，做推理并渲染，然后发给主线程
        # ------------------------------------------------------------------
        while self._running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if frame is None:
                # 收到 None 表示主线程准备停止
                break

            try:
                # 推理时指定 size=416，让模型先把图像缩放到 416x416 再做 NMS，速度更快
                results = model(frame, size=416)
                processed = results.render()[0]   # BGR ndarray，已在原始比例上渲染好检测框
                # 发射给主线程
                self.processed_frame.emit(processed)

            except Exception as e:
                # 推理过程出错，将错误信息发给主线程（但线程继续尝试后续帧）
                tb = "".join(traceback.format_exception_only(type(e), e))
                err_msg = "检测线程推理时出错：\n" + tb
                self.error_occurred.emit(err_msg)
                continue

        # 线程退出前，清空并丢弃队列中剩余的帧
        while not self.frame_queue.empty():
            _ = self.frame_queue.get()

    def stop(self):
        self._running = False
        # 放入 None 以解除 get() 阻塞
        try:
            self.frame_queue.put_nowait(None)
        except:
            pass


# --------------------------------------------------------------------------
# TrafficSignUI：主窗口，包括左右两个 QLabel 分别显示“原始”和“检测后”视频
# --------------------------------------------------------------------------
class TrafficSignUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("交通标志检测与识别（实时对比）")
        self.resize(1600, 900)

        self.orig_cam = None       # 用来读取原始视频
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)

        self.frame_queue = None    # 用来传帧给检测线程
        self.detect_thread = None
        self.fps = 30
        self.total_frames = 0

        # 用于“等第一帧处理完再播放”逻辑
        self.waiting_start = False
        self.first_frame = None

        self._setup_ui()
        self._create_actions()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # 左侧主区域：上下两行，第一行放两个并排的固定大小 QLabel，第二行放进度条+滑块+按钮
        left_widget = QWidget()
        v_layout = QVBoxLayout(left_widget)
        v_layout.setContentsMargins(10, 10, 10, 10)

        # —— 第一行：两个并排的 QLabel（左: 原始；右: 检测后），均固定 640×480 ——
        h_top = QHBoxLayout()
        # 原始视频 QLabel
        self.orig_label = QLabel("等待开始")
        self.orig_label.setAlignment(Qt.AlignCenter)
        self.orig_label.setFixedSize(640, 480)
        self.orig_label.setStyleSheet("border:2px dashed #999; background:#000; color:#fff;")
        self.orig_label.setScaledContents(False)

        # 检测后视频 QLabel
        self.proc_label = QLabel("等待开始")
        self.proc_label.setAlignment(Qt.AlignCenter)
        self.proc_label.setFixedSize(640, 480)
        self.proc_label.setStyleSheet("border:2px dashed #999; background:#000; color:#fff;")
        self.proc_label.setScaledContents(False)

        font = QFont()
        font.setPointSize(14)
        self.orig_label.setFont(font)
        self.proc_label.setFont(font)

        h_top.addWidget(self.orig_label)
        h_top.addWidget(self.proc_label)
        v_layout.addLayout(h_top)

        # —— 第二行：进度条（暂不显示） ——
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        v_layout.addWidget(self.progress_bar)

        # —— 第三行：滑块 + 时间标签 ——
        ctrl = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderMoved.connect(self._seek_frame)
        ctrl.addWidget(self.slider)
        self.time_label = QLabel("00:00:00 / 00:00:00")
        self.time_label.setFixedWidth(160)
        ctrl.addWidget(self.time_label)
        v_layout.addLayout(ctrl)

        # —— 第四行：按钮栏 ——
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

        # 右侧：选项卡 (检测结果 小图 + 文本) / (历史记录)
        tabs = QTabWidget()
        # 检测结果 Tab
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

        # 历史记录 Tab
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
        # 文件菜单
        file_menu = self.menuBar().addMenu("文件")
        file_menu.addAction(
            QAction("打开(&O)", self, shortcut=QKeySequence("Ctrl+O"), triggered=self._open_video)
        )
        file_menu.addAction(
            QAction("退出(&Q)", self, shortcut=QKeySequence("Ctrl+Q"), triggered=self.close)
        )
        # 视图菜单
        view_menu = self.menuBar().addMenu("视图")
        self.act_overlay = QAction("检测叠加层", self, checkable=True, checked=True)
        self.act_overlay.triggered.connect(self._toggle_overlay)
        view_menu.addAction(self.act_overlay)

    def _open_video(self):
        """打开视频：先读第一帧给检测线程，等第一帧处理完成后再一起播放"""
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频 (*.mp4 *.avi)")
        if not path:
            return

        # 如果已有摄像头或线程在运行，先释放
        if self.orig_cam:
            self.timer.stop()
            self.orig_cam.release()
            self.orig_cam = None
        if self.detect_thread:
            self.detect_thread.stop()
            self.detect_thread.wait()
            self.detect_thread = None

        # —— 打开原始视频 —— #
        self.orig_cam = cv2.VideoCapture(path)
        if not self.orig_cam.isOpened():
            self.statusBar().showMessage("无法打开视频文件")
            return

        # 读取总帧数和 FPS
        self.total_frames = int(self.orig_cam.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.orig_cam.get(cv2.CAP_PROP_FPS) or 30
        self.slider.setMaximum(self.total_frames)
        self.slider.setEnabled(True)
        dur = QTime(0, 0, 0).addMSecs(int(self.total_frames / self.fps * 1000))
        self.time_label.setText(f"00:00:00 / {dur.toString('hh:mm:ss')}")

        # —— 启动检测线程 —— #
        self.frame_queue = queue.Queue(maxsize=5)
        self.detect_thread = RealTimeDetectThread(self.frame_queue, device='cpu')
        self.detect_thread.processed_frame.connect(self._on_processed_frame)
        self.detect_thread.error_occurred.connect(self._on_detect_error)
        self.detect_thread.start()

        # —— 先读取并缓存第一帧，送给检测线程等待处理 —— #
        ret, frame0 = self.orig_cam.read()
        if not ret:
            self.statusBar().showMessage("无法读取第一帧")
            return

        # 缓存第一帧，等待处理完再播放
        self.first_frame = frame0.copy()
        self.waiting_start = True

        # 把第一帧放入队列，让检测线程开始工作
        try:
            self.frame_queue.put_nowait(frame0.copy())
        except queue.Full:
            pass

        # 左、右都暂时显示“等待开始”
        self.orig_label.setText("等待检测完成")
        self.proc_label.setText("等待检测完成")
        self.statusBar().showMessage("正在等待第一帧检测完成...")

    def _update_frame(self):
        """定时器触发：读一帧原始视频，显示左侧，并把 BGR 帧放入队列给检测线程"""
        if not self.orig_cam:
            return

        ret, frame = self.orig_cam.read()
        if not ret:
            # 视频播放结束
            self.timer.stop()
            self.statusBar().showMessage("播放结束")
            if self.detect_thread:
                self.detect_thread.stop()
            return

        # —— 显示原始帧到左侧 QLabel —— #
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        self.orig_label.setPixmap(
            pix.scaled(self.orig_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

        # —— 将 BGR 帧放入队列（非阻塞），供检测线程处理 —— #
        try:
            self.frame_queue.put_nowait(frame.copy())
        except queue.Full:
            # 队列已满就丢弃本帧
            pass

        # —— 更新滑块和时间标签 —— #
        pos = int(self.orig_cam.get(cv2.CAP_PROP_POS_FRAMES))
        self.slider.blockSignals(True)
        self.slider.setValue(pos)
        self.slider.blockSignals(False)
        cur = QTime(0, 0, 0).addMSecs(int(pos / self.fps * 1000))
        tot = QTime(0, 0, 0).addMSecs(int(self.total_frames / self.fps * 1000))
        self.time_label.setText(f"{cur.toString('hh:mm:ss')} / {tot.toString('hh:mm:ss')}")

    def _on_processed_frame(self, processed_bgr):
        """收到检测线程处理后（BGR ndarray）的帧，显示到右侧 QLabel"""
        # 如果还在等待第一帧（waiting_start），则先把 first_frame 显示到左侧、processed_bgr 显示到右侧，
        # 再一起启动定时器，从第二帧开始正常播放。
        if self.waiting_start:
            # 显示第一帧：左侧显示原始 first_frame，右侧显示 processed_bgr
            rgb0 = cv2.cvtColor(self.first_frame, cv2.COLOR_BGR2RGB)
            h0, w0, _ = rgb0.shape
            img0 = QImage(rgb0.data, w0, h0, 3 * w0, QImage.Format_RGB888)
            pix0 = QPixmap.fromImage(img0)
            self.orig_label.setPixmap(
                pix0.scaled(self.orig_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

            rgb1 = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
            h1, w1, _ = rgb1.shape
            img1 = QImage(rgb1.data, w1, h1, 3 * w1, QImage.Format_RGB888)
            pix1 = QPixmap.fromImage(img1)
            self.proc_label.setPixmap(
                pix1.scaled(self.proc_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

            # 第一步显示完毕后，让定时器开始从“第 2 帧”播放
            self.waiting_start = False
            # 先更新状态栏并启动 timer
            self.statusBar().showMessage("实时播放开始...")
            self.timer.start(int(1000 / self.fps))
            return

        # 如果不是第一帧，表示常规播放时仅需刷新右侧
        rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        self.proc_label.setPixmap(
            pix.scaled(self.proc_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _on_detect_error(self, errmsg):
        """检测线程发来的错误，弹窗提示并状态栏显示"""
        QMessageBox.critical(self, "检测线程出错", errmsg)
        self.statusBar().showMessage("检测线程出错，详见弹窗", 5000)

    def _seek_frame(self, frame_no):
        """拖动滑块跳帧：先清空旧队列，再跳转原始视频到指定帧"""
        if not self.orig_cam:
            return
        if self.frame_queue:
            with self.frame_queue.mutex:
                self.frame_queue.queue.clear()
        self.orig_cam.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

    def _toggle_play(self):
        if not self.orig_cam:
            return
        if self.timer.isActive():
            self.timer.stop()
            self.statusBar().showMessage("已暂停")
        else:
            self.timer.start(int(1000 / self.fps))
            self.statusBar().showMessage("播放中...")

    def _save_frame(self):
        if not self.orig_cam:
            return
        ret, frame = self.orig_cam.read()
        if not ret:
            return
        p, _ = QFileDialog.getSaveFileName(self, "保存当前帧（原始画面）", "", "PNG (*.png)")
        if p:
            cv2.imwrite(p, frame)
            self.statusBar().showMessage(f"已保存 {p}")

    def _toggle_overlay(self):
        """勾选/取消“检测叠加层”：仅清空 / 恢复右侧 QLabel 显示"""
        enabled = self.act_overlay.isChecked()
        if not enabled:
            self.proc_label.clear()
            self.statusBar().showMessage("检测叠加层已关闭", 2000)
        else:
            self.statusBar().showMessage("检测叠加层已开启", 2000)

    def closeEvent(self, event):
        """窗口关闭时，确保停止定时器、释放摄像头、停止检测线程"""
        if self.timer.isActive():
            self.timer.stop()
        if self.orig_cam:
            self.orig_cam.release()
        if self.detect_thread:
            self.detect_thread.stop()
            self.detect_thread.wait()
        event.accept()


if __name__ == '__main__':
    """
    使用说明：
    1. 将本文件 (realtime_tsr.py) 放在你本地克隆后的 yolov5_local 根目录中；
       同级还需要有：hubconf.py, models/, utils/ 等 yolov5 源码，以及 best_1.pt。
       最终目录示例：

       yolov5_local/
       ├─ realtime_tsr.py
       ├─ test_load.py
       ├─ best_1.pt
       ├─ hubconf.py
       ├─ models/
       ├─ utils/
       └─ …(其它 yolov5 文件)…

    2. 安装依赖：
       pip install pyqt5 torch torchvision opencv-python

    3. 先在命令行里测试模型加载：
       python test_load.py
       若输出“模型加载成功！”，才继续下一步。

    4. 运行：
       python realtime_tsr.py

    5. 在弹出的窗口中，点击“打开视频”选择 .mp4/.avi 文件。
       * 此时程序先读取并缓存第一帧，送给检测线程。
       * 检测线程处理完第一帧后，左侧显示第一帧原始画面，右侧显示第一帧检测画面，
         并立刻启动“实时播放”。
       * 之后定时器会按 FPS 继续读取并显示后续帧，左侧播放原始视频，
         右侧并行显示检测结果（由于模型推理速度有限，右侧会有一定滞后，但丢帧逻辑保证不卡死）。

       你可以拖动滑块跳帧（此时会清空旧队列并从新帧继续检测），
       切换“开始/暂停”，或“截屏保存”当前原始帧。

    6. 如果要使用 GPU 推理，请把下面一行：
         self.detect_thread = RealTimeDetectThread(self.frame_queue, device='cpu')
       改为：
         self.detect_thread = RealTimeDetectThread(self.frame_queue, device='cuda:0')
    """

    app = QApplication(sys.argv)
    window = TrafficSignUI()
    window.show()
    sys.exit(app.exec_())
