import sys
import os
import cv2
import queue
import torch
import traceback
import numpy as np
from pathlib import Path
from skimage import exposure  # 用于直方图均衡化
import time

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSplitter,
    QVBoxLayout, QHBoxLayout, QTabWidget, QTextEdit, QListWidget,
    QFileDialog, QStatusBar, QSizePolicy, QSlider, QAction, QProgressBar, QMessageBox,
    QComboBox, QCheckBox, QGroupBox  # 添加QCheckBox和QGroupBox
)
from PyQt5.QtCore import Qt, QTimer, QTime, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QKeySequence, QFont


# --------------------------------------------------------------------------
# RealTimeDetectThread：从队列读取原始帧，用本地 yolo 模型做推理并发射渲染后的帧
# --------------------------------------------------------------------------
class RealTimeDetectThread(QThread):
    processed_frame = pyqtSignal(object)  # 发射处理后的 BGR ndarray
    error_occurred = pyqtSignal(str)  # 发射错误信息（字符串）
    detection_log = pyqtSignal(list)  # 发射检测结果日志 (list of dict)

    def __init__(self, frame_queue: queue.Queue, device: str = 'gpu', preprocess_methods: list = None):
        super().__init__()
        self.frame_queue = frame_queue
        self.device = device
        self.preprocess_methods = preprocess_methods or ['none']  # 存储预处理方法列表
        self._running = True
        self.last_frame = None  # 用于存储上一帧，用于降噪处理
        self.last_processed_time = 0  # 用于帧率控制
        self.min_interval = 1.0 / 15  # 最小处理间隔（15fps）
        self.frame_buffer = []  # 用于存储处理后的帧
        self.buffer_size = 2  # 缓冲区大小
        self.alpha = 0.8  # 平滑过渡系数

    def _smooth_transition(self, current_frame, previous_frame):
        """平滑过渡处理"""
        if previous_frame is None:
            return current_frame
        return cv2.addWeighted(current_frame, self.alpha, previous_frame, 1 - self.alpha, 0)

    def _fast_denoise(self, frame):
        """快速降噪处理（优化性能）"""
        try:
            # 缩小图像尺寸以加快处理速度
            h, w = frame.shape[:2]
            # 进一步降低分辨率，例如缩小到八分之一
            scale = 0.125
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
            
            # 使用更轻量级的降噪方法组合
            # 可以考虑只使用高斯模糊或简单的中值滤波如果双边滤波仍然太慢
            # blurred = cv2.GaussianBlur(small, (3, 3), 0) # 轻微高斯模糊
            # denoised = cv2.bilateralFilter(blurred, 5, 50, 50) # 双边滤波，保留边缘
            
            # 尝试使用中值滤波，通常比双边滤波更快，但可能会损失更多细节
            denoised = cv2.medianBlur(small, 3) # 中值滤波，减小核大小以提高速度

            # 恢复原始尺寸
            result = cv2.resize(denoised, (w, h))
            
            return result
            
        except Exception as e:
            self.error_occurred.emit(f"快速降噪处理失败: {str(e)}")
            # 如果降噪处理失败，返回原始帧
            return frame

    def _preprocess_frame(self, frame):
        """应用选定的预处理方法"""
        if 'none' in self.preprocess_methods or not self.preprocess_methods:
            return frame.copy()
        
        processed = frame.copy()
        
        # 检查是否需要跳过处理（帧率控制）
        current_time = time.time()
        if current_time - self.last_processed_time < self.min_interval:
            # 如果缓冲区有帧，使用最新的帧
            if self.frame_buffer:
                return self.frame_buffer[-1]
            return processed
        
        # --- 优化：在较低分辨率下进行图像处理 ---
        h_orig, w_orig = processed.shape[:2]
        # 缩小图像到较低分辨率，例如四分之一尺寸
        scale_factor = 0.5  # 0.5 表示缩小到一半，即面积缩小到四分之一
        # 可以根据需要调整 scale_factor，例如 0.25 或更小
        processed_small = cv2.resize(processed, (int(w_orig * scale_factor), int(h_orig * scale_factor)))

        # 对缩小后的图像应用选中的预处理方法
        for method in self.preprocess_methods:
            try:
                if method == 'clahe':
                    # CLAHE (对比度受限的自适应直方图均衡化) - 增强局部对比度
                    lab = cv2.cvtColor(processed_small, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
                    cl = clahe.apply(l)
                    processed_small = cv2.merge((cl, a, b))
                    processed_small = cv2.cvtColor(processed_small, cv2.COLOR_LAB2BGR)
                    
                elif method == 'histeq':
                    # 全局直方图均衡化 - 增强整体对比度
                    ycrcb = cv2.cvtColor(processed_small, cv2.COLOR_BGR2YCrCb)
                    y, cr, cb = cv2.split(ycrcb)
                    y_eq = exposure.equalize_hist(y)
                    y_eq = (y_eq * 255).astype(np.uint8)
                    processed_small = cv2.merge((y_eq, cr, cb))
                    processed_small = cv2.cvtColor(processed_small, cv2.COLOR_YCrCb2BGR)
                    
                elif method == 'sharpen':
                    # 锐化处理 - 增强边缘和细节
                    kernel = np.array([[-1, -1, -1],
                                       [-1,  9, -1],
                                       [-1, -1, -1]])
                    processed_small = cv2.filter2D(processed_small, -1, kernel)
                    
                elif method == 'denoise':
                    # 快速降噪处理 - 减少图像噪点
                    # 使用中值滤波，核大小可调
                    processed_small = cv2.medianBlur(processed_small, 3)
                    
                elif method == 'contrast':
                    # 对比度亮度调整 - 整体调整亮度和对比度
                    alpha = 1.1  # 对比度控制 (1.0-2.0)
                    beta = 10    # 亮度控制 (0-50)
                    processed_small = cv2.convertScaleAbs(processed_small, alpha=alpha, beta=beta)
            
            except Exception as e:
                # 如果某个预处理方法失败，记录错误并继续处理
                error_msg = f"预处理方法 {method} 执行失败: {str(e)}"
                self.error_occurred.emit(error_msg)
                continue

        # 将处理后的缩小图像放大回原始尺寸
        processed = cv2.resize(processed_small, (w_orig, h_orig))

        # 更新帧缓冲区
        self.frame_buffer.append(processed)
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
        
        # 应用平滑过渡
        if len(self.frame_buffer) > 1:
            processed = self._smooth_transition(processed, self.frame_buffer[-2])
        
        # 更新最后处理时间
        self.last_processed_time = current_time
        return processed

    def run(self):
        try:
            # ------------------------------------------------------------------
            # 1) 获取脚本所在目录（假设本脚本位于 yolov5_local 根目录）
            repo_dir = Path(__file__).parent.resolve()

            # 2) 模型文件 best.pt 也在同一目录
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

            # 优化模型参数，特别是NMS相关的阈值
            model.conf = 0.25  # 降低置信度阈值，提高召回率
            model.iou = 0.45   # 降低IOU阈值，减少重复检测
            model.max_det = 100  # 限制最大检测数量
            model.agnostic = True  # 类别无关的NMS
            model.multi_label = True  # 允许多标签检测

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
                # 添加预处理步骤
                preprocessed = self._preprocess_frame(frame)
                
                # 推理时指定 size=640，让模型先把图像缩放到 640x640 再做 NMS，速度更快
                results = model(preprocessed, size=640)

                # 提取并格式化检测结果
                detections = []
                # results.pandas().xyxy[0] 可以获取包含框、置信度、类别等信息的DataFrame
                # 但直接操作 torch tensor 更灵活，且避免 pandas 依赖
                pred = results.pred[0]  # 获取原始预测结果 (tensor)

                # 遍历每个检测结果
                for *box, conf, cls in pred:
                    label = model.names[int(cls)]
                    x1, y1, x2, y2 = [int(x) for x in box]
                    detections.append({
                        "label": label,
                        "confidence": float(conf),
                        "box": [x1, y1, x2, y2]
                    })

                # 发射检测日志
                self.detection_log.emit(detections)

                # 渲染结果
                processed = results.render()[0]  # BGR ndarray，已在原始比例上渲染好检测框
                processed = processed.copy()  # 确保图像可写

                # 发射渲染后的帧
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
# TrafficSignUI：主窗口，包括左右两个 QLabel 分别显示"原始"和"检测后"视频
# --------------------------------------------------------------------------
class TrafficSignUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("交通标志检测与识别（实时对比）")
        self.resize(1600, 900)

        self.orig_cam = None  # 用来读取原始视频
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)

        self.frame_queue = None  # 用来传帧给检测线程
        self.detect_thread = None
        self.fps = 30
        self.total_frames = 0
        self.preprocess_methods = ['none']  # 默认无预处理

        # 用于"等第一帧处理完再播放"逻辑
        self.waiting_start = False
        self.first_frame = None

        self._setup_ui()
        self._create_actions()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)  # 增加整体边距
        layout.setSpacing(10)  # 增加组件间距

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # 左侧主区域：上下两行，第一行放两个并排的 QLabel，第二行放进度条+滑块+按钮
        left_widget = QWidget()
        v_layout = QVBoxLayout(left_widget)
        v_layout.setContentsMargins(0, 0, 0, 0)  # 移除内部边距
        v_layout.setSpacing(10)  # 增加垂直间距

        # —— 第一行：两个并排的 QLabel（左: 原始；右: 检测后）——
        h_top = QHBoxLayout()
        h_top.setSpacing(10)  # 增加水平间距

        # 原始视频 QLabel
        self.orig_label = QLabel("等待开始")
        self.orig_label.setAlignment(Qt.AlignCenter)
        self.orig_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 允许扩展
        self.orig_label.setStyleSheet("border:2px dashed #999; background:#000; color:#fff;")
        self.orig_label.setMinimumSize(320, 240)  # 设置最小尺寸而不是固定尺寸

        # 检测后视频 QLabel
        self.proc_label = QLabel("等待开始")
        self.proc_label.setAlignment(Qt.AlignCenter)
        self.proc_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 允许扩展
        self.proc_label.setStyleSheet("border:2px dashed #999; background:#000; color:#fff;")
        self.proc_label.setMinimumSize(320, 240)  # 设置最小尺寸而不是固定尺寸

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
        ctrl.setSpacing(10)  # 增加水平间距
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
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # 按钮水平扩展
            btn.clicked.connect(slot)
            btns.addWidget(btn)

        # 添加预处理选项组
        preprocess_group = QGroupBox("预处理选项")
        preprocess_layout = QVBoxLayout()
        
        # 创建复选框
        self.preprocess_checks = {}
        for method, text in [
            ('clahe', 'CLAHE(对比度增强)'),
            ('histeq', '直方图均衡化'),
            ('sharpen', '锐化处理'),
            ('denoise', '降噪处理'),
            ('contrast', '对比度亮度调整')
        ]:
            check = QCheckBox(text)
            check.setToolTip(f"启用{text}预处理")
            check.stateChanged.connect(self._update_preprocess_methods)
            self.preprocess_checks[method] = check
            preprocess_layout.addWidget(check)
        
        preprocess_group.setLayout(preprocess_layout)
        btns.addWidget(preprocess_group)

        v_layout.addLayout(btns)

        splitter.addWidget(left_widget)

        # 右侧：选项卡 (检测结果 小图 + 文本) / (历史记录)
        tabs = QTabWidget()
        # 检测结果 Tab
        result_tab = QWidget()
        res_layout = QVBoxLayout(result_tab)
        res_layout.setContentsMargins(0, 0, 0, 0)  # 移除内部边距
        res_layout.setSpacing(10)  # 增加垂直间距

        # 将 sign_img 更改为可以扩展
        self.sign_img = QLabel("无检测结果")
        self.sign_img.setAlignment(Qt.AlignCenter)
        self.sign_img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 允许扩展
        self.sign_img.setMinimumSize(200, 200)  # 设置最小尺寸
        res_layout.addWidget(self.sign_img)

        # 将 info_text 更改为可以扩展，用于显示日志
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 允许扩展
        res_layout.addWidget(self.info_text)
        tabs.addTab(result_tab, "检测结果")

        # 历史记录 Tab
        hist_tab = QWidget()
        h_layout = QVBoxLayout(hist_tab)
        h_layout.setContentsMargins(0, 0, 0, 0)  # 移除内部边距
        self.hist_list = QListWidget()
        self.hist_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 允许扩展
        h_layout.addWidget(self.hist_list)
        tabs.addTab(hist_tab, "历史记录")

        splitter.addWidget(tabs)
        splitter.setStretchFactor(0, 3)  # 左侧区域占比更大
        splitter.setStretchFactor(1, 1)  # 右侧区域占比更小

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

    def _update_preprocess_methods(self):
        """更新预处理方法列表"""
        selected_methods = []
        for method, check in self.preprocess_checks.items():
            if check.isChecked():
                selected_methods.append(method)
        
        # 如果没有选中任何方法，默认使用'none'
        if not selected_methods:
            selected_methods = ['none']
        
        self.preprocess_methods = selected_methods
        
        # 如果检测线程正在运行，更新其预处理方法
        if self.detect_thread:
            self.detect_thread.preprocess_methods = self.preprocess_methods
            
        # 显示当前选中的预处理方法
        method_names = {
            'clahe': 'CLAHE',
            'contrast': '对比度调整'
        }
        selected_names = [method_names.get(m, m) for m in selected_methods]
        self.statusBar().showMessage(f"当前预处理方法: {', '.join(selected_names)}", 3000)

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
        self.detect_thread = RealTimeDetectThread(
            self.frame_queue, 
            device='cuda:0',
            preprocess_methods=self.preprocess_methods  # 使用预处理方法列表
        )
        self.detect_thread.processed_frame.connect(self._on_processed_frame)
        self.detect_thread.error_occurred.connect(self._on_detect_error)
        self.detect_thread.detection_log.connect(self._on_detection_log)
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

        # 左、右都暂时显示"等待开始"
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

            # 第一步显示完毕后，让定时器开始从"第 2 帧"播放
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

    def _on_detection_log(self, detections):
        """接收检测线程发来的日志信息并显示"""
        self.info_text.clear()  # 清空旧日志
        if not detections:
            self.info_text.append("无检测结果")
            return

        # 汇总检测结果，按类别计数
        detection_summary = {}
        for det in detections:
            label = det["label"]
            if label in detection_summary:
                detection_summary[label] += 1
            else:
                detection_summary[label] = 1

        # 显示汇总信息
        self.info_text.append("检测结果汇总：")
        for label, count in detection_summary.items():
            self.info_text.append(f"- {label}: {count} 个")

        self.info_text.append("\n详细列表：")
        # 显示详细检测信息（标签、置信度、坐标）
        for det in detections:
            label = det["label"]
            conf = det["confidence"]
            box = det["box"]
            self.info_text.append(
                f"- 标签: {label}, 置信度: {conf:.2f}, 坐标: [{box[0]}, {box[1]}, {box[2]}, {box[3]}]")

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
        """勾选/取消"检测叠加层"：仅清空 / 恢复右侧 QLabel 显示"""
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
    app = QApplication(sys.argv)
    window = TrafficSignUI()
    window.show()
    sys.exit(app.exec_())
