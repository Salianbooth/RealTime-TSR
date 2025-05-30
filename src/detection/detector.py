import sys
import pathlib
import cv2
import torch
import numpy as np

# 将项目根目录加入 Python 模块搜索路径
FILE = pathlib.Path(__file__).resolve()
PROJECT_ROOT = FILE.parents[2]  # RealTime-TSR 根目录
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yolov5_local.models.common import DetectMultiBackend
from yolov5_local.utils.general import non_max_suppression
from yolov5_local.utils.torch_utils import select_device
from yolov5_local.utils.augmentations import letterbox

class YOLOv5Detector:
    def __init__(self,
                 weights='yolov5s.pt',
                 device='',        # '' 自动选择 cuda 或 cpu
                 img_size=(640, 640),
                 conf_thres=0.25,
                 iou_thres=0.45):
        # 1. 设备
        self.device = select_device(device)
        # 2. 模型加载
        self.model = DetectMultiBackend(weights, device=self.device, dnn=False, data=None)
        self.stride, self.names = self.model.stride, self.model.names
        # 3. 输入尺寸与阈值
        self.img_size = img_size
        self.conf_thres, self.iou_thres = conf_thres, iou_thres

    def detect(self, frame: np.ndarray):
        # 1. 预处理：letterbox + BGR->RGB + HWC->CHW
        img = letterbox(frame, self.img_size, stride=self.stride, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        tensor = torch.from_numpy(img).to(self.device).float() / 255.0
        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)

        # 2. 推理
        pred = self.model(tensor, augment=False, visualize=False)
        # 3. NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

        # 4. 解析并映射坐标
        dets = []
        h0, w0 = frame.shape[:2]
        gain = min(self.img_size[0] / h0, self.img_size[1] / w0)
        pad = ((self.img_size[1] - w0 * gain) / 2, (self.img_size[0] - h0 * gain) / 2)
        for det in pred[0]:
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            x1 = int((x1 - pad[0]) / gain)
            y1 = int((y1 - pad[1]) / gain)
            x2 = int((x2 - pad[0]) / gain)
            y2 = int((y2 - pad[1]) / gain)
            dets.append((x1, y1, x2, y2, self.names[int(cls)], float(conf)))
        return dets
