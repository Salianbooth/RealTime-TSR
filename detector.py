# detector.py
import cv2
import torch
from pathlib import Path

class Detector:
    def __init__(self):
        # 假设 yolov5_local 位于当前脚本同级目录
        repo_dir = str(Path(__file__).parent / "yolov5_local")
        weights_path = str(Path(__file__).parent / "yolov5_local" / "best_1.pt")

        try:
            # 从本地加载模型
            self.model = torch.hub.load(
                repo_or_dir=repo_dir,
                model="custom",
                path=weights_path,
                source="local",
                device="cpu"
            )
        except Exception as e:
            print("Detector: 加载模型失败:", e)
            self.model = None

    def detect(self, frame_bgr):
        """
        对 BGR 图像进行检测，返回：
          - 带框的 BGR 图像
          - 文本描述（拼成一个字符串：例如 "步行:0.85, 禁止超车:0.72"）
        """
        if self.model is None:
            return frame_bgr, "未加载模型"

        # YOLOv5 要求 RGB 输入
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 推理
        results = self.model(frame_rgb)
        detections = results.xyxy[0]  # Nx6: x1,y1,x2,y2,conf,cls

        texts = []
        if detections is not None and len(detections):
            # 遍历所有检测框，画矩形和标签
            for det in detections:
                x1, y1, x2, y2, conf, cls_idx = det.tolist()
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                conf = float(conf)
                cls_idx = int(cls_idx)

                # 从 model.names 中获取类别名称（已经是中文）
                label = self.model.names[cls_idx]
                texts.append(f"{label}:{conf:.2f}")

                # 随机或固定颜色: BGR
                color = (0, 255, 0)
                # 画矩形
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

                # 在框上方写标签文字
                txt = f"{label} {conf:.2f}"
                # 计算文字大小
                (txt_w, txt_h), baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                # 画填充矩形作为文字背景
                cv2.rectangle(frame_bgr, (x1, y1 - txt_h - 4), (x1 + txt_w, y1), color, -1)
                # 写白色文字
                cv2.putText(frame_bgr, txt, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        text_str = ", ".join(texts) if texts else "未检测到目标"
        return frame_bgr, text_str
