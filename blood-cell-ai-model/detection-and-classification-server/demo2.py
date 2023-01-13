import torch.nn as nn
import numpy as np
import os
import torch
from torchvision import transforms
from PIL import Image
import time
from efficientnet_pytorch import EfficientNet
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, letterbox
import cv2


class objectDetectionAndClassification:
    classes = 1
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    cmap1 = {
                "0": '单核细胞',
                "1": '血小板',
                "2": "中幼红细胞",
                "3": "晚幼红细胞",
                "4": '成熟红细胞',
                "5": '中性晚幼粒细胞',
                "6": '中性杆状核粒细胞',
                "7": '中性分叶核粒细胞',
                "8": "嗜酸性晚幼粒细胞",
                "9": "成熟淋巴细胞",
            },
    cmap2 = {
                "0": '103',
                "1": '141',
                "2": "2",
                "3": "3",
                "4": '4',
                "5": '56',
                "6": '57',
                "7": '58',
                "8": "60",
                "9": "92",
            },

    def __init__(self, detection_weights_path: str, classification_weights_path: str, work_root: str):
        self.work_root = work_root
        assert os.path.exists(detection_weights_path), f"weights {detection_weights_path} not found."
        assert os.path.exists(classification_weights_path), f"weights {classification_weights_path} not found."
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("using {} device.".format(self._device))
        # 在这里定义出检测模型
        self._detection_model_weights = 'yolov5.pt'
        self._detection_model = attempt_load(self._detection_model_weights, self._device)  # gpu
        # self._detection_model = attempt_load(self._detection_model_weights, self._device).float()#cpu

        self._detection_model.to(self._device)
        self._detection_model.eval()

        self._classification_model = EfficientNet.from_name('efficientnet-b4')
        in_features = self._classification_model._fc.in_features
        self._classification_model._fc = nn.Linear(in_features=in_features, out_features=10, bias=True)
        self._classification_model.load_state_dict(torch.load(classification_weights_path, map_location='cpu'))
        self._classification_model.to(self._device)
        self._classification_model.eval()  # 进入验证模式

    def _is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

    def preprocess(self, img):
        img0 = img.copy()
        img = letterbox(img, new_shape=640)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self._device)
        img = img.half()  # gpu
        # img = img.float()#cpu
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img0, img

    def detection_model_eval(self, image_path):
        """
                        识别单张图片 转换为mask图片
                        :param image_path: 原始图片绝对路径
                        :return: image_path, [[检测框位置信息], ……]
                        """
        im = cv2.imread(image_path)
        # im = torch.from_numpy(im).to(self._device)
        im0, img = self.preprocess(im)

        pred = self._detection_model(img, augment=False)[0]
        pred = pred.float()
        pred = non_max_suppression(pred, 0.4, 0.3)

        pred_boxes = []
        # image_info = {}
        count = 0
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # for *x, conf, cls_id in det:
                for x in det:
                    # lbl = self.names[int(cls_id)]
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append([x1, y1, x2, y2])
                    count += 1
                    # key = '{}-{:02}'.format(lbl, count)
                    # image_info[key] = ['{}×{}'.format(x2 - x1, y2 - y1), np.round(float(conf), 3)]

        # im = self.plot_bboxes(im, pred_boxes)
        return image_path, pred_boxes

    def classification_model_eval(self, detection_result):
        """
        识别单张图片 转换为mask图片
        :param detection_result: 原始图片绝对路径与检测框位置信息
        :return: [[[检测框位置信息], 分类结果], ……]
        """
        original_img = Image.open(detection_result[0])
        original_img = np.array(original_img)
        # 如果原图没有目标，则不需要预测
        if (original_img == 0).all():
            return
        height, width = original_img.shape[:2]
        original_img = Image.fromarray(original_img)
        final_output = []
        for f in detection_result[1]:
            img = original_img.crop(f)
            # from pil image to tensor and normalize
            data_transform = transforms.Compose([transforms.Resize(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=self.mean, std=self.std)])
            img = data_transform(img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)
            _, output = self._classification_model(img.to(self._device)).max(1)
            # 将output返回为细胞对应序号
            output = self.cmap2[0][str(output.item())]
            final_output.append([f, output])
        return final_output


if __name__ == '__main__':
    d_pth = "yolov5.pt"
    c_pth = "efficientnet_b4_model.pth"
    p_pth = "1086hp90j4rgsvdqeo12nood7l.jpg"
    work_space = "/work-space"
    start = time.time()
    model = objectDetectionAndClassification(d_pth, c_pth, work_space)
    result = model.detection_model_eval(p_pth)
    f_result = model.classification_model_eval(result)
    print(result, f_result)
    end = time.time()
    print("time: ", end - start)
