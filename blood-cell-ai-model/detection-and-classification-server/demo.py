import gc
import os
import torch
from net import UNet
from torchvision import transforms
import numpy as np
from PIL import Image
import time
import file_util as file_util

class DicomModel:
    classes = 1
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    def __init__(self, weights_path: str, work_root: str):
        self.work_root = work_root
        assert os.path.exists(weights_path), f"weights {weights_path} not found."
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("using {} device.".format(self._device))

        self._model = UNet(in_channels=3, num_classes=self.classes + 1, base_c=32)
        self._model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
        self._model.to(self._device)
        self._model.eval()  # 进入验证模式

    def _is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

    def _time_synchronized(self):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        return time.time()

    def to_mask(self, folder, outFolderName):
        """
        识别整个文件夹内图片，并转换
        :param folder: image 目录下 文件夹名
        :return: None
        """
        path = folder
        out_path = os.path.join(self.work_root, "predict_mask", outFolderName)
        file_util.remove_all_files(out_path)

        image_filenames = [x for x in os.listdir(path) if self._is_image_file(x)]
        input_image_path = [os.path.join(path, x) for x in image_filenames]
        output_image_path = [os.path.join(out_path, x) for x in image_filenames]
        for i in range(len(image_filenames)):
            self.model_eval(input_image_path[i], output_image_path[i])

    def model_eval(self, image_path, out_path):
        """
        识别单张图片 转换为mask图片
        :param image_path: 原始图片绝对路径
        :param out_path: 输出mask图片绝对路径
        :return: None
        """
        original_img = Image.open(image_path)
        original_img = np.array(original_img)
        # 如果原图没有目标，则不需要预测
        if (original_img == 0).all():
            return
        height, width = original_img.shape[:2]
        # 定义mask大小
        mask = np.zeros_like(original_img[:, :, 0])
        # 裁剪原图片，裁剪出RoI，同时h,w需要为16的倍数
        pos = np.where(original_img > 0)
        pos_x = 16 - (np.max(pos[0]) - np.min(pos[0])) % 16
        pos_y = 16 - (np.max(pos[1]) - np.min(pos[1])) % 16

        if np.min(pos[0]) - pos_x // 2 < 0 or np.min(pos[1]) - pos_y // 2 < 0:  # 如果超出左边界则右移
            if np.max(pos[0]) + pos_x > height or np.max(pos[1]) + pos_y > width:  # 如果右移超出右边界则不处理
                return
            else:  # 右移
                original_img = original_img[np.min(pos[0]):np.max(pos[0]) + pos_x,
                               np.min(pos[1]):np.max(pos[1]) + pos_y, :]
        else:
            if np.max(pos[0]) + pos_x - pos_x // 2 > height or np.max(
                    pos[1]) + pos_y - pos_y // 2 > width:  # 如果超出右边界则不处理
                return
            else:
                original_img = original_img[np.min(pos[0]) - pos_x // 2:np.max(pos[0]) + pos_x - pos_x // 2,
                               np.min(pos[1]) - pos_y // 2:np.max(pos[1]) + pos_y - pos_y // 2, :]

        original_img = Image.fromarray(original_img)
        # from pil image to tensor and normalize
        data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=self.mean, std=self.std)])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        with torch.no_grad():
            # init model
            img_height, img_width = img.shape[-2:]
            print(img_height, img_width, end=" ")
            init_img = torch.zeros((1, 3, img_height, img_width), device=self._device)
            self._model(init_img)

            t_start = self._time_synchronized()
            output = self._model(img.to(self._device))
            t_end = self._time_synchronized()
            print("inference+NMS time: {}".format(t_end - t_start))

            prediction = output['out'].argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)

            # 将前景对应的像素值改成255(白色)
            prediction[prediction == 1] = 255

            if np.min(pos[0]) - pos_x // 2 < 0 or np.min(pos[1]) - pos_y // 2 < 0:
                if np.max(pos[0]) + pos_x > height or np.max(pos[1]) + pos_y > width:  # 如果右移超出右边界则不处理
                    return
                else:  # 右移
                    mask[np.min(pos[0]):np.max(pos[0]) + pos_x, np.min(pos[1]):np.max(pos[1]) + pos_y] = prediction
            else:
                if np.max(pos[0]) + pos_x - pos_x // 2 > height or np.max(
                        pos[1]) + pos_y - pos_y // 2 > width:  # 如果超出右边界则不处理
                    return
                else:
                    mask[np.min(pos[0]) - pos_x // 2:np.max(pos[0]) + pos_x - pos_x // 2,
                    np.min(pos[1]) - pos_y // 2:np.max(pos[1]) + pos_y - pos_y // 2] = prediction
            mask = Image.fromarray(mask)
            out_parent_folder = os.path.abspath(os.path.dirname(out_path))
            if not os.path.exists(out_parent_folder):
                os.makedirs(out_parent_folder)
            mask.save(out_path)

            # clear
            del mask, original_img, img, init_img, output, pos, prediction
            gc.collect()

if __name__ == '__main__':
    pth = "C:\\user-data\\project\\scu-yang\\dicom-model-server\\last_weight0625_1.pth"
    work_space = "/work-space"
    start = time.time()
    model = DicomModel(pth, work_space)
    model.to_mask("people1")
    end = time.time()
    print("time: ", end - start)