import numpy as np
import random
import cv2
import pandas as pd
import torch
import gc
from glob import glob
import os
from PIL import Image, ImageFile, ImageEnhance
from scipy.ndimage import rotate
import re

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 1000000000000

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
yx_MEAN = [0.485, 0.456, 0.406]
yx_STD = [0.229, 0.224, 0.225]
# WHITE_BALANCE_LST = ["NF-10", "NF-26", "NF-20", "NF-19", "NF-17", "NF-1", "NF-27", "NF-25", "NF-6", "NF-8", "NF-5", "NF-9", "NF-29", "NF-4", "NF-28"]
file_path = r"E:\Multimodal_tumor\qun_zi_liao\all_data\实验集2.xlsx"


lesion_to_label = {
    '鳞癌': 0,
    '腺癌': 1,
    'SCLC': 2,
    'NSCLC': 3
}

def load_lesions(data_dir, pid, lesion_height, lesion_width, lesion_depth, img_extn, is_training, excel_file, num_lesions=4, split="train"):
    files = glob(os.path.join(data_dir, pid, f"*.{img_extn}"))
    files = [file for file in files if not file.endswith(f"_mask.{img_extn}")] # 过滤掉掩码文件

    lesions = []
    keys = []
    lesions_label = []

    if len(files) == 0:
        print(data_dir, pid)

    if num_lesions > 0:
        files = np.random.choice(files, num_lesions, replace=True)

    # 读取Excel文件
    excel_df = pd.read_excel(excel_file)

    for file in files:
        key = os.path.basename(file).split('.')[0]
        prefix_name = '_'.join(key.split('_')[:-1])
        # 从Excel中查找对应的病理标签
        pathology_label = excel_df[excel_df["放疗号"] == prefix_name]["病理"].values
        # 如果找到了对应的病理标签
        if len(pathology_label) > 0:
            lesion_label = pathology_label[0]  # 提取第一个匹配的病理标签
        else:
            lesion_label = -1  # 没有找到匹配的病理标签
        lesion = [
            np.asarray(Image.open(file.replace("/0/", "/-1/")).convert("L")),
            np.asarray(Image.open(file).convert("L")),
            np.asarray(Image.open(file.replace("/0/", "/1/")).convert("L")),
        ]
        # Resize the entire 3D image to the target dimensions
        lesion = resize_3d_image(np.array(lesion), lesion_height, lesion_width, lesion_depth)

        if is_training:
            lesion = random_transform_np(lesion, max_rotation=30, pad_value=0)

        lesions.append(lesion)
        keys.append(key)

        lesions_label.append(lesion_to_label[lesion_label])

    if len(lesions) == 0:
        print(data_dir, pid, "+++++++") # debug
        return torch.Tensor(1)
    lesions = np.stack(lesions, axis=0)
    lesions = lesions.astype(float)
    lesions /= 255.0
    lesions -= yx_MEAN
    lesions /= yx_STD

    lesions = torch.Tensor(lesions.transpose(0, 4, 3, 1, 2)).float() # (N_l, 4, 3, 224, 224)
    lesions_label = torch.LongTensor(lesions_label)
    return lesions, keys, lesions_label

def random_transform_np(img_np, max_rotation=10, pad_value=255):
    # img_np: (H, W, D)  三维图像
    h, w, d = img_np.shape  # 获取图像的高度、宽度和深度（即切片数量）

    # flip the bag(flip across length axis or spatial axes)
    if random.random() < 0.5:
        flip_code = random.choice([0, 1, 2])  # 0 for horizontal and 1 for vertical
        img_np = np.flip(img_np, axis=flip_code)

    # rotate the image (rotate along the length axis or spatial axes)
    if random.random() < 0.5:
        angle = random.choice(np.arange(-max_rotation, max_rotation + 1).tolist())
        # Choose a random axis for the 3D rotation (x, y, or z)
        axis = random.choice([0, 1, 2])  # 0 = x-axis, 1 = y-axis, 2 = z-axis
        # Perform the 3D rotation using scipy.ndimage.rotate
        img_np = rotate(img_np, angle, axes=(axis, (axis + 1) % 3), reshape=True, mode='nearest', cval=pad_value)

    # random crop and scale
    if random.random() < 0.5:
        x = random.randint(0, w - w // 4)
        y = random.randint(0, h - h // 4)
        z = random.randint(0, d - d // 4)
        img_np = img_np[y:, x:, z: ]
        img_np = cv2.resize(img_np, (w, h, d))

    return img_np

def resize_3d_image(img_np, target_height, target_width, target_depth):
    """
    Resize a 3D image (height, width, depth) to the target size (target_height, target_width, target_depth).
    """
    # img_np is assumed to be a 3D image (height, width, depth)
    h, w, d = img_np.shape

    # Resize each slice (depth dimension) to match the target height and width
    resized_slices = []
    for i in range(d):
        slice = img_np[:, :, i]  # Get the ith slice (2D array)
        resized_slice = cv2.resize(slice, (target_width, target_height))  # Resize slice
        resized_slices.append(resized_slice)

    # Convert the resized slices back into a 3D image
    resized_image = np.stack(resized_slices, axis=-1)  # (target_height, target_width, target_depth)

    # If depth is different from the target depth, resize depth dimension (this might involve interpolation)
    if d != target_depth:
        resized_image = cv2.resize(resized_image, (target_width, target_height, target_depth), interpolation=cv2.INTER_LINEAR)

    return resized_image

def convert_label(label_type, OS, OSCensor, liaoxiao, bingli):
    if label_type == "ORR":
        if liaoxiao >= 1:
            return 1
        elif liaoxiao < 1:
            return 0
        else:
            return -1
    elif label_type == "ORR_OS180":
        if liaoxiao >= 1:
            return 1
        elif liaoxiao < 1:
            return 0
        else:
            cutoff = 180 # about six months
            if OS > cutoff:
                return 1
            elif OSCensor == 1 and OS <= cutoff:
                return 0
            else:
                return -1
    elif label_type == "BL":
        if bingli == "鳞癌":
            return 0
        elif bingli == "腺癌":
            return 1
        elif bingli == "SCLC":
            return 2
        elif bingli == "NSCLC":
            return 3
        else:
            return -1
    else:
        raise NotImplementedError

def ccd_sex(x):
    # 患者：性别
    x = str(x).strip()
    if x in ["男"]:
        return torch.FloatTensor([0.0, 1.0])
    elif x in ["女"]:
        return torch.FloatTensor([1.0, 0.0])
    else:
        raise NotImplementedError

def ccd_age(x, split_age=60):
    # 患者：年龄
    if x <= split_age:
        return torch.FloatTensor([1.0, 0.0])
    elif x > split_age:
        return torch.FloatTensor([0.0, 1.0])
    else:
        raise NotImplementedError

def ccd_buwei(x):
    # 患者：肿瘤部位
    x = str(x).strip()
    if x in ["GEJ"]:
        return torch.FloatTensor([1.0, 0.0])
    elif x in ["non-GJE"]:
        return torch.FloatTensor([0.0, 1.0])
    else:
        raise NotImplementedError

def ccd_leixing(x):
    # 患者：治疗线数
    x = str(x).strip()
    if "放疗" in x:
        return torch.FloatTensor([1.0, 0.0, 1.0])
    elif "序贯放化疗":
        return torch.FloatTensor([0.0, 1.0, 0.0])
    elif "同步放化疗":
        return torch.FloatTensor([0.0, 0.0, 1.0])
    else:
        raise  NotImplementedError

def ccd_time(x):
    # 患者：开始治疗时间
    if int(x) >= 2021 and int(x) < 2022:
        return torch.FloatTensor([1.0, 0.0, 0.0])
    elif int(x) >= 2022 and int(x) < 2023:
        return torch.FloatTensor([0.0, 1.0, 0.0])
    elif int(x) >= 2023 and int(x) <= 2024:
        return torch.FloatTensor([0.0, 0.0, 1.0])
    else:
        raise NotImplementedError

def ccd_tnm_fenqi(x):
    # 病理：T/N/M分期
    t_match = re.search(r'T(\d+[a-zA-Z]?)', x)  # 匹配 T 部分
    n_match = re.search(r'N(\d+[a-zA-Z]?)', x)  # 匹配 N 部分
    m_match = re.search(r'M(\d)', x)  # 匹配 M 部分

    # 提取 T, N, M 分期
    t_stage = t_match.group(0) if t_match else 'T0'  # 如果找不到 T，则默认是 T0
    n_stage = n_match.group(0) if n_match else 'N0'  # 如果找不到 N，则默认是 N0
    m_stage = m_match.group(0) if m_match else 'M0'  # 如果找不到 M，则默认是 M0

    t_unique = sorted(set(t_stage))
    n_unique = sorted(set(n_stage))
    m_unique = sorted(set(m_stage))

    # 创建动态的独热编码映射
    t_label = {stage: [1 if i == idx else 0 for i in range(len(t_unique))] for idx, stage in enumerate(t_unique)}
    n_label = {stage: [1 if i == idx else 0 for i in range(len(n_unique))] for idx, stage in enumerate(n_unique)}
    m_label = {stage: [1 if i == idx else 0 for i in range(len(m_unique))] for idx, stage in enumerate(m_unique)}
    # 获取对应的分类标签并转换为向量
    t_class = t_label.get(t_stage, [0] * len(t_unique))  # 如果找不到对应的 T 值，返回全 0 向量
    n_class = n_label.get(n_stage, [0] * len(n_unique))  # 如果找不到对应的 N 值，返回全 0 向量
    m_class = m_label.get(m_stage, [0, 0])  # 如果找不到对应的 M 值，返回全 0 向量

    # 拼接 T, N, M 分期的独热编码向量
    encoded = t_class + n_class + m_class

    # 返回独热编码的张量
    return torch.FloatTensor(encoded)

def ccd_fenqi(x):
    # 病理：总分期
    # 检查输入格式
    if isinstance(x, str) and len(x) == 2:
        digit = x[0]  # 获取数字部分
        letter = x[1]  # 获取字母部分

        # 数字部分的映射
        if digit == '1':
            digit_encoding = [1.0, 0.0, 0.0, 0.0]  # 映射到类别1
        elif digit == '2':
            digit_encoding = [0.0, 1.0, 0.0, 0.0]  # 映射到类别2
        elif digit == '3':
            digit_encoding = [0.0, 0.0, 1.0, 0.0]  # 映射到类别3
        elif digit == '4':
            digit_encoding = [0.0, 0.0, 0.0, 1.0]  # 映射到类别4
        else:
            raise NotImplementedError(f"Unsupported digit: {digit}")

        # 字母部分的映射
        if letter == 'A':
            letter_encoding = [1.0, 0.0, 0.0]  # 映射到字母A
        elif letter == 'B':
            letter_encoding = [0.0, 1.0, 0.0]  # 映射到字母B
        elif letter == 'C':
            letter_encoding = [0.0, 0.0, 1.0]  # 映射到字母C
        else:
            raise NotImplementedError(f"Unsupported letter: {letter}")

        # 拼接数字和字母的独热编码
        encoding = digit_encoding + letter_encoding

        return torch.FloatTensor(encoding)

    elif str(x) == str(np.nan):
        return torch.FloatTensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 返回默认值（可根据需求调整）

    else:
        raise NotImplementedError

def ccd_fenhua(x):
    # 病理：分化程度
    if x in ["低分化"]:
        return torch.FloatTensor([1.0, 0.0, 0.0, 0.0])
    elif x in ["中分化"]:
        return torch.FloatTensor([0.0, 1.0, 0.0, 0.0])
    elif x in ["高分化"]:
        return torch.FloatTensor([0.0, 0.0, 1.0, 0.0])
    elif x in ["弥漫型"]:
        return torch.FloatTensor([0.0, 0.0, 0.0, 1.0])
    elif str(x) == str(np.nan):
        return torch.FloatTensor([0.0, 0.0, 0.0, 0.0])
    else:
        return torch.FloatTensor([0.0, 0.0, 0.0, 0.0])
        raise NotImplementedError

def ccd_yx_flag(x):
    # 患者：第二次定位时间
    if int(x) >= 2021 and int(x) < 2022:
        return torch.FloatTensor([1.0, 0.0, 0.0])
    elif int(x) >= 2022 and int(x) < 2023:
        return torch.FloatTensor([0.0, 1.0, 0.0])
    elif int(x) >= 2023 and int(x) <= 2024:
        return torch.FloatTensor([0.0, 0.0, 1.0])
    else:
        raise NotImplementedError

def ccd_jibing(x):
    # 疾病类型的映射
    disease_types = {
        "肺气肿": [1.0, 0.0, 0.0],
        "间质性肺病": [0.0, 1.0, 0.0],
        "肺结核": [0.0, 0.0, 1.0],
    }
    # 如果 x 为 0，返回全零的独热编码
    if x == "0":
        return torch.FloatTensor([0.0, 0.0, 0.0])

    # 如果 x 包含 "+"，拆分疾病类型
    if "+" in x:
        diseases = x.split("+")

        # 初始化全零的独热编码
        encoding = torch.FloatTensor([0.0, 0.0, 0.0])

        # 遍历疾病列表，为每个疾病添加其独热编码
        for disease in diseases:
            disease = disease.strip()  # 去除两侧空格
            if disease in disease_types:
                encoding += torch.FloatTensor(disease_types[disease])
            else:
                raise NotImplementedError(f"Unknown disease type: {disease}")

        return encoding

    else:
        # 如果只有一个疾病类型
        disease = x.strip()  # 去除两侧空格
        if disease in disease_types:
            return torch.FloatTensor(disease_types[disease])
        else:
            raise NotImplementedError(f"Unknown disease type: {disease}")

def ccd_xiyan(x):
    # 是否吸烟
    if x == 1:
        return torch.FloatTensor([1.0, 0.0])
    elif x == 0:
        return torch.FloatTensor([0.0, 1.0])
    elif str(x) == str(np.nan):
        return torch.FloatTensor([0.0, 0.0])
    else:
        raise NotImplementedError

def ccd_xiyan_num(x):
    if isinstance(x, float) and np.isnan(x):  # 检查 x 是否为 NaN
        return torch.FloatTensor([0.0, 0.0])  # 如果是 NaN，返回 [0.0, 0.0]
    else:
        return torch.FloatTensor([x, 1.0])  # 如果不是 NaN，返回对应的 [x, 1.0]

def ccd_shoushu(x):
    # 是否手术
    if x == 1:
        return torch.FloatTensor([1.0, 0.0])
    elif x == 0:
        return torch.FloatTensor([0.0, 1.0])
    elif str(x) == str(np.nan):
        return torch.FloatTensor([0.0, 0.0])
    else:
        raise NotImplementedError

def ccd_zl_per(x):
    # 是否免疫治疗
    if x == 1:
        return torch.FloatTensor([1.0, 0.0])
    elif x == 0:
        return torch.FloatTensor([0.0, 1.0])
    elif str(x) == str(np.nan):
        return torch.FloatTensor([0.0, 0.0])
    else:
        raise NotImplementedError

def ccd_feiyan(x):
    # 是否有放射性肺炎
    if x == 1:
        return torch.FloatTensor([1.0, 0.0])
    elif x == 0:
        return torch.FloatTensor([0.0, 1.0])
    elif str(x) == str(np.nan):
        return torch.FloatTensor([0.0, 0.0])
    else:
        raise NotImplementedError

def ccd_shiguanyan(x):
    # 是否有放射性食管炎
    if x == 1:
        return torch.FloatTensor([1.0, 0.0])
    elif x == 0:
        return torch.FloatTensor([0.0, 1.0])
    elif str(x) == str(np.nan):
        return torch.FloatTensor([0.0, 0.0])
    else:
        raise NotImplementedError