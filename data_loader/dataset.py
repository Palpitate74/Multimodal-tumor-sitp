from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import pandas as pd
from copy import deepcopy
import pickle
from data_loader.utils import *
from utils.print_utils import *
import numpy as np


# CTV_BLACK_LST = []
# PTV_BLACK_LST = []
BLACK_LST = []
YX_BLACK_LST = []
# BL_BLACK_LST = []
yx_black_lst_v1 = []
# bl_black_lst_v1 = []
file_path = r"E:\Multimodal_tumor\qun_zi_liao\all_data\实验集2.xlsx"


class YXDataset(Dataset):
    def __init__(self, opts, split, split_file, printer, ignore=True, cohort=None):
        super().__init__()

        self.split = split
        self.opts = opts
        self.is_training = ("train" == split)

        self.yx_img_dir = opts.yx_img_dir
        self.yx_img_extn = opts.yx_img_extn
        self.yx_num_lesions = opts.yx_num_lesions
        self.yx_lesion_size = opts.yx_lesion_size

        pd_data = pd.read_excel(split_file)
        self.yx_num, self.blyx_num = 0, 0
        self.id_lst, self.name_lst, self.yx_pid_lst, self.liaoxiao_lst = [], [], [], []
        self.OS_lst, self.OSCensor_lst = [], []
        self.fangan_lst, self.label_lst, self.time_lst = [], [], []
        self.sex_lst, self.age_lst, self.jibing_lst, self.leixing_lst, self.buwei_lst = [], [], [], [], []
        self.zl_per_lst = []
        self.feiyan_lst, self.shiguanyan_lst = [], []
        self.xiyan_lst, self.xiyan_num_lst, self.shoushu_lst, self.mianyi_lst = [], [], [], []
        self.tnm_fenqi_lst, self.fenqi_lst = [], []
        self.fev1_lst, self.fev1_per_lst, self.fvc_lst, self.fvc_per_lst, self.dlco_lst, self.dlco_per_lst, self.dlcova_lst, self.dlcova_per_lst, self.tumor_lst = [], [], [], [], [], [], [], [], []
        self.hb_lst, self.leu_lst, self.neu_lst, self.pla_lst, self.til_lst = [], [], [], [], []
        self.yx_flag_lst = []

        for (id, name, yx_pid, start_time, liaoxiao, OS, OSCensor, fangan,
            sex, age, jibing, leixing,
            xiyan, xiyan_num, shoushu, zl_per,
            tnm_fenqi, fenqi,
            fev1, fev1_per, fvc, fvc_per, dlco, dlco_per, dlcova, dlcova_per,
            hb, leu, neu, pla, til,
            feiyan, shiguanyan,
            tumor, buwei,
            yx_flag, bingli
            ) in zip(
            pd_data["放疗号"], pd_data["姓名"], pd_data["影像勾画编号"], pd_data["第一次定位时间"], pd_data["最佳疗效"],
            pd_data["OS"], pd_data["OSCensor"], pd_data["化疗方案"],
            pd_data["性别"], pd_data["年龄"], pd_data["肺部疾病"], pd_data["治疗类型"],
            pd_data["是否吸烟"],pd_data["吸烟指数"],pd_data["是否手术"],pd_data["是否免疫治疗"],
            pd_data["肿瘤T/N/M分期"], pd_data["肿瘤总分期"],
            pd_data["FEV1"], pd_data["FEV1(实/预）%"], pd_data["FVC"], pd_data["FVC(实/预）"], pd_data["DLCO"], pd_data["DLCO(实/预）"], pd_data["DLCO/VA"], pd_data["DLCO/VA(实/预）"],
            pd_data["Hb最低(放疗期间)"], pd_data["白细胞最低"], pd_data["中性粒细胞最低"], pd_data["血小板最低"], pd_data["淋巴细胞最低"],
            pd_data["放射性肺炎（复诊激素药物）"], pd_data["放射性食管炎（进食疼痛/梗阻）"],
            pd_data["肿瘤占比"], pd_data["转移部位"],
            pd_data["第一次定位时间"], pd_data["病理"]):

            if id in BLACK_LST:
                continue

            label = convert_label(self.opts.label_type, OS, OSCensor, liaoxiao, bingli)
            if label < -1 or (yx_pid in YX_BLACK_LST):
                continue
            if str(yx_pid) != str(np.nan) and yx_flag == -1:
                continue

            if cohort is not None:
                assert cohort in ["her2", "ci", "all"], (cohort)
                if cohort == "ci" and fangan == 0:
                    continue
                elif cohort == "her2" and fangan == 1:
                    continue
                elif cohort == "all":
                    pass

            if opts.model == "yingxiang" and (str(yx_pid) == str(np.nan) or yx_flag == -1):
                continue


            self.yx_num += int(str(yx_pid) != str(np.nan))

            self.id_lst.append(id)
            self.name_lst.append(name)
            self.yx_pid_lst.append(str(int(yx_pid)) if not np.isnan(yx_pid) else str(np.nan))
            self.liaoxiao_lst.append("NA" if str(liaoxiao) == str(np.nan) else liaoxiao)

            self.OS_lst.append(OS)
            self.OSCensor_lst.append(OSCensor)


            self.fangan_lst.append(fangan)
            self.label_lst.append(label)

            self.sex_lst.append(sex)
            self.age_lst.append(age)
            self.jibing_lst.append(jibing)
            self.leixing_lst.append(leixing)
            self.time_lst.append(str(start_time).split("/")[0])

            self.xiyan_lst.append(xiyan)
            self.xiyan_num_lst.append(xiyan_num)
            self.shoushu_lst.append(shoushu)
            self.zl_per_lst.append(zl_per)

            self.tnm_fenqi_lst.append(tnm_fenqi)
            self.fenqi_lst.append(fenqi)
            self.fev1_lst.append(fev1)
            self.fev1_per_lst.append(fev1_per)
            self.fvc_lst.append(fvc)
            self.fvc_per_lst.append(fvc_per)
            self.dlco_lst.append(dlco)
            self.dlco_per_lst.append(dlco_per)
            self.dlcova_lst.append(dlcova)
            self.dlcova_per_lst.append(dlcova_per)

            self.hb_lst.append(hb)
            self.leu_lst.append(leu)
            self.neu_lst.append(neu)
            self.pla_lst.append(pla)
            self.til_lst.append(til)

            self.feiyan_lst.append(feiyan)
            self.shiguanyan_lst.append(shiguanyan)

            self.tumor_lst.append(tumor)
            self.buwei_lst.append(buwei)

            self.yx_flag_lst.append(yx_flag)

        self.diag_labels = deepcopy(self.label_lst) # 进行深拷贝
        self.n_classes = len(np.unique(self.diag_labels)) # 获取标签种类数
        self.printer = printer

        print_info_message('Samples in {}: {}\t(yx={})'.format(
            split_file, self.__len__(), self.yx_num), self.printer)
        print_info_message('-- {} ({:.2f}%) Non-response | {} ({:.2f}%) Response | {} ({:.2f}%) Others'.format(
            sum(np.asarray(self.label_lst) == 0), 100.0 * sum(np.asarray(self.label_lst) == 0) / self.__len__(),
            sum(np.asarray(self.label_lst) == 1), 100.0 * sum(np.asarray(self.label_lst) == 1) / self.__len__(),
            sum(np.asarray(self.label_lst) == -1), 100.0 * sum(np.asarray(self.label_lst) == -1) / self.__len__(),
        ), self.printer)

    def __len__(self):
        return len(self.yx_pid_lst)

    def _generate_mask_bags_label(self, mask_bags):
        # mask_bags: (N_B, B_H, B_W, B_L)
        mask_bags_label = []
        mask_bags = mask_bags.reshape((mask_bags.shape[0], -1, mask_bags.shape[-1]))  # (N_B, B_H * B_W, B_L)
        for nb in range(mask_bags.shape[0]):
            # 对每个位置 (B_H * B_W)，计算该位置的最常见标签
            mask_bags_label.append(np.argmax(np.bincount(mask_bags[nb].flatten())))  # 使用 flatten 扁平化 B_L 维度
        mask_bags_label = torch.LongTensor(mask_bags_label)  # (N_B,)
        return mask_bags_label
    # def _generate_mask_bags_label(self, mask_bags):
    #     # mask_bags: (N_B, B_H, B_W, B_L)
    #     mask_bags_label = []
    #     mask_bags = mask_bags.reshape((mask_bags.shape[0], -1, mask_bags.shape[-1])) # (N_B, B_H * B_W, B_L)
    #     mask_bags = mask_bags.reshape((mask_bags.shape[0], -1)) # (N_B, B_H * B_W * B_L)
    #     for nb in range(mask_bags.shape[0]):
    #         # 对每个位置 (B_H * B_W)，计算该位置的最常见标签
    #         mask_bags_label.append(np.argmax(np.bincount(mask_bags[nb])))
    #     mask_bags_label = torch.LongTensor(mask_bags_label) # (N_B,)
    #     return mask_bags_label

    def _generate_mask_words_label(self, mask_words):
        # mask_words: (N_B, N_W, W_H, W_W, W_L)
        mask_words_label = []
        mask_words = mask_words.reshape((mask_words.shape[0], mask_words.shape[1], -1, mask_words[-1])) # (N_B, N_W, W_H * W_W, W_L)
        for nb in range(mask_words.shape[0]):
            mask_words_label_tmp = []
            for nw in range(mask_words.shape[1]):
                mask_words_label_tmp.append(np.argmax(np.bincount(mask_words[nb, nw].flatten())))
            mask_words_label.append(mask_words_label_tmp)
        mask_words_label = torch.LongTensor(mask_words_label) # (N_B, N_W)
        return mask_words_label

    def _load_yx_data(self, index):
        if self.yx_pid_lst[index] != str(np.nan):
            yx_pid = self.yx_pid_lst[index]
            lesions, keys, lesions_label = load_lesions(self.yx_img_dir, yx_pid,
                            self.yx_lesion_size, self.yx_lesion_size, self.yx_lesion_size,
                            self.yx_img_extn, is_training=self.is_training, excel_file=file_path, num_lesions=self.opts.yx_num_lesions, split=self.split)

            # radiomics_file = os.path.join(self.opts.yx_rad_dir, f"radiomics_{yx_pid}_norm.csv")
            # radiomics_data = pd.read_csv(radiomics_file).values
            # radiomics_dict = {}
            # for line in radiomics_data:
            #     radiomics_dict[str(line[0])] = torch.FloatTensor(np.nan_to_num(np.asarray(line[1:], dtype=np.float32), 0.0)).float() # (736,)
            # radiomics_feat = torch.stack([radiomics_dict[key] for key in keys], dim=0) # (N_B, 736)
            flag = 1
        else:
            lesions = torch.zeros(max(1, self.opts.yx_num_lesions), 3, self.opts.yx_lesion_size, self.opts.yx_lesion_size, self.opts.yx_lesion_size).float()
            # radiomics_feat = torch.zeros(max(1, self.opts.yx_num_lesions), 736).float()
            lesions_label = torch.full((max(1, self.opts.yx_num_lesions),), -1).long()
            flag = 0
        return lesions, lesions_label, flag

    def __getitem__(self, index):
        lesions, lesions_label, yx_flag = self._load_yx_data(index)

        assert yx_flag, (self.id_lst[index], self.yx_pid_lst[index])

        return {
            "id": self.id_lst[index],
            "name": self.name_lst[index],


            "lesions": lesions,
            "lesions_label": lesions_label,
            "yx_pid": self.yx_pid_lst[index],
            "yx_flag": yx_flag,

            "liaoxiao": self.liaoxiao_lst[index],
            "os": self.OS_lst[index],
            "os_censor": self.OSCensor_lst[index],


            "fangan": self.fangan_lst[index],
            "label": self.label_lst[index],


            "clinical_sex": ccd_sex(self.sex_lst[index]), # (2,)
            "clinical_age": ccd_age(self.age_lst[index]), # (2,)
            "clinical_buwei": ccd_buwei(self.buwei_lst[index]), # (2,)
            "clinical_leixing": ccd_leixing(self.leixing_lst[index]), # (2,)
            "clinical_time": ccd_time(self.time_lst[index]), # (3,)
            "clinical_tnm_fenqi": ccd_tnm_fenqi(self.tnm_fenqi_lst[index]),
            "clinical_fenqi": ccd_fenqi(self.fenqi_lst[index]),
            "clinical_yx_flag": ccd_yx_flag(self.yx_flag_lst[index]), # (3,)
            "clinical_jibing": ccd_jibing(self.jibing_lst[index]),
            "clinical_xiyan": ccd_xiyan(self.xiyan_lst[index]),
            "clinical_xiyan_num": ccd_xiyan_num(self.xiyan_num_lst[index]),
            "clinical_shoushu": ccd_shoushu(self.shoushu_lst[index]),
            "clinical_zl_per": ccd_zl_per(self.zl_per_lst[index]),

            "fev1": self.fev1_lst[index],
            "fev1_per": self.fev1_per_lst[index],
            "fvc1": self.fvc_lst[index],
            "fvc_per": self.fvc_per_lst[index],
            "dlco": self.dlco_lst[index],
            "dlco_per": self.dlco_per_lst[index],
            "dlcova": self.dlcova_lst[index],
            "dlcova_per": self.dlcova_per_lst[index],

            "hb": self.hb_lst[index],
            "leu": self.leu_lst[index],
            "neu": self.neu_lst[index],
            "pla": self.pla_lst[index],
            "til": self.til_lst[index],
            "tumor": self.tumor_lst[index],

            "clinical_feiyan": ccd_feiyan(self.feiyan_lst[index]),
            "clinical_shiguanyan": ccd_shiguanyan(self.shiguanyan_lst[index])

        }