"""
默认使用的是YXModelOnly类，通过继承，YXModelOnly自动获得YXModel的所有函数
1.55行：目前是对病灶进行4分类预测：utils中的'鳞癌': 0,'腺癌': 1,'SCLC': 2,'NSCLC': 3。
2.57-64行/155-163行：临床特征的处理（已适配到我们的数据集上）
3.第一版测试时关注：如果没有组学特征的数据集，使用YXModelOnly代码是否会报错。
"""
import torch
import torch.nn as nn
from torch.nn import init
from model.nn_layers.ffn import FFN
from model.nn_layers.attn_layers import *
from typing import NamedTuple, Optional
from torch import Tensor
from torchvision import models
import torch.nn.functional as F
from copy import deepcopy
from model.feature_extractors.mnasnet import MNASNet
from model.nn_layers.transformer import *

class YXModel(nn.Module):#基础影像模型（含临床特征融合）
    def __init__(self, opts, *args, **kwargs):
        super().__init__()
        self.opts = opts
        """
        24-38行均属于CNN骨干网络的构建。由于没有找到'checkpoints/mnasnet_s_1.0_imagenet_224x224.pth'这个预训练权重，因此用默认的resnet18.直接运行33和38行
        """
        if opts.yx_cnn_name == "mnasnet":
            s = 1.0
            weight = 'checkpoints/mnasnet_s_1.0_imagenet_224x224.pth'
            backbone = MNASNet(alpha=s)
            pretrained_dict = torch.load(weight, map_location=torch.device('cpu'))
            backbone.load_state_dict(pretrained_dict)
            del backbone.classifier
            self.cnn = backbone
        else:
            backbone = eval(f"models.{opts.yx_cnn_name}")(pretrained=opts.yx_cnn_pretrained)#默认为resnet18
            # if opts.yx_cnn_name == "mnasnet1_0":
            if "mnasnet" in opts.yx_cnn_name:
                self.cnn = nn.Sequential(*[*list(backbone.children())[:-1], nn.AdaptiveAvgPool2d(1)])
            else:
                self.cnn = nn.Sequential(*list(backbone.children())[:-1])#移除分类层，创建特征提取器。
        #空间注意力层。学习空间注意力权重，突出重要区域。
        self.attn_layer = nn.Conv2d(opts.yx_cnn_features+3, 1, kernel_size=1, padding=0, stride=1)
        #特征投影层。[B,N_1,512]-->[B,N_1,128]
        self.cnn_project = nn.Linear(opts.yx_cnn_features, opts.yx_out_features)
        #病灶间的注意力机制。学习不同病灶之间的关联性。
        self.attn_over_lesions = nn.MultiheadAttention(embed_dim=opts.yx_out_features,
                                num_heads=1, dropout=opts.yx_dropout, batch_first=True)
        #对聚合后的特征进行非线性变换
        self.ffn_attn_l2p = FFN(input_dim=opts.yx_out_features, scale=2, p=opts.yx_dropout)
        #计算每个病灶的重要性权重
        self.lesions_weight_fn = nn.Linear(opts.yx_out_features, 1, bias=False)
        self.attn_dropout = nn.Dropout(p=opts.yx_attn_dropout)#防止过拟合
        self.attn_fn = nn.Softmax(dim=-1)#将权重归一化为概率分布

        self.lesions_classifier = nn.Linear(opts.yx_out_features, 4)#对每个病灶进行4分类预测

        # 临床特征融入 - 适配肺癌数据集
        self.sex_fc = nn.Linear(2, opts.yx_out_features)           # 性别 (2,)
        self.age_fc = nn.Linear(2, opts.yx_out_features)           # 年龄 (2,)
        self.buwei_fc = nn.Linear(2, opts.yx_out_features)         # 肿瘤部位 (2,)
        self.leixing_fc = nn.Linear(3, opts.yx_out_features)       # 治疗类型 (3,)
        self.time_fc = nn.Linear(3, opts.yx_out_features)          # 治疗时间 (3,)
        self.fenqi_fc = nn.Linear(7, opts.yx_out_features)         # 总分期 (7,) 
        self.jibing_fc = nn.Linear(3, opts.yx_out_features)        # 合并疾病 (3,)
        self.xiyan_fc = nn.Linear(2, opts.yx_out_features)         # 吸烟史 (2,)
        self.xiyan_num_fc = nn.Linear(2, opts.yx_out_features)     # 吸烟量 (2,)
        self.shoushu_fc = nn.Linear(2, opts.yx_out_features)       # 手术史 (2,)
        self.zl_per_fc = nn.Linear(2, opts.yx_out_features)        # 免疫治疗 (2,)
        #临床-影像注意力：临床特征作为key/value，影像特征作为query
        self.clinical_image_attn = nn.MultiheadAttention(embed_dim=opts.yx_out_features, num_heads=1, dropout=opts.yx_dropout, batch_first=True)

        # 组学特征融入X 不用管，用不到组学特征。
        self.radiomics_fc = nn.Linear(736, opts.yx_out_features)
        self.radiomics_image_attn = nn.MultiheadAttention(embed_dim=opts.yx_out_features, num_heads=1, dropout=opts.yx_dropout, batch_first=True)

        #self.yx_classifier = nn.Linear(opts.yx_out_features, opts.n_classes)

    def energy_function(self, x, weight_fn, need_attn=False):
        # x: (B, N, C)
        x = weight_fn(x).squeeze(dim=-1) # (B, N)
        energy: Tensor[Optional] = None
        if need_attn:
            energy = x
        x = self.attn_fn(x)
        x = self.attn_dropout(x)
        x = x.unsqueeze(dim=-1) # (B, N, 1)
        return x, energy
    #并行融合
    def parallel_radiomics_clinical(self, patient_from_lesions, radiomics_feat, clinical_feat):
        radiomics_image_feat, radiomics_attnmap = self.radiomics_image_attn(key=radiomics_feat,
                        query=patient_from_lesions.unsqueeze(dim=1), value=radiomics_feat)
        clinical_image_feat, clinical_attnmap = self.clinical_image_attn(key=clinical_feat,
                        query=patient_from_lesions.unsqueeze(dim=1), value=clinical_feat)
        patient_from_lesions = patient_from_lesions \
                                + clinical_image_feat.squeeze(dim=1) \
                                + radiomics_image_feat.squeeze(dim=1)

        self.info_dict["clinical_weight"] = clinical_attnmap[0, 0].detach().cpu().numpy()
        return patient_from_lesions
    #串行融合：影像 → 影像+组学 → 影像+组学+临床（先融合组学特征，再融合临床特征）
    def series_radiomics_clinical(self, patient_from_lesions, radiomics_feat, clinical_feat):
        radiomics_image_feat, _ = self.radiomics_image_attn(key=radiomics_feat,
                        query=patient_from_lesions.unsqueeze(dim=1), value=radiomics_feat)
        patient_from_lesions = patient_from_lesions + radiomics_image_feat.squeeze(dim=1) 
        clinical_image_feat, _ = self.clinical_image_attn(key=clinical_feat,
                        query=patient_from_lesions.unsqueeze(dim=1), value=clinical_feat)
        patient_from_lesions = patient_from_lesions + clinical_image_feat.squeeze(dim=1)
        return patient_from_lesions
    #串行融合：影像 → 影像+临床 → 影像+临床+组学（先融合临床特征，再融合组学特征）
    def series_clinical_radiomics(self, patient_from_lesions, radiomics_feat, clinical_feat):
        clinical_image_feat, _ = self.clinical_image_attn(key=clinical_feat,
                        query=patient_from_lesions.unsqueeze(dim=1), value=clinical_feat)
        patient_from_lesions = patient_from_lesions + clinical_image_feat.squeeze(dim=1)
        radiomics_image_feat, _ = self.radiomics_image_attn(key=radiomics_feat,
                        query=patient_from_lesions.unsqueeze(dim=1), value=radiomics_feat)
        patient_from_lesions = patient_from_lesions + radiomics_image_feat.squeeze(dim=1) 
        return patient_from_lesions

    def forward(self, batch, *args, **kwargs):
        lesions = batch["lesions"] # (B, N_l, 3, H, W)
        yx_flag = batch["clinical_yx_flag"] # (B, 3)

        B, N_l, C, H, W = lesions.shape
        # (B, N_l, 3, H, W) --> (B, N_l, C)
        lesions_cnn = self.cnn(lesions.view(B*N_l, C, H, W))
        yx_flag = yx_flag.view(B, 1, 3, 1, 1).repeat(1, N_l, 1, *lesions_cnn.shape[-2:])
        yx_flag = yx_flag.view(B*N_l, 3, *lesions_cnn.shape[-2:])
        attn_lesions_cnn = torch.cat([lesions_cnn, yx_flag], dim=1)
        attn_lesions_cnn = torch.sigmoid(self.attn_layer(attn_lesions_cnn))
        lesions_cnn = lesions_cnn * attn_lesions_cnn
        lesions_cnn = lesions_cnn.view(B, N_l, -1)
        lesions_cnn = self.cnn_project(lesions_cnn)

        self.info_dict = {
            "id": batch["id"][0],
            "lesions_label": batch["lesions_label"][0].detach().cpu().numpy(),
        }

        lesions_attn, lesions_attnmap = self.attn_over_lesions(key=lesions_cnn, query=lesions_cnn, value=lesions_cnn)
        self.info_dict["lesions_attnmap"] = lesions_attnmap[0].detach().cpu().numpy()
        lesions_attn_energy, lesions_attn_energy_unnorm = self.energy_function(lesions_attn, self.lesions_weight_fn)
        # (B, N_l, C) x (B, N_l, 1) --> (B, C)
        self.info_dict["lesions_weight"] = lesions_attn_energy[0, ..., 0].detach().cpu().numpy()
        patient_from_lesions = torch.sum(lesions_attn * lesions_attn_energy, dim=1)
        patient_from_lesions = self.ffn_attn_l2p(patient_from_lesions)

        lesions_pred = self.lesions_classifier(lesions_attn)

        self.lesions_attn_energy_unnorm = lesions_attn_energy_unnorm
        """
        将11个临床特征分别通过全连接层映射到统一维度，然后堆叠成(B, 11, 128)的张量
        1.性别
        2.年龄 
        3.肿瘤部位
        4.治疗类型
        5.治疗时间
        6.总分期
        7.合并疾病
        8.吸烟史
        9.吸烟量
        10.手术史
        11.免疫治疗
        """
        clinical_feat = torch.stack([
            self.sex_fc(batch["clinical_sex"]),
            self.age_fc(batch["clinical_age"]),
            self.buwei_fc(batch["clinical_buwei"]),
            self.leixing_fc(batch["clinical_leixing"]),
            self.time_fc(batch["clinical_time"]),
            self.fenqi_fc(batch["clinical_fenqi"]),
            self.jibing_fc(batch["clinical_jibing"]),
            self.xiyan_fc(batch["clinical_xiyan"]),
            self.xiyan_num_fc(batch["clinical_xiyan_num"]),
            self.shoushu_fc(batch["clinical_shoushu"]),
            self.zl_per_fc(batch["clinical_zl_per"]),
        ], dim=1) # (B, 11, C)
        
        radiomics_feat = self.radiomics_fc(batch["yx_radiomics_feat"]) # (B, N_l, 736) --> (B, N_l, C)
        
        if self.opts.feat_fusion_mode == "parallel":
            patient_from_lesions = self.parallel_radiomics_clinical(patient_from_lesions, radiomics_feat, clinical_feat)
        elif self.opts.feat_fusion_mode == "series_rc":
            patient_from_lesions = self.series_radiomics_clinical(patient_from_lesions, radiomics_feat, clinical_feat)
        elif self.opts.feat_fusion_mode == "series_cr":
            patient_from_lesions = self.series_clinical_radiomics(patient_from_lesions, radiomics_feat, clinical_feat)
        else:
            raise NotImplementedError

        if self.opts.attnmap_weight_dir not in [None, "None"]:
            npz_dir = os.path.join(self.opts.attnmap_weight_dir, "YX")
            os.makedirs(npz_dir, exist_ok=True)
            if batch["yx_flag"][0].item():
                np.savez(os.path.join(npz_dir, self.info_dict["id"]+".npz"),
                        lesions_label=self.info_dict["lesions_label"],
                        lesions_attnmap=self.info_dict["lesions_attnmap"],
                        lesions_weight=self.info_dict["lesions_weight"],
                        clinical_weight=self.info_dict["clinical_weight"])
        
        return {
            "feat": patient_from_lesions,
            "lesions_pred": lesions_pred,
            #"yx_pred": self.yx_classifier(patient_from_lesions)
        }

class YXModelOnly(YXModel):#通过继承
    def __init__(self, opts, *args, **kwargs):
        super().__init__(opts, *args, **kwargs)
        self.classifier = nn.Linear(opts.yx_out_features, opts.n_classes)

    def forward(self, batch, *args, **kwargs):
        yx_results = super().forward(batch, *args, **kwargs)
        yx_pred = self.classifier(yx_results["feat"])
        return {
            "pred": yx_pred,
            "feat": yx_results["feat"],
            "lesions_pred": yx_results["lesions_pred"],
        }

class YXModelNoneClinicalRadiomics(YXModel):
    def forward(self, batch, *args, **kwargs):
        lesions = batch["lesions"] # (B, N_l, 3, H, W)
        yx_flag = batch["clinical_yx_flag"] # (B, 3)

        B, N_l, C, H, W = lesions.shape
        # (B, N_l, 3, H, W) --> (B, N_l, C)
        lesions_cnn = self.cnn(lesions.view(B*N_l, C, H, W))
        yx_flag = yx_flag.view(B, 1, 3, 1, 1).repeat(1, N_l, 1, *lesions_cnn.shape[-2:])
        yx_flag = yx_flag.view(B*N_l, 3, *lesions_cnn.shape[-2:])
        attn_lesions_cnn = torch.cat([lesions_cnn, yx_flag], dim=1)
        attn_lesions_cnn = torch.sigmoid(self.attn_layer(attn_lesions_cnn))
        lesions_cnn = lesions_cnn * attn_lesions_cnn
        lesions_cnn = lesions_cnn.view(B, N_l, -1)
        lesions_cnn = self.cnn_project(lesions_cnn)

        lesions_attn, lesions_attnmap = self.attn_over_lesions(key=lesions_cnn, query=lesions_cnn, value=lesions_cnn)
        lesions_attn_energy, lesions_attn_energy_unnorm = self.energy_function(lesions_attn, self.lesions_weight_fn)
        # (B, N_l, C) x (B, N_l, 1) --> (B, C)
        patient_from_lesions = torch.sum(lesions_attn * lesions_attn_energy, dim=1)
        patient_from_lesions = self.ffn_attn_l2p(patient_from_lesions)

        lesions_pred = self.lesions_classifier(lesions_attn)

        self.lesions_attn_energy_unnorm = lesions_attn_energy_unnorm
        
        return {
            "feat": patient_from_lesions,
            "lesions_pred": lesions_pred,
            #"yx_pred": self.yx_classifier(patient_from_lesions)
        }