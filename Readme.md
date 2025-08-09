- 目前先用GTV数据作为靶区数据
- PFS为无进展生存期，实验集无，予以删除
- cohort为MuMo论文中分出的不同类型的患者，此处我们设定为None
- _generate_mask_bags_label中掩码改为三维数组形式


## question：
1. start_time和yx_flag怎么定义
2. _generate_mask_bags_label和_generate_mask_words_label的理解和修改
3. ccd_sex，ccd_xiyan_num这种数量指标是否需要二值化

## 符号说明：
- id：放疗号/病案号
- name：患者姓名
- yx_pid：放疗号
- start_time：第一次定位时间【原模型为开始治疗日期】
- liaoxiao：目前无，予以保留，后续可以用两次定位CT肿瘤体积比填充
- OS：生存期
- OSCensor是否存活，实验集并无此数据，保留，可用末次随访时间-第一次定位CT时间代替
- fangan：化疗方案（含吉西他滨or含紫杉醇or均不含）
- sex：性别
- age：年龄
- jibing：肺部原有疾病
- leixing：治疗类型（放疗or序贯放化疗or同步放化疗）
- xiyan：是否吸烟
- xiyan_num：吸烟指数
- shoushu：是否手术
- zl_per：是否联合免疫治疗
- tnm_fenqi：T/N/M分期
- fenqi：总分期
- 通气残气弥散检查指标【_per为百分比】：fev1, fev1_per, fvc, fvc_per, dlco, dlco_per, dlcova, dlcova_per
- 【放疗期间最低
- hb：血红蛋白
- leu：白细胞
- neu：中性粒细胞
- pla：血小板
- til：淋巴细胞】
- feiyan：是否患放射性肺炎
- shiguanyan：是否患放射性食管炎
- tumor：肿瘤占比，目前无，但可以获得
- buwei：转移部位
- yx_flag：第二次定位时间【原模型为影像采样时间】
- bingli：病理类型
