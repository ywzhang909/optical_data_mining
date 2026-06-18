# %%
import os
import pandas as pd
from sqlalchemy import create_engine

from dotenv import load_dotenv

if load_dotenv('../.env.prod'):
    url = f'mysql+pymysql://{os.getenv("DB_USER")}:{os.getenv("DB_PASSWORD")}@{os.getenv("DB_HOST")}:{os.getenv("DB_PORT")}/{os.getenv("DB_NAME")}'
else:
    url = 'mysql+pymysql://root:123456@localhost:3306/data_mining'

eng = create_engine(url)
eng.url

# %[markdown]
# 读取实验数据
# %%
exp_data = pd.read_sql_table('experiment', con=eng)
exp_data['date'] = pd.to_datetime(exp_data['experiment_num'], format='%Y%m%d')
exp_data.set_index('experiment_id', inplace=True)

exp_extra_data = exp_data['experiment_extra'].apply(pd.Series)
exp_data = pd.concat([exp_data, exp_extra_data], axis=1)
exp_data.head()
# %%
hel_exp = exp_data[exp_data.experiment_type_id == 2]
exp_params = (
'experiment_polarization_state',
       'experiment_radiation_mode', 'experiment_radiation_duration',
       'experiment_radiation_power', 'experiment_laser_sub_beam',
       'experiment_extra', '横向风', '厚度', '靶材', '子束', '距离', '靶材厚度', '靶材形状', '材质',
       '纵向风', '湍流', '通风'
)

# %[markdown]
# 数据质量分析
# %%
exp_data['exp_id'] = exp_data.index
exp_data['datagroup'] = exp_data['experiment_num'] + exp_data['experiment_group'].apply(str)
exp_data['sub_beam_count'] = exp_data['experiment_laser_sub_beam'].apply(lambda s: s.count('1') if s else 0)
exp_sub_beams = exp_data.groupby(['experiment_laser_sub_beam','子束']).agg(
    {'datagroup': [list,'count'],
     'experiment_laser_start_time' : list,
      'sub_beam_count': 'first'}
)
exp_sub_beams
# %%
hel_exp['experiment_radiation_mode'].value_counts()

# %% _sub_beam 处理子束的情况：如果子束不为空, 则_sub_beam 为 子束的数量；如果子束为空, 则_sub_beam 为空或experiment_laser_sub_beam中1的数量
sub_beam_feat_map = {
    '31路子束': '31路',
    '31路': '31路',
    '31子束': '31路',
    '33路子束': '33路',
    '全子束': '64路',
    '33子束': '33路',
    '高吸收18路': '18路',
}
# 定义处理函数（逐行）
def compute_sub_beam(row):
    sub_beam_val = row["子束"]
    # 判断“子束”是否为空：包括 NaN、None、空字符串或只含空白
    if pd.isna(sub_beam_val) or (isinstance(sub_beam_val, str) and sub_beam_val.strip() == ""):
        # 从 experiment_laser_sub_beam 列取值并统计 '1' 的数量
        beam_str = row["experiment_laser_sub_beam"]
        if pd.isna(beam_str):
            return pd.NaT
        else:
            return f"{(beam_str).count('1')}路"
    else:
        # 非空：使用 param_dict 映射，若不在 dict 中可返回 None 或保留原值等
        return sub_beam_feat_map.get(sub_beam_val, pd.NaT)  # 未映射时设为 NaN

# 应用函数创建新列
hel_exp["_sub_beam"] = hel_exp.apply(compute_sub_beam, axis=1)
# hel_exp[hel_exp['_sub_beam'].isna()].to_csv(
#     f'子束为空.csv',
#     index=True,
#     encoding='gbk'
# )
# %% 找出exp_params全部为空的行: 0
exp_params = [
'experiment_polarization_state',
       'experiment_radiation_mode', 'experiment_radiation_duration',
       'experiment_radiation_power',
       '横向风', '厚度', '靶材', '距离', '靶材厚度', '靶材形状', '材质',
       '纵向风', '湍流', '通风', '_sub_beam'
]
#
hel_exp[exp_params].isna().all(axis=1).value_counts()
# %% 找出出光属性任一为空的行：0
laser_params = [
    'experiment_polarization_state',
    'experiment_radiation_mode', 'experiment_radiation_duration',
    'experiment_radiation_power', '_sub_beam'
]
# laser_params_name = {
#     'experiment_radiation_mode': '光强分布',
#     'experiment_radiation_duration': '出光时长',
#     'experiment_radiation_power': '功率',
#     '_sub_beam': '统计子束'
# }
hel_exp[hel_exp[laser_params].isna().any(axis=1)].to_csv(
    f'出光属性（{"、".join(laser_params)}为空）.csv',
    index=True,
    encoding='gbk'
)

# %% 环境参数
env_params = ['横向风', '纵向风', '湍流', '通风']
hel_exp[env_params].describe()
# %%
'''
靶材
不锈钢    50
铝合金     7
铝       1
'''
# hel_exp['靶材'].value_counts()
hel_exp['靶材'] = hel_exp['靶材'].map(
    {'不锈钢': '不锈钢', '铝合金': '铝合金', '铝': '铝合金'})
# %%
'''
距离
1km      50
1.5km    11
2.4km     5
'''
hel_exp['距离'].value_counts()

# %%
'''
靶材厚度
2mm    1
'''
hel_exp['靶材厚度'].value_counts()

# %%

# hel_exp['材质'].value_counts()
hel_exp['靶材'] = hel_exp.apply(lambda x: x['材质'] if \
                (pd.isna(x['靶材']) or (x['靶材'] == '纯铝')) else x['靶材'], axis=1)
# %%
target_params = [
       '靶材',
    #    '距离',
    #    '靶材厚度'
]

# %% 找出'靶材', '距离'任一为空的行
hel_exp[hel_exp[target_params].isna().any(axis=1)].to_csv(
    f'靶材信息（{"、".join(target_params)}为空）.csv',
    index=True,
    encoding='gbk'
)

# %%
ana_params = target_params + laser_params

hel_exp = hel_exp.dropna(how='any',subset=ana_params)

# %[markdown]
# 数据聚合
# %% 按'靶材', '距离', '靶材厚度', '材质'分组
hel_exp.groupby(target_params).count()
# hel_exp.groupby('靶材').count()
# %%
hel_exp.groupby(laser_params).agg(
    {'靶材': set, '距离': set, '靶材厚度': set}
)
# %%
# TODO distinct count
hel_exp.groupby(ana_params).groups

# %%[markdown]
# 找出所有数字光学的数据

