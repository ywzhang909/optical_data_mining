from pathlib import Path

import pandas as pd

def get_date_dir(root_dir:Path):
    '''获取根目录下所有日期目录'''
    # 扫描data文件夹下的日期

    date_dirs = list(root_dir.glob('20'+'[0-9]'*6))
    date_df = pd.DataFrame(date_dirs, columns=['date_dir'])
    date_df['date'] = date_df['date_dir'].apply(lambda x: x.name)
    date_df['date'] = pd.to_datetime(date_df['date'])
    date_df.set_index('date', inplace=True)

    date_df['tasks_dir'] = date_df['date_dir'].apply(lambda x: list(x.glob('20'+'[0-9]'*9)))
    date_df = date_df.explode('tasks_dir')
    date_df['tasks'] = date_df['tasks_dir'].apply(lambda x: x.name)
    date_df['task_id'] = date_df['tasks'].apply(lambda x: x[8:])
    
    return date_df

def get_experiments(experiement:Path):
    experiments_df = date_df
    experiments_df['experiments_target_dir'] = experiments_df['tasks_dir'].apply(lambda x: list(x.glob('target/*')))
    experiments_df = experiments_df[experiments_df['experiments_target_dir'].apply(len) > 0]
    experiments_df = experiments_df.explode('experiments_target_dir')
    experiments_df['experiment_name'] = experiments_df['experiments_target_dir'].apply(lambda x: str(x.name))



if __name__ == '__main__':
    date_dir = get_date_dir(Path('data'))

    print(date_dir)