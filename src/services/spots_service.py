import numpy as np

from numpy.typing import NDArray

from utils.light_spots_utils import get_centroids_from_img

def centroids_shifting(imags:list[NDArray]):
    '''监视点光瞳光轴检测相机质心随时间漂移数据，可用于绘制曲线'''
    pass

def _get_centroids_from_data(txt_data):
    pass

def _get_centroids_from_img(imgs:NDArray):
    pass

def far_field_diameter(imags:list[NDArray], angle:float = 0) -> NDArray:
    '''远场光斑拟合XY直径（数据）随时间变化，可用于绘图
    1. 
    '''
    assert 0<=angle<=360
    pass

def near_field_gauss_fitting(imags:list[NDArray]) -> NDArray:
    '''高斯近场直径拟合计算（数据+绘图）'''
    pass

def flatten_uniformity(imags):
    '''平定光均匀度（数据+绘图）'''
    pass

def wf_rms(wfs):
    pass

def wf_pv(wfs):
    pass

def wf_zernike(wfs, order:int=5):
    '''波前倾斜到像散泽尼克系数变化'''
    pass

def transfer_effecient():
    '''根据功率计算传输效率，以某一次结果为参考98.4%，计算相对值'''
    pass


if __name__ == '__main__':
    pass