"""
RabbitMQ光斑特征提取服务

从RabbitMQ获取光斑类型（光瞳、光轴）和图片路径，然后分别计算对应的特征
"""
import os
import sys
import json
import pika
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append('./src')

import dotenv
from data_mining.image.common import get_profiles, read_tiff_to_numpy

dotenv.load_dotenv()

# RabbitMQ 配置
RABBITMQ_HOST = os.environ.get('RABBITMQ_HOST', 'localhost')
RABBITMQ_PORT = int(os.environ.get('RABBITMQ_PORT', 5672))
RABBITMQ_USER = os.environ.get('RABBITMQ_USER', 'guest')
RABBITMQ_PASSWORD = os.environ.get('RABBITMQ_PASSWORD', 'guest')
RABBITMQ_QUEUE = os.environ.get('RABBITMQ_QUEUE', 'spot_features')

# 图像配置
EXP_DIR = os.environ.get('EXP_DIR', 'X:/')

# 特征计算相关导入
from scipy.ndimage import center_of_mass
from scipy.optimize import curve_fit
from cv2 import findContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE
from cv2 import THRESH_BINARY, threshold, fitEllipse, contourArea
from cv2 import drawContours, FILLED, meanStdDev
import cv2


def convert_to_cv(float_image) -> np.ndarray:
    """将图像转换为OpenCV兼容的uint8格式"""
    assert isinstance(float_image, np.ndarray)
    image_min = np.min(float_image)
    image_max = np.max(float_image)
    normalized_image = (float_image - image_min) / (image_max - image_min) * 255
    return normalized_image.astype(np.uint8)


def d4sigma(img: np.ndarray, pixel_size_um=1.0):
    """计算图像的 D4σ 直径"""
    total = img.sum()
    cy, cx = center_of_mass(img)
    h, w = img.shape
    y, x = np.mgrid[0:h, 0:w]
    
    mu_xx = np.sum((x - cx)**2 * img) / total
    mu_yy = np.sum((y - cy)**2 * img) / total
    Dx = 4 * np.sqrt(max(mu_xx, 0)) * pixel_size_um
    Dy = 4 * np.sqrt(max(mu_yy, 0)) * pixel_size_um
    
    return {
        'center_x': float(cx),
        'center_y': float(cy),
        'D_x': float(Dx),
        'D_y': float(Dy),
        'avg_sigma2': float(np.sqrt(Dx * Dy)),
    }


def ellipse_fit(uint8_image):
    """椭圆拟合特征提取"""
    noise_threshhold = np.max(uint8_image) * 0.3
    binary_image = threshold(uint8_image, noise_threshhold, 255, THRESH_BINARY)[1]
    
    try:
        contours, _ = findContours(binary_image, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
        assert contours, "No contours found"
        largest_contour = max(contours, key=contourArea)
        (ellipse_center_x, ellipse_center_y), (short_axis, long_axis), angle = fitEllipse(largest_contour)
    except (AssertionError, ValueError):
        return {
            'ellipse_center_x': np.nan,
            'ellipse_center_y': np.nan,
            'short_axis': np.nan,
            'long_axis': np.nan,
            'ellipticity': np.nan,
            'angle': np.nan,
            'uniformity': np.nan
        }
    
    area = contourArea(largest_contour)
    mask = np.zeros_like(uint8_image, dtype=np.uint8)
    if area > 100:
        drawContours(mask, [largest_contour], -1, (255,), thickness=FILLED)
        mean_val, std_val = meanStdDev(uint8_image, mask=mask)
        uniformity = std_val[0][0] / mean_val[0][0]
    else:
        uniformity = np.nan
    
    return {
        'ellipse_center_x': ellipse_center_x,
        'ellipse_center_y': ellipse_center_y,
        'short_axis': short_axis,
        'long_axis': long_axis,
        'ellipticity': long_axis / short_axis,
        'angle': angle,
        'uniformity': uniformity
    }


def find_spot_border(image):
    """寻找光斑边界"""
    denoised_image = cv2.GaussianBlur(image, (3, 3), 0)
    noise_threshold = np.max(denoised_image) / np.e
    denoised_image = np.where(denoised_image > noise_threshold, denoised_image, 0)
    
    if denoised_image.dtype != np.uint8:
        denoised_image = denoised_image.astype(np.uint8)
    
    denoised_image = cv2.fastNlMeansDenoising(denoised_image, None, 10, 7, 21)
    near_binary_img = threshold(denoised_image, noise_threshold, 255, THRESH_BINARY)[1]
    
    contours, _ = findContours(near_binary_img, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    else:
        x, y = np.nan, np.nan
        radius = np.nan
    
    return {
        'border_x': x,
        'border_y': y,
        'border_radius': radius
    }


def calculate_sharpness(img: np.ndarray):
    """计算锐度"""
    gradient_x = np.gradient(img, axis=1)
    gradient_y = np.gradient(img, axis=0)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    return float(np.mean(gradient_magnitude))


def calc_lr_diff(profile):
    """计算左右差异特征"""
    init_mu = np.median(profile)
    indices = np.where(profile > init_mu)[0]
    if len(indices) == 0:
        return {
            'flat_fit_diff': np.nan,
            'flat_fit_ndiff': np.nan,
            'flat_fit_diameter': np.nan,
        }
    
    p0 = [indices[0], indices[-1], np.max(profile), np.max(profile), init_mu]
    bounds = ([0, 0, 0, 0, 0], [len(profile), len(profile), np.max(profile), np.max(profile), np.max(profile)])
    
    try:
        params, _ = curve_fit(
            lambda x, l, r, la, ra, mu: np.where(
                (x >= l) & (x <= r),
                la + (ra - la) / (r - l) * (x - l) if r != l else la,
                mu
            ),
            np.arange(len(profile)), profile, p0=p0, bounds=bounds
        )
        l, r, la, ra, mu = params
        return {
            'flat_fit_diff': float(np.abs(ra - la)),
            'flat_fit_ndiff': float(np.abs(ra - la) / np.abs(r - l)),
            'flat_fit_diameter': float(r - l)
        }
    except:
        return {
            'flat_fit_diff': np.nan,
            'flat_fit_ndiff': np.nan,
            'flat_fit_diameter': np.nan,
        }


def extract_pupil_features(img_array: np.ndarray):
    """
    提取光瞳特征
    
    Args:
        img_array: 图像数组
    
    Returns:
        dict: 光瞳特征
    """
    # 去暗场
    black_threshold = np.median(img_array)
    denoise_img = np.where(img_array > black_threshold, img_array - black_threshold, 0)
    intensity = float(np.sum(denoise_img))
    
    # D4σ 直径
    d4sigma_features = d4sigma(denoise_img)
    
    # 椭圆拟合
    uint8_img = convert_to_cv(denoise_img)
    ellipse_features = ellipse_fit(uint8_img)
    
    # 边界检测
    border_features = find_spot_border(uint8_img)
    
    # 锐度
    sharpness = calculate_sharpness(denoise_img)
    
    # 投影特征
    center_x = d4sigma_features['center_x']
    center_y = d4sigma_features['center_y']
    
    try:
        profiles = get_profiles(denoise_img, (center_x, center_y))
        profile_features = calc_lr_diff(profiles['vertical'])
    except Exception:
        profile_features = {
            'flat_fit_diff': np.nan,
            'flat_fit_ndiff': np.nan,
            'flat_fit_diameter': np.nan,
        }
    
    return {
        'spot_type': 'pupil',
        'intensity': intensity,
        **d4sigma_features,
        **ellipse_features,
        **border_features,
        'sharpness': sharpness,
        **profile_features
    }


def extract_axis_features(img_array: np.ndarray):
    """
    提取光轴（焦斑）特征
    
    Args:
        img_array: 图像数组
    
    Returns:
        dict: 光轴特征
    """
    # 去暗场
    black_threshold = np.median(img_array)
    denoise_img = np.where(img_array > black_threshold, img_array - black_threshold, 0)
    intensity = float(np.sum(denoise_img))
    
    # D4σ 直径
    d4sigma_features = d4sigma(denoise_img)
    
    # 椭圆拟合
    uint8_img = convert_to_cv(denoise_img)
    ellipse_features = ellipse_fit(uint8_img)
    
    # 边界检测
    border_features = find_spot_border(uint8_img)
    
    return {
        'spot_type': 'axis',
        'intensity': intensity,
        **d4sigma_features,
        **ellipse_features,
        **border_features
    }


def extract_features(spot_type: str, image_path: str) -> dict:
    """
    根据光斑类型提取对应特征
    
    Args:
        spot_type: 光斑类型 ('pupil' 光瞳 或 'axis' 光轴)
        image_path: 图片路径
    
    Returns:
        dict: 特征字典
    """
    # 读取图像
    img_path = Path(image_path)
    if not img_path.is_absolute():
        img_path = Path(EXP_DIR) / image_path
    
    if not img_path.exists():
        raise FileNotFoundError(f"图像文件不存在: {img_path}")
    
    img_array = read_tiff_to_numpy(img_path)
    
    # 根据类型提取特征
    if spot_type.lower() in ['pupil', '光瞳']:
        return extract_pupil_features(img_array)
    elif spot_type.lower() in ['axis', '光轴']:
        return extract_axis_features(img_array)
    else:
        raise ValueError(f"未知的光斑类型: {spot_type}")


def process_message(ch, method, properties, body):
    """
    处理RabbitMQ消息
    
    消息格式:
    {
        "spot_type": "pupil" | "axis",
        "image_path": "/path/to/image.tiff"
    }
    """
    try:
        message = json.loads(body)
        spot_type = message['spot_type']
        image_path = message['image_path']
        
        print(f"处理消息: spot_type={spot_type}, image_path={image_path}")
        
        # 提取特征
        features = extract_features(spot_type, image_path)
        features['image_path'] = image_path
        
        print(f"提取的特征: {features}")
        
        # TODO: 可以选择将结果发送到另一个队列或保存到数据库
        # result_queue = os.environ.get('RABBITMQ_RESULT_QUEUE', 'spot_features_result')
        # ch.basic_publish(exchange='', routing_key=result_queue, body=json.dumps(features))
        
        # 确认消息
        ch.basic_ack(delivery_tag=method.delivery_tag)
        
    except Exception as e:
        print(f"处理消息失败: {e}")
        # 拒绝消息，不重新入队
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def run_consumer():
    """运行RabbitMQ消费者"""
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASSWORD)
    parameters = pika.ConnectionParameters(
        host=RABBITMQ_HOST,
        port=RABBITMQ_PORT,
        credentials=credentials
    )
    
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    
    # 声明队列
    channel.queue_declare(queue=RABBITMQ_QUEUE, durable=True)
    
    # 设置QoS，一次只处理一条消息
    channel.basic_qos(prefetch_count=1)
    
    # 开始消费
    channel.basic_consume(
        queue=RABBITMQ_QUEUE,
        on_message_callback=process_message
    )
    
    print(f"开始监听队列: {RABBITMQ_QUEUE}")
    channel.start_consuming()


def publish_test_message(spot_type: str, image_path: str):
    """
    发布测试消息到队列
    
    Args:
        spot_type: 光斑类型 ('pupil' 或 'axis')
        image_path: 图片路径
    """
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASSWORD)
    parameters = pika.ConnectionParameters(
        host=RABBITMQ_HOST,
        port=RABBITMQ_PORT,
        credentials=credentials
    )
    
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    
    # 声明队列
    channel.queue_declare(queue=RABBITMQ_QUEUE, durable=True)
    
    message = {
        'spot_type': spot_type,
        'image_path': image_path
    }
    
    channel.basic_publish(
        exchange='',
        routing_key=RABBITMQ_QUEUE,
        body=json.dumps(message),
        properties=pika.BasicProperties(
            delivery_mode=2,  # 持久化
        )
    )
    
    print(f"消息已发送: {message}")
    connection.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='RabbitMQ光斑特征提取服务')
    parser.add_argument('--mode', choices=['consumer', 'test'], default='consumer',
                        help='运行模式: consumer-消费消息, test-发送测试消息')
    parser.add_argument('--spot-type', choices=['pupil', 'axis', '光瞳', '光轴'],
                        help='光斑类型 (测试模式使用)')
    parser.add_argument('--image-path', help='图片路径 (测试模式使用)')
    
    args = parser.parse_args()
    
    if args.mode == 'consumer':
        run_consumer()
    elif args.mode == 'test':
        if not args.spot_type or not args.image_path:
            parser.error('--spot-type 和 --image-path 在测试模式下是必需的')
        publish_test_message(args.spot_type, args.image_path)
