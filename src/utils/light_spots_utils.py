from PIL import Image

import math
import numpy as np

from scipy.optimize import curve_fit
import cv2

def read_tiff_to_numpy(file_path):
    """
    Read a TIFF image file and convert it to a NumPy array.

    Args:
        file_path (str): The path to the TIFF image file.

    Returns:
        np.ndarray: A NumPy array representing the TIFF image.
    """
    try:
        # Open the TIFF image using PIL
        image = Image.open(file_path)
        if image.mode != 'L':
            image = image.convert('L')
        # Convert the image to a NumPy array
        image_array = np.array(image)
        return image_array
    except Exception as e:
        print(f"Error reading the TIFF file: {e}")
        return None

def find_centroid(img):
    """
    Calculate the centroid coordinates of an object in a binary image.

    Args:
        image (np.ndarray): A binary image represented as a NumPy array.

    Returns:
        tuple: A tuple containing the (y, x) coordinates of the centroid.
    """
    # Calculate centroid
    total = np.sum(img)
    if total == 0:
        raise ValueError("Empty image - all pixel values are zero")
    # Get image dimensions
    height, width = img.shape
    # Create coordinate grids
    x, y = np.indices((width, height))
    # Calculate weighted coordinates
    x_center = int(np.sum(x * img.T) / total)
    y_center = int(np.sum(y * img.T) / total)
    
    return x_center, y_center


def find_lightest_centroid(img):
    """
    Calculate the centroid coordinates of the lightest object in a binary image.

    Args:
        image (np.ndarray): A binary image represented as a NumPy array.

    Returns:
        tuple: A tuple containing the (y, x) coordinates of the centroid.
    """
    # Calculate centroid
    total = np.sum(img)
    if total == 0:
        raise ValueError("Empty image - all pixel values are zero")
    _img = img.copy()
    max_intensity = np.max(_img)
    _img[img!=max_intensity] = 0

    return find_centroid(_img)

def cartesian_to_polar(x, y, c_x, c_y):
    """
    Convert Cartesian coordinates to polar coordinates with a custom origin.

    Args:
        x (np.ndarray): x-coordinates of the points.
        y (np.ndarray): y-coordinates of the points.
        c_x (float): x-coordinate of the custom origin.
        c_y (float): y-coordinate of the custom origin.

    Returns:
        tuple: A tuple containing the radial distances (r) and angles (theta).
    """
    dx = x - c_x
    dy = y - c_y
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    return r, theta

def polar_to_cartesian(r, theta, c_x, c_y):
    """
    Convert polar coordinates to Cartesian coordinates with a custom origin.

    Args:
        r (np.ndarray): Radial distances from the custom origin.
        theta (np.ndarray): Angles in radians.
        c_x (float): x-coordinate of the custom origin.
        c_y (float): y-coordinate of the custom origin.

    Returns:
        tuple: A tuple containing the x-coordinates (x) and y-coordinates (y).
    """
    x = c_x + r * np.cos(theta)
    y = c_y + r * np.sin(theta)
    return x, y

def normalize_data(data):
    """
    Normalize the data to the range [0, 1].
    Args:
        data (np.ndarray): The input data array.
    Returns:
        np.ndarray: The normalized data array.
    """
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return data
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

def gaussian(x, mu, sigma):
    """
    Define the Gaussian function.

    Args:
        x (np.ndarray): Input x values.
        A (float): Amplitude of the Gaussian.
        mu (float): Mean of the Gaussian.
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        np.ndarray: Output values of the Gaussian function.
    """
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def fitting_gaussian(data):
    """
    Fit a Gaussian function to a given data series.
    Args:
        data (np.ndarray): The data series to fit a Gaussian function.
    Returns:
        tuple: A tuple containing the fitted parameters (A, b, mu, sigma) and the fitted curve.
    """
    x_data = np.arange(len(data))
    initial_guess = [np.argmax(data), 10]
    popt, covariance = curve_fit(gaussian, x_data, data, p0=initial_guess)

    return popt, covariance


def calculate_diameter(popt):
    """
    Calculate the diameter at y = 1/e + b for a given data series.

    Args:
        data (np.ndarray): The data series to fit a Gaussian function.

    Returns:
        float: The calculated diameter.
    """
    mu, sigma = popt
    diameter = 2 * math.sqrt(2 * sigma**2)
    return diameter

def calculate_xy_diameters(image, centroid):
    """
    Calculate the diameters at y = 1/e + b in x and y directions.

    Args:
        image (np.ndarray): The input image array.
        centroid (tuple): The (y, x) coordinates of the centroid.

    Returns:
        tuple: A tuple containing the x-direction diameter and y-direction diameter.
    """
    c_y, c_x = centroid
    # Extract data for x and y directions
    y_data = image[:, c_x]
    x_data = image[c_y, :]

    # Calculate diameters
    x_diameter = calculate_diameter(x_data)
    y_diameter = calculate_diameter(y_data)

    return x_diameter, y_diameter

def extract_radial_data(image, centroid_x, centroid_y, angle):
    """
    Extract data along a radial line from the centroid at a given angle.

    Args:
        image (np.ndarray): The input image array.
        centroid_x (int): The x-coordinate of the centroid.
        centroid_y (int): The y-coordinate of the centroid.
        angle (float): The angle in degrees.

    Returns:
        np.ndarray: The extracted data along the radial line.
    """
    height, width = image.shape
    angle_rad = np.deg2rad(angle)
    max_length = int(max(
        math.sqrt(centroid_x**2 + centroid_y**2),
        math.sqrt((width - centroid_x)**2 + centroid_y**2),
        math.sqrt(centroid_x**2 + (height - centroid_y)**2),
        math.sqrt((width - centroid_x)**2 + (height - centroid_y)**2)
    ))
    distances = np.arange(-max_length, max_length + 1)
    x_coords = np.round(centroid_x + distances * np.cos(angle_rad)).astype(int)
    y_coords = np.round(centroid_y + distances * np.sin(angle_rad)).astype(int)
    valid_mask = (0 <= x_coords) & (x_coords < width) & (0 <= y_coords) & (y_coords < height)
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]
    return image[y_coords, x_coords]


def calculate_diameter_at_angle(image, centroid_x, centroid_y, angle):
    """
    Calculate the diameter at y = 1/e + b at a given angle.

    Args:
        image (np.ndarray): The input image array.
        centroid (tuple): The (y, x) coordinates of the centroid.
        angle (float): The angle in degrees.

    Returns:
        float: The calculated diameter, or None if fitting fails or A <= 0.
    """
    radial_data = extract_radial_data(image, centroid_x, centroid_y, angle)
    return calculate_diameter(radial_data)


def find_spot_border(image):
    """
    处理光斑图片，计算噪声阈值，去除噪声并拟合包含光斑的圆形。

    参数:
    image (numpy.ndarray): 输入的光斑图片，应为单通道灰度图像。

    返回:
    numpy.ndarray: 去除噪声后的图像。
    tuple: 拟合圆形的圆心坐标 (x, y) 和半径。
    """
    # 步骤 2: canny边缘检测
    noise_threshold = np.mean(image) * 0.5
    denoised_image = np.where(image > noise_threshold, image, 0)
    # 步骤 3: 去除噪声
    denoised_image = cv2.fastNlMeansDenoising(denoised_image, None, 10, 7, 21)
    # denoised_image = cv2.Canny(image, 1, 1)

    # 步骤 3: 拟合一个圆形正好包含光斑
    contours, _ = cv2.findContours(denoised_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        center = (int(x), int(y))
        radius = int(radius)
    else:
        center = (0, 0)
        radius = 0

    return denoised_image, (center, radius)