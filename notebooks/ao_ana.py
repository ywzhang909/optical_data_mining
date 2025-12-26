# %%
import pandas as pd
from scipy.ndimage import center_of_mass
from scipy.optimize import curve_fit
import numpy as np

from pathlib import Path
import matplotlib.pyplot as plt
# %%

data_path = Path('D:/workspace/data-mining/data/ao/wf-less/20251226_163326')
data = pd.read_pickle(data_path.glob('*.pkl').__next__(), compression='zip')

# %%

img_ser = data['_img']
best_iter = data['J'].argmax()

def gaussian(x, mu, sigma, A, b):
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
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + b

def fitting_gaussian(data):
    """
    Fit a Gaussian function to a given data series.
    Args:
        data (np.ndarray): The data series to fit a Gaussian function.
    Returns:
        tuple: A tuple containing the fitted parameters (A, b, mu, sigma) and the fitted curve.
    """
    x_data = np.arange(len(data))
    initial_guess = [np.argmax(data), 10, np.max(data), 0]
    try:
        (mu, sigma, A, b), covariance = curve_fit(gaussian, x_data, data, p0=initial_guess)
    except RuntimeError:
        return (np.nan, np.nan, np.nan, np.nan), np.nan

    return (mu, sigma, A, b), covariance

def calculate_diameter(sigma):
    diameter = 2 * sigma
    return diameter

def calculate_xy_diameters(image):
    """
    Calculate the diameters at y = 1/e + b in x and y directions.

    Args:
        image (np.ndarray): The input image array.
        centroid (tuple): The (y, x) coordinates of the centroid.

    Returns:
        tuple: A tuple containing the x-direction diameter and y-direction diameter.
    """
    coords = center_of_mass(image)
    center_y = int(coords[0])  # type: ignore
    center_x = int(coords[1])  # type: ignore
    # Extract data for x and y directions
    y_data = image[:, center_x]
    x_data = image[center_y, :]

    # Calculate diameters
    (mu, sigma, A, b), conv = fitting_gaussian(x_data)
    x_diameter = calculate_diameter(sigma)
    (mu, sigma, A, b), conv = fitting_gaussian(y_data)
    y_diameter = calculate_diameter(sigma)

    return {
        "x":x_diameter, 
        "y":y_diameter
    }

diameter = data.apply(lambda x: calculate_xy_diameters(x['_img']), axis=1, result_type='expand')
data = pd.concat([data, diameter], axis=1)
data['intensity'] = img_ser.apply(lambda x: np.max)
# %%
data['xy'] = np.sqrt(data['x']**2 + data['y']**2)
data[
    ['xy',
    'intensity',
    'pib']
].plot(subplots=True, legend=True)
# %%
from PIL import Image

Image.read(data_path)
