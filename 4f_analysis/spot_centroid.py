import numpy as np
from PIL import Image

def calculate_centroid(image_path, output_path):
# Open the TIFF image
with Image.open(image_path) as img:
# Convert to grayscale if necessary
if img.mode != 'L':
img = img.convert('L')
# Convert to numpy array
data = np.array(img)
# Calculate centroid
total = np.sum(data)
if total == 0:
raise ValueError("Empty image - all pixel values are zero")
# Get image dimensions
height, width = data.shape
# Create coordinate grids
x, y = np.indices((width, height))
# Calculate weighted coordinates
x_center = np.sum(x * data) / total
y_center = np.sum(y * data) / total
# Save results to text file
with open(output_path, 'w') as f:
f.write(f"Centroid coordinates (x, y): ({x_center:.2f}, {y_center:.2f})\n")
f.write(f"Total intensity: {total:.2f}\n")
return x_center, y_center

# Example usage
if __name__ == "__main__":
image_path = "4f分析/光瞳20250606 10：23：11.96(20%平顶31).TIFF"
output_path = "4f分析/光瞳质心20250606 10：23：07(20%平顶31).TXT"
x, y = calculate_centroid(image_path, output_path)
print(f"Centroid calculated: ({x:.2f}, {y:.2f})")
