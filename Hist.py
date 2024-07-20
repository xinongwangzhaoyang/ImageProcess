import os
import pandas as pd
from PIL import Image

def extract_histogram(image_path):
    image = Image.open(image_path).convert('RGB')
    histogram = image.histogram()
    r_hist = histogram[0:256]
    g_hist = histogram[256:512]
    b_hist = histogram[512:768]
    return r_hist, g_hist, b_hist
def save_histograms_to_excel(histograms, output_excel, color):
    # Prepare data for DataFrame
    data = []
    for filename, histogram in histograms.items():
        row = [filename] + histogram
        data.append(row)

    # Create a DataFrame
    columns = ['Filename'] + [f'{color}_{i}' for i in range(256)]
    df = pd.DataFrame(data, columns=columns)

    # Save DataFrame to Excel
    df.to_excel(output_excel, index=False)

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Dictionaries to hold histograms for each color channel
    r_histograms = {}
    g_histograms = {}
    b_histograms = {}

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            r_hist, g_hist, b_hist = extract_histogram(image_path)

            # Store histograms in dictionaries
            r_histograms[filename] = r_hist
            g_histograms[filename] = g_hist
            b_histograms[filename] = b_hist

    # Save all histograms to Excel files
    save_histograms_to_excel(r_histograms, os.path.join(output_folder, 'red_histograms.xlsx'), 'Red')
    save_histograms_to_excel(g_histograms, os.path.join(output_folder, 'green_histograms.xlsx'), 'Green')
    save_histograms_to_excel(b_histograms, os.path.join(output_folder, 'blue_histograms.xlsx'), 'Blue')
# 示例使用
input_folder = './../images'  # 替换为实际输入文件夹路径
output_folder = './Hist'  # 替换为保存Excel文件的路径

process_folder(input_folder, output_folder)
