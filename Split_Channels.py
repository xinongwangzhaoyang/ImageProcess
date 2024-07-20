import os
from PIL import Image


def split_channels(image_path, output_red_path, output_green_path, output_blue_path):
    # 打开图像
    image = Image.open(image_path).convert('RGB')
    r, g, b = image.split()

    # 创建单独的红色、绿色和蓝色图像
    red_image = Image.merge("RGB", (r, Image.new('L', image.size), Image.new('L', image.size)))
    green_image = Image.merge("RGB", (Image.new('L', image.size), g, Image.new('L', image.size)))
    blue_image = Image.merge("RGB", (Image.new('L', image.size), Image.new('L', image.size), b))

    # 保存处理后的图像
    red_image.save(output_red_path)
    green_image.save(output_green_path)
    blue_image.save(output_blue_path)


def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    red_folder = os.path.join(output_folder, 'red')
    green_folder = os.path.join(output_folder, 'green')
    blue_folder = os.path.join(output_folder, 'blue')

    for folder in [red_folder, green_folder, blue_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            output_red_path = os.path.join(red_folder, filename)
            output_green_path = os.path.join(green_folder, filename)
            output_blue_path = os.path.join(blue_folder, filename)
            split_channels(image_path, output_red_path, output_green_path, output_blue_path)


# 示例使用
input_folder = './outputs/1000x1000'  # 替换为实际输入文件夹路径
output_folder = './outputs/SplitChannels/Split1000'  # 替换为实际输出文件夹路径

process_folder(input_folder, output_folder)
