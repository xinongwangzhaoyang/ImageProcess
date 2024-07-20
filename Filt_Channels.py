import os
from PIL import Image


def remove_color_above_threshold(image_path, output_path, UpFilt,DownFlit):
    # 打开图像
    image = Image.open(image_path).convert('RGB')
    pixels = image.load()

    # 遍历图像的每个像素
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            r, g, b = pixels[i, j]
            # 如果任何颜色通道值大于阈值，将该像素改为黑色（或其他你想要的颜色）
            if r > UpFilt or g > UpFilt or b > UpFilt or r < DownFlit or g < DownFlit or b < DownFlit:
                pixels[i, j] = (0, 0, 0)  # 将该像素设为黑色

    # 保存处理后的图像
    image.save(output_path)


def process_folder(input_folder, output_folder, UpFilt,DownFlit):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, filename)
            remove_color_above_threshold(image_path, output_image_path, UpFilt,DownFlit)


# 示例使用
UpFilt = 200
DownFlit=50
input_folder = './../images'  # 替换为实际输入文件夹路径
output_folder = 'outputs/FiltChannels/origin/up200&down50'  # 替换为实际输出文件夹路径

process_folder(input_folder, output_folder, UpFilt,DownFlit)
