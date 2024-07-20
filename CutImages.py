import os
from PIL import Image
# 进行图像的剪切操作，可以设置crop_left、crop_top、rop_right、crop_bottom的值来自定义剪切区域，
# 示例剪切的是中心区域的一部分图片
def crop_center(image, crop_width, crop_height):
    img_width, img_height = image.size
    crop_left = (img_width - crop_width) // 2
    crop_top = (img_height - crop_height) // 2
    crop_right = crop_left + crop_width
    crop_bottom = crop_top + crop_height
    return image.crop((crop_left, crop_top, crop_right, crop_bottom))


def process_images(input_folder, output_folder, crop_width, crop_height):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)
            cropped_image = crop_center(image, crop_width, crop_height)
            cropped_image.save(os.path.join(output_folder, filename))


input_folder = ''  # 替换为实际输入文件夹路径
output_folder = ''  # 替换为实际输出文件夹路径
crop_width = 1000  # 设置裁切区域的宽度
crop_height = 1000  # 设置裁切区域的高度

process_images(input_folder, output_folder, crop_width, crop_height)
