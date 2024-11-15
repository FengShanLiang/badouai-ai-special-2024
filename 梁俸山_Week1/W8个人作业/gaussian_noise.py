import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_gaussian_noise_to_image(image, mean=0, std=25):
    """
    向图像添加高斯噪声
    :param image: 输入图像（numpy数组）
    :param mean: 噪声均值
    :param std: 噪声标准差
    :return: 添加噪声后的图像
    """
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

# 读取原始图像
image_path = 'image.jpg'  # 替换为你的图像路径
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 读取灰度图像

# 添加高斯噪声
noisy_image = add_gaussian_noise_to_image(image, mean=0, std=25)

# 保存加入噪声后的图片
output_path = 'noisy_image.jpg'  # 设置输出路径
cv2.imwrite(output_path, noisy_image)
print(f"噪声图片已保存到: {output_path}")


