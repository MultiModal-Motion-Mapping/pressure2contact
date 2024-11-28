import numpy as np
import matplotlib.pyplot as plt

lista = np.array([ 41,  36,  36,  36,  60,  46,  32,  43,  42,  70,  38,  31,  47,
        56,  51,  54, 100,  62,  36,  40,  32,  56,  55,  80, 133, 119,
        53,  37,  62,  71,  54,  41,  35,  72, 198, 182,  75,  55,  50,
        66,  35, 120,  94,  83,  53,  47,  40,  44,  53,  46,  62,  63,
        78,  74,  29,  55,  67,  66,  87,  69,  33,  31,  51,  72,  85,
       135, 124,  50,  36,  54, 102, 151, 217, 245,  76,  40,  92, 158,
       225, 255, 144,  59,  82,  95, 140,  81, 121, 203, 142,  62, 155,
       100,  82,  83,  48,  49, 158, 107,  42,  85,  60])

hist, bins = np.histogram(lista, bins=np.arange(257))  # 0-255的值，257个bin

# 2. 计算累积分布函数（CDF）
cdf = np.cumsum(hist)  # 累计和
cdf_normalized = cdf / cdf[-1]  # 归一化到[0, 1]

# 3. 使用CDF对原始数据进行均衡化
equalized_data = np.interp(lista, bins[:-1], cdf_normalized).astype(float)

print('hi')
# # 4. 绘制原始数据和均衡化后的数据的直方图
# plt.figure(figsize=(10, 5))

# # 原始直方图
# plt.subplot(1, 2, 1)
# plt.hist(lista, bins=256, range=(0, 255), color='blue', alpha=0.7, label='Original')
# plt.title('Original Histogram')
# plt.xlabel('Pixel Value')
# plt.ylabel('Frequency')

# # 均衡化后的直方图
# plt.subplot(1, 2, 2)
# plt.hist(equalized_data, bins=256, range=(0, 1), color='red', alpha=0.7, label='Equalized')
# plt.title('Equalized Histogram (Normalized to [0, 1])')
# plt.xlabel('Pixel Value')
# plt.ylabel('Frequency')

# plt.tight_layout()
# plt.show()