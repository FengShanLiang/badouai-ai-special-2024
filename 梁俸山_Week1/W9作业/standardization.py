from sklearn.preprocessing import StandardScaler
import numpy as np

# 假设有一个数据集
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]], dtype=float)

# 创建标准化对象
scaler = StandardScaler()

# 进行标准化
standardized_data = scaler.fit_transform(data)

print("原始数据：")
print(data)
print("标准化后的数据：")
print(standardized_data)
