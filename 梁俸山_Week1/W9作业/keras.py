import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 生成简单的训练数据
X_train = np.random.rand(100, 3)  # 100个样本，每个样本有3个特征
y_train = np.random.randint(0, 2, (100,))  # 100个二分类标签

# 构建神经网络
model = Sequential([
    Dense(16, input_dim=3, activation='relu'),  # 隐藏层，16个神经元，输入维度3
    Dense(1, activation='sigmoid')  # 输出层，二分类
])

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.01),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=8)

# 测试数据预测
X_test = np.random.rand(10, 3)  # 10个样本
predictions = model.predict(X_test)
print("预测结果：")
print(predictions)
