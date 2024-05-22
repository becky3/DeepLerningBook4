import dezero.layers as L
import numpy as np

linear = L.Linear(10)  # 出力サイズだけを指定

batch_size, input_size = 100, 5

x = np.random.randn(batch_size, input_size)
y = linear(x)

print("y shape:", y.shape)
print("param shape:", linear.W.shape, linear.b.shape)

for param in linear.params():
    print(param.name, param.shape)
