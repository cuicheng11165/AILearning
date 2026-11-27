"""
MLP（多层感知机）房价预测极简 Demo（纯 Python 实现，零第三方依赖）

对应笔记：docs/02-architectures/01-mlp-and-fnn.md
运行方式：python demos/mlp/house_price_prediction.py

演示流程：
1. 生成带非线性规律的模拟房价数据集（4 个特征 -> 1 个连续房价）
2. 划分训练集 / 测试集，并对特征做 Z-score 标准化
3. 构建一个简单 MLP：输入层(4) -> 隐藏层(16, ReLU) -> 输出层(1)
4. 完整执行：前向传播 -> 计算 MSE 损失 -> 反向传播 -> 梯度下降更新参数
5. 在测试集评估误差，并对几套新房子做房价预测
"""

import math
import random


# ============================================================
# 1. 构造模拟房价数据（包含非线性关系，体现 MLP 的拟合能力）
# ============================================================
def true_price_formula(area: float, rooms: int, age: float, subway_km: float) -> float:
    """
    真实房价规律（单位：万元），刻意加入一些非线性效应：
    - 面积越大价格越高（基础单价约 4.5 万/平，大户型略带溢价）
    - 房间数合理增加会提升总价
    - 房龄越老折旧越多（早期折旧快，后期趋缓）
    - 离地铁越近越贵（1.5km 内地铁房有明显溢价，呈非线性衰减）
    """
    base = 4.2 * area + 0.004 * (area ** 2)
    room_bonus = 18.0 * rooms
    age_penalty = -35.0 * math.sqrt(age)
    subway_bonus = 95.0 / (1.0 + subway_km)  # 越近溢价越高
    return base + room_bonus + age_penalty + subway_bonus


def generate_dataset(n_samples: int = 300, seed: int = 42):
    rng = random.Random(seed)
    features = []
    targets = []

    for _ in range(n_samples):
        area = rng.uniform(45.0, 180.0)                      # 面积：45~180 ㎡
        rooms = max(1, min(5, round(area / 35.0 + rng.uniform(-0.7, 0.7))))  # 房间数：1~5
        age = rng.uniform(0.0, 30.0)                         # 房龄：0~30 年
        subway_km = rng.uniform(0.2, 6.0)                    # 距地铁：0.2~6.0 km

        noise = rng.gauss(0.0, 12.0)                         # 模拟市场波动噪声（±12万）
        price = true_price_formula(area, rooms, age, subway_km) + noise

        features.append([area, float(rooms), age, subway_km])
        targets.append(price)

    return features, targets


# ============================================================
# 2. 特征标准化（仅用训练集均值和标准差，避免数据泄漏）
# ============================================================
def fit_standardizer(X_train):
    n_samples = len(X_train)
    n_features = len(X_train[0])
    means = [sum(row[j] for row in X_train) / n_samples for j in range(n_features)]
    stds = []
    for j in range(n_features):
        var = sum((row[j] - means[j]) ** 2 for row in X_train) / n_samples
        stds.append(math.sqrt(var) if var > 1e-12 else 1.0)
    return means, stds


def transform_features(X, means, stds):
    return [[(row[j] - means[j]) / stds[j] for j in range(len(means))] for row in X]


# ============================================================
# 3. 手写一个极简 2 层 MLP：4 -> 16 (ReLU) -> 1 (线性输出)
# ============================================================
class SimpleMLP:
    def __init__(self, input_dim: int, hidden_dim: int, seed: int = 42):
        rng = random.Random(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # He 初始化（适合 ReLU 激活函数）
        scale1 = math.sqrt(2.0 / input_dim)
        self.W1 = [[rng.gauss(0.0, scale1) for _ in range(input_dim)] for _ in range(hidden_dim)]
        self.b1 = [0.0 for _ in range(hidden_dim)]

        scale2 = math.sqrt(1.0 / hidden_dim)
        self.W2 = [rng.gauss(0.0, scale2) for _ in range(hidden_dim)]
        self.b2 = 0.0

    def forward(self, x):
        """
        前向传播：
        z1 = W1 * x + b1
        h1 = ReLU(z1)
        y_hat = W2 · h1 + b2   （回归任务输出层不加非线性激活）
        """
        z1 = []
        h1 = []
        for i in range(self.hidden_dim):
            s = self.b1[i]
            w_row = self.W1[i]
            for j in range(self.input_dim):
                s += w_row[j] * x[j]
            z1.append(s)
            h1.append(s if s > 0.0 else 0.0)  # ReLU

        y_hat = self.b2
        for i in range(self.hidden_dim):
            y_hat += self.W2[i] * h1[i]

        return y_hat, z1, h1

    def train_epoch(self, X, y, lr: float = 0.01):
        """一个 epoch 的全批量梯度下降（Batch GD），逻辑最直观"""
        n = len(X)
        dW1 = [[0.0] * self.input_dim for _ in range(self.hidden_dim)]
        db1 = [0.0] * self.hidden_dim
        dW2 = [0.0] * self.hidden_dim
        db2 = 0.0

        total_loss = 0.0

        for x_i, y_i in zip(X, y):
            # 1) 前向传播
            y_hat, z1, h1 = self.forward(x_i)

            # 2) 计算单样本 MSE 损失：(y_hat - y)^2
            err = y_hat - y_i
            total_loss += err * err

            # 3) 反向传播（链式法则）
            # dL / dy_hat = 2 * (y_hat - y) / n
            d_yhat = (2.0 * err) / n

            # 输出层参数梯度
            db2 += d_yhat
            for i in range(self.hidden_dim):
                dW2[i] += d_yhat * h1[i]

            # 隐藏层参数梯度
            for i in range(self.hidden_dim):
                if z1[i] > 0.0:  # ReLU 导数：z > 0 时为 1，否则为 0
                    d_z1 = d_yhat * self.W2[i]
                    db1[i] += d_z1
                    for j in range(self.input_dim):
                        dW1[i][j] += d_z1 * x_i[j]

        # 4) 梯度下降更新参数
        self.b2 -= lr * db2
        for i in range(self.hidden_dim):
            self.W2[i] -= lr * dW2[i]
            self.b1[i] -= lr * db1[i]
            for j in range(self.input_dim):
                self.W1[i][j] -= lr * dW1[i][j]

        return total_loss / n

    def predict(self, x):
        y_hat, _, _ = self.forward(x)
        return y_hat


# ============================================================
# 4. 训练、评估与新房预测
# ============================================================
def evaluate(model: SimpleMLP, X, y_true_wan, y_scale: float):
    abs_errors = []
    sq_errors = []
    for x_i, y_true in zip(X, y_true_wan):
        pred_wan = model.predict(x_i) * y_scale
        diff = pred_wan - y_true
        abs_errors.append(abs(diff))
        sq_errors.append(diff * diff)
    mae = sum(abs_errors) / len(abs_errors)
    rmse = math.sqrt(sum(sq_errors) / len(sq_errors))
    return mae, rmse


def main():
    # 生成 300 条数据：240 训练 + 60 测试
    X_raw, y_raw = generate_dataset(n_samples=300, seed=42)
    split = 240
    X_train_raw, y_train_wan = X_raw[:split], y_raw[:split]
    X_test_raw, y_test_wan = X_raw[split:], y_raw[split:]

    # 特征标准化（让面积、房间数、房龄、地铁距离处于同一量纲）
    means, stds = fit_standardizer(X_train_raw)
    X_train = transform_features(X_train_raw, means, stds)
    X_test = transform_features(X_test_raw, means, stds)

    # 目标值缩放：单位从“万元”缩放到“百万元”（数值更平滑，梯度不易爆炸）
    y_scale = 100.0
    y_train = [p / y_scale for p in y_train_wan]

    # 初始化 MLP：4 维输入 -> 16 个隐藏神经元 (ReLU) -> 1 维输出
    model = SimpleMLP(input_dim=4, hidden_dim=16, seed=42)

    print("=== 开始训练 MLP（结构：4 -> 16(ReLU) -> 1）===")
    epochs = 1200
    lr = 0.03
    for epoch in range(1, epochs + 1):
        loss = model.train_epoch(X_train, y_train, lr=lr)
        if epoch == 1 or epoch % 200 == 0:
            train_mae, _ = evaluate(model, X_train, y_train_wan, y_scale)
            test_mae, test_rmse = evaluate(model, X_test, y_test_wan, y_scale)
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"Train MSE(缩放后): {loss:7.4f} | "
                f"Train MAE: {train_mae:5.1f} 万 | "
                f"Test MAE: {test_mae:5.1f} 万 (RMSE: {test_rmse:5.1f} 万)"
            )

    # 用训练好的 MLP 预测几套新房子
    print("\n=== 用训练好的 MLP 预测新房价格 ===")
    sample_houses = [
        ("老破小近地铁", [60.0, 2, 22.0, 0.4]),
        ("刚需三居次新", [89.0, 3, 5.0, 1.2]),
        ("远郊大平层　", [150.0, 4, 3.0, 4.5]),
        ("核心次新三居", [135.0, 3, 2.0, 0.5]),
    ]

    for name, feat in sample_houses:
        x_norm = transform_features([feat], means, stds)[0]
        pred = model.predict(x_norm) * y_scale
        actual = true_price_formula(feat[0], int(feat[1]), feat[2], feat[3])
        err = pred - actual
        print(
            f"[{name}] 面积={feat[0]:3.0f}㎡, {int(feat[1])}室, 房龄={feat[2]:2.0f}年, 地铁={feat[3]:.1f}km "
            f"--> 真实参考价: {actual:5.1f}万 | MLP预测: {pred:5.1f}万 (误差 {err:+5.1f}万)"
        )


if __name__ == "__main__":
    main()
