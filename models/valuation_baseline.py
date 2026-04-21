import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. 固定获取项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "ames_housing_train_processed.csv")
MODEL_PLOT_PATH = os.path.join(PROJECT_ROOT, "models", "valuation_baseline_plot.png")


# 2. 读取数据
def load_data():
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"❌ 未找到数据文件：{PROCESSED_DATA_PATH}")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df['GrLivArea'].values.reshape(-1, 1)
    y = df['SalePrice'].values
    print(f"✅ 数据加载完成：{len(X)} 条样本")
    return X, y


# 3. 手写一元线性回归
class LinearRegression:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y):
        # 增加全1列
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # 闭式解
        weights = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        self.b = weights[0]
        self.w = weights[1]
        print(f"✅ 模型训练完成：")
        print(f"   截距 b：{self.b:.2f}")
        print(f"   系数 w：{self.w:.2f}")
        print(f"   模型公式：房价 = {self.b:.2f} + {self.w:.2f} × 居住面积")

    def predict(self, X):
        return self.b + self.w * X


# 4. 修复后的 R² 计算（安全写法）
def calculate_r2(y_true, y_pred):
    """
    计算决定系数 R²，修正了数组维度问题
    """
    # 先把一维数组压平，避免维度不匹配
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    print(f"✅ 模型评估：R² 得分 = {r2:.4f}")
    return r2


# 5. 可视化
def plot_result(X, y, y_pred):
    # 解决中文乱码
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(10, 6))

    plt.scatter(X, y, color='#1f77b4', alpha=0.6, label='真实房价', s=30)
    plt.plot(X, y_pred, color='#ff7f0e', linewidth=2, label='回归预测线')

    plt.xlabel('居住面积（平方英尺）')
    plt.ylabel('房价（美元）')
    plt.title('居住面积 vs 房价（一元线性回归）')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(MODEL_PLOT_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 可视化图表已保存：{MODEL_PLOT_PATH}")


# 6. 主执行
if __name__ == "__main__":
    try:
        X, y = load_data()
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # 现在会输出正确的 R²（大约 0.5 左右）
        calculate_r2(y, y_pred)

        plot_result(X, y, y_pred)
        print("\n🎉 一元线性回归模型全流程执行完成！")
    except Exception as e:
        print(f"\n❌ 执行失败：{str(e)}")