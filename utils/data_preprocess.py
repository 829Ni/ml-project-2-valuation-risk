import pandas as pd
import os
import sys

# 1. 获取项目根目录路径（解决工作目录问题）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# 2. 确保基础文件夹存在
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


# ====================== 房产数据预处理（核心逻辑）======================
def process_housing_data():
    try:
        # 直接读取 data/raw/ 里的 train.csv 和 test.csv（使用绝对路径）
        train_path = os.path.join(RAW_DATA_DIR, "train.csv")
        test_path = os.path.join(RAW_DATA_DIR, "test.csv")

        # 检查文件是否存在
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"❌ 未找到训练文件：{train_path}，请确认文件已放入 data/raw/")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"❌ 未找到测试文件：{test_path}，请确认文件已放入 data/raw/")

        # 读取CSV文件
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)

        # 核心特征（完全匹配你CSV里的列名）
        core_features = [
            'Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea',
            'Street', 'Alley', 'Neighborhood', 'Condition1', 'Condition2',
            'GrLivArea', 'TotalBsmtSF', 'GarageArea', 'OverallQual',
            'OverallCond', 'YearBuilt', 'CentralAir', 'Fireplaces',
            'PoolArea', 'YrSold', 'MoSold', 'SaleCondition', 'SalePrice'
        ]

        # 筛选特征（训练集包含SalePrice，测试集不包含）
        df_train = train[core_features].copy()
        df_test = test[[col for col in core_features if col != 'SalePrice']].copy()

        # 缺失值处理
        df_train['LotFrontage'].fillna(df_train['LotFrontage'].median(), inplace=True)
        df_test['LotFrontage'].fillna(df_test['LotFrontage'].median(), inplace=True)
        df_train['TotalBsmtSF'].fillna(0, inplace=True)
        df_train['GarageArea'].fillna(0, inplace=True)
        df_test['TotalBsmtSF'].fillna(0, inplace=True)
        df_test['GarageArea'].fillna(0, inplace=True)
        df_train['Alley'].fillna('NA', inplace=True)
        df_test['Alley'].fillna('NA', inplace=True)

        # 分类特征转数值
        df_train['CentralAir'] = df_train['CentralAir'].map({'Y': 1, 'N': 0})
        df_test['CentralAir'] = df_test['CentralAir'].map({'Y': 1, 'N': 0})

        # 业务计算：每平米单价（仅训练集）
        df_train['UnitPrice'] = df_train['SalePrice'] / df_train['GrLivArea']

        # 计算同小区均价（用于后续估值对比）
        neighborhood_avg_price = df_train.groupby('Neighborhood')['UnitPrice'].mean().to_dict()
        df_train['NeighborhoodAvgUnitPrice'] = df_train['Neighborhood'].map(neighborhood_avg_price)
        df_test['NeighborhoodAvgUnitPrice'] = df_test['Neighborhood'].map(neighborhood_avg_price)

        # 保存处理后的数据（使用绝对路径）
        train_output = os.path.join(PROCESSED_DATA_DIR, "ames_housing_train_processed.csv")
        test_output = os.path.join(PROCESSED_DATA_DIR, "ames_housing_test_processed.csv")
        df_train.to_csv(train_output, index=False)
        df_test.to_csv(test_output, index=False)

        print("✅ 房产数据预处理完成！")
        print(f"📁 训练集输出：{train_output}")
        print(f"📁 测试集输出：{test_output}")
        print(f"📊 训练集数据形状：{df_train.shape}")
        print(f"📊 测试集数据形状：{df_test.shape}")

    except Exception as e:
        print(f"\n❌ 房产数据预处理失败：{str(e)}")
        sys.exit(1)


# ====================== 信贷数据预处理 ======================
def process_credit_data():
    try:
        credit_path = os.path.join(RAW_DATA_DIR, "german.data")
        if not os.path.exists(credit_path):
            print("\n⚠️ 未找到 german.data，跳过信贷数据处理")
            return

        # 信贷数据列名
        col_names = [
            'checking_acc', 'duration', 'credit_history', 'purpose', 'credit_amount',
            'savings', 'employment', 'installment_rate', 'status_sex', 'guarantors',
            'residence', 'property', 'age', 'installment_plans', 'housing',
            'existing_credits', 'job', 'maintenance', 'telephone', 'foreign_worker', 'risk'
        ]

        # 读取空格分隔的信贷数据
        df_credit = pd.read_csv(credit_path, sep=" ", names=col_names)
        # 风险标签转为 0=好客户，1=坏客户
        df_credit['risk'] = df_credit['risk'] - 1

        # 保存处理后的数据（使用绝对路径）
        credit_output = os.path.join(PROCESSED_DATA_DIR, "german_credit_processed.csv")
        df_credit.to_csv(credit_output, index=False)

        print("\n✅ 信贷数据预处理完成！")
        print(f"📁 信贷数据输出：{credit_output}")
        print(f"📊 信贷数据形状：{df_credit.shape}")

    except Exception as e:
        print(f"\n❌ 信贷数据预处理失败：{str(e)}")


# ====================== 执行预处理 ======================
if __name__ == "__main__":
    print("🚀 开始执行数据预处理...\n")

    process_housing_data()
    process_credit_data()

    print("\n🎉 所有数据预处理步骤执行完毕！")