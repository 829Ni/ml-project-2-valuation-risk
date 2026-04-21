import pandas as pd
import os

# 1. 获取项目根目录（和 data_preprocess.py 保持一致）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# 2. 定义文件路径
train_processed_path = os.path.join(PROCESSED_DATA_DIR, "ames_housing_train_processed.csv")
test_processed_path = os.path.join(PROCESSED_DATA_DIR, "ames_housing_test_processed.csv")
credit_processed_path = os.path.join(PROCESSED_DATA_DIR, "german_credit_processed.csv")

# 3. 检查房产训练集
print("===== 房产训练集 =====")
if os.path.exists(train_processed_path):
    df_train = pd.read_csv(train_processed_path)
    print(f"✅ 数据行数：{len(df_train)}")
    print(f"✅ 数据列数：{len(df_train.columns)}")
    print("\n关键列统计：")
    print(df_train[['GrLivArea', 'SalePrice', 'UnitPrice']].describe().round(2))
else:
    print(f"❌ 未找到文件：{train_processed_path}")

# 4. 检查房产测试集
print("\n===== 房产测试集 =====")
if os.path.exists(test_processed_path):
    df_test = pd.read_csv(test_processed_path)
    print(f"✅ 数据行数：{len(df_test)}")
    print(f"✅ 数据列数：{len(df_test.columns)}")
else:
    print(f"❌ 未找到文件：{test_processed_path}")

# 5. 检查信贷数据
print("\n===== 信贷数据 =====")
if os.path.exists(credit_processed_path):
    df_credit = pd.read_csv(credit_processed_path)
    print(f"✅ 数据行数：{len(df_credit)}")
    print(f"✅ 数据列数：{len(df_credit.columns)}")
    print(f"✅ 风险分布：{df_credit['risk'].value_counts().to_dict()}（0=好客户，1=坏客户）")
else:
    print(f"❌ 未找到文件：{credit_processed_path}")

print("\n🎉 数据检查完成！")