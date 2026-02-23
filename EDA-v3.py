# 1. 基础设置和数据加载
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# 设置图表样式 (移除中文字体设置，使用默认英文字体)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 2. 加载数据
# from google.colab import drive
#
# drive.mount('/content/drive')

# 读取数据
df = pd.read_csv("C:/Users/DELL/PycharmProjects/pythonProject/529/group_project/train_data.csv")
print("Dataset shape:", df.shape)
print("\nDataset basic information:")
print(df.info())
target_col = 'contest-tmp2m-14d__tmp2m'

# 3. 数据概览
print("=" * 80)
print("Data Overview")
print("=" * 80)

# 显示前几行数据
print("\nFirst 5 rows:")
print(df.head())

# 检查数据类型
print("\nData type distribution:")
print(df.dtypes.value_counts())

# 检查缺失值
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_values,
    'Missing Percentage': missing_percentage
}).sort_values('Missing Count', ascending=False)

print("\nMissing value analysis:")
print(missing_df[missing_df['Missing Count'] > 0].head(20))

# 3.5 缺失值处理
print("\n" + "=" * 80)
print("Missing Value Handling")
print("=" * 80)

# 备份原始缺失情况
missing_before = df.isnull().sum().sum()
print(f"Total missing values before handling: {missing_before}")

# 步骤1: 删除缺失率超过50%的特征
missing_ratio = df.isnull().mean() * 100
high_missing_cols = missing_ratio[missing_ratio > 50].index.tolist()
if high_missing_cols:
    print(f"Dropping columns with >50% missing values: {high_missing_cols}")
    df.drop(columns=high_missing_cols, inplace=True)

# 步骤2: 处理目标变量缺失 - 删除对应行
if target_col in df.columns:
    target_missing = df[target_col].isnull().sum()
    if target_missing > 0:
        print(f"Dropping {target_missing} rows with missing target variable.")
        df.dropna(subset=[target_col], inplace=True)

# 步骤3: 处理日期列缺失（如果有） - 删除对应行
if 'startdate' in df.columns:
    date_missing = df['startdate'].isnull().sum()
    if date_missing > 0:
        print(f"Dropping {date_missing} rows with missing startdate.")
        df.dropna(subset=['startdate'], inplace=True)

# 步骤4: 对数值特征进行中位数填充，并添加缺失指示列
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# 排除目标变量（已处理）和可能的非预测列（但保留所有数值列）
if target_col in numeric_cols:
    numeric_cols.remove(target_col)

for col in numeric_cols:
    if df[col].isnull().any():
        # 记录缺失指示器
        missing_indicator = df[col].isnull().astype(int)
        # 中位数填充
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        # 添加缺失指示列
        df[col + '_missing'] = missing_indicator
        print(f"  Filled {col} with median, added missing indicator.")

# 步骤5: 对分类特征进行众数填充，并添加缺失指示列
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
# 排除日期列（已处理）
if 'startdate' in categorical_cols:
    categorical_cols.remove('startdate')

for col in categorical_cols:
    if df[col].isnull().any():
        missing_indicator = df[col].isnull().astype(int)
        # 众数填充，若无众数则用 'Unknown'
        mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
        df[col].fillna(mode_val, inplace=True)
        df[col + '_missing'] = missing_indicator
        print(f"  Filled {col} with mode, added missing indicator.")

# 步骤6: 检查剩余缺失值
missing_after = df.isnull().sum().sum()
print(f"Total missing values after handling: {missing_after}")
if missing_after > 0:
    print("Remaining missing values (should be none if all handled):")
    print(df.isnull().sum()[df.isnull().sum() > 0])
else:
    print("No missing values remain.")

# 4. 时间序列分析
print("\n" + "=" * 80)
print("Time Series Analysis")
print("=" * 80)


# 转换日期格式
def fix_date_format(x):
    try:
        # 如果是 YYYY/MM/DD 格式
        if isinstance(x, str) and x.count('/') == 2 and len(x.split('/')[0]) == 4:
            return pd.to_datetime(x, format='%Y/%m/%d')
        else:
            return pd.to_datetime(x, format='%m/%d/%y')
    except Exception as e:
        return pd.NaT


df['startdate'] = df['startdate'].apply(fix_date_format)
df['year'] = df['startdate'].dt.year
df['month'] = df['startdate'].dt.month
df['day'] = df['startdate'].dt.day
df['day_of_year'] = df['startdate'].dt.dayofyear
df['week_of_year'] = df['startdate'].dt.isocalendar().week

# 检查时间范围
print(f"Time range: {df['startdate'].min()} to {df['startdate'].max()}")
print(f"Years covered: {sorted(df['year'].unique())}")

# 检查重复数据
duplicate_rows = df.duplicated().sum()
print(f"Duplicate rows: {duplicate_rows}")

# 5. 目标变量分析
print("\n" + "=" * 80)
print("Target Variable Analysis - contest-tmp2m-14d__tmp2m")
print("=" * 80)

target_col = 'contest-tmp2m-14d__tmp2m'
if target_col in df.columns:
    target_stats = df[target_col].describe()
    print("Target variable statistics:")
    print(target_stats)

    # 目标变量分布
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 直方图
    axes[0, 0].hist(df[target_col], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(df[target_col].mean(), color='red', linestyle='--',
                       label=f'Mean: {df[target_col].mean():.2f}°C')
    axes[0, 0].axvline(df[target_col].median(), color='green', linestyle='--',
                       label=f'Median: {df[target_col].median():.2f}°C')
    axes[0, 0].set_xlabel('Temperature (°C)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Target Variable Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 箱线图
    axes[0, 1].boxplot(df[target_col].dropna())
    axes[0, 1].set_ylabel('Temperature (°C)')
    axes[0, 1].set_title('Target Variable Boxplot')
    axes[0, 1].grid(True, alpha=0.3)

    # QQ图
    stats.probplot(df[target_col].dropna(), dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('QQ Plot (Normality Test)')
    axes[1, 0].grid(True, alpha=0.3)

    # 时间序列图 (采样显示)
    sample_df = df.iloc[::100]  # 每100个数据点取一个
    axes[1, 1].plot(sample_df['startdate'], sample_df[target_col], alpha=0.7)
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Temperature (°C)')
    axes[1, 1].set_title('Target Variable Time Series (Sampled)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 正态性检验
    from scipy.stats import shapiro

    sample_target = df[target_col].dropna().sample(n=min(5000, len(df)), random_state=42)
    stat, p_value = shapiro(sample_target)
    print(f"\nShapiro-Wilk Normality Test:")
    print(f"  Statistic: {stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Normal distribution: {'Yes' if p_value > 0.05 else 'No'}")

# 6. 地理位置分析
print("\n" + "=" * 80)
print("Geographic Analysis")
print("=" * 80)

if 'lat' in df.columns and 'lon' in df.columns:
    print(f"Latitude range: {df['lat'].min():.2f} to {df['lat'].max():.2f}")
    print(f"Longitude range: {df['lon'].min():.2f} to {df['lon'].max():.2f}")
    print(f"Unique locations: {df[['lat', 'lon']].drop_duplicates().shape[0]}")

    # 地理位置分布
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 纬度分布
    axes[0].hist(df['lat'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0].set_xlabel('Latitude')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Latitude Distribution')
    axes[0].grid(True, alpha=0.3)

    # 经度分布
    axes[1].hist(df['lon'], bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Longitude Distribution')
    axes[1].grid(True, alpha=0.3)

    # 地理位置散点图
    sample_df = df.iloc[::100]  # 采样以减少数据点
    scatter = axes[2].scatter(sample_df['lon'], sample_df['lat'],
                              c=sample_df[target_col] if target_col in df.columns else 'blue',
                              alpha=0.5, s=10, cmap='coolwarm')
    axes[2].set_xlabel('Longitude')
    axes[2].set_ylabel('Latitude')
    axes[2].set_title('Geographic Distribution')
    if target_col in df.columns:
        plt.colorbar(scatter, ax=axes[2], label='Temperature (°C)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# 7. 数值特征分析
print("\n" + "=" * 80)
print("Numerical Features Analysis")
print("=" * 80)

# 选择数值型特征
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Numerical features count: {len(numeric_cols)}")
print("\nStatistics of first 10 numerical features:")
print(df[numeric_cols[:10]].describe().T)

# 8. 特征相关性分析
print("\n" + "=" * 80)
print("Feature Correlation Analysis")
print("=" * 80)

# 计算与目标变量的相关性
if target_col in numeric_cols:
    correlations = {}
    for col in numeric_cols:
        if col != target_col and df[col].notna().sum() > 0:
            valid_data = df[[col, target_col]].dropna()
            if len(valid_data) > 10:  # 确保有足够的数据点
                corr = valid_data.corr().iloc[0, 1]
                correlations[col] = corr

    # 转换为DataFrame并排序
    corr_df = pd.DataFrame(list(correlations.items()),
                           columns=['Feature', 'Correlation']).sort_values('Correlation', key=abs, ascending=False)

    print("Top 20 features correlated with target variable:")
    print(corr_df.head(20))

    # 可视化相关性最高的特征
    top_features = corr_df.head(10)['Feature'].tolist()

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, feature in enumerate(top_features[:10]):
        if feature in df.columns:
            valid_data = df[[feature, target_col]].dropna().iloc[::50]  # 采样显示
            axes[i].scatter(valid_data[feature], valid_data[target_col],
                            alpha=0.3, s=10)
            axes[i].set_xlabel(feature[:20] + '...' if len(feature) > 20 else feature)
            axes[i].set_ylabel('Target Temperature')
            axes[i].set_title(f'{feature[:15]}...\nCorrelation: {correlations[feature]:.3f}')
            axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 相关性热力图 (前15个特征)
    top_corr_features = corr_df.head(15)['Feature'].tolist() + [target_col]
    corr_matrix = df[top_corr_features].corr()

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap of Top 15 Features with Target', fontsize=14)
    plt.tight_layout()
    plt.show()

# 9. 分类变量分析
print("\n" + "=" * 80)
print("Categorical Features Analysis")
print("=" * 80)

# 识别分类变量 (非数值型且不是日期)
categorical_cols = []
for col in df.columns:
    if df[col].dtype == 'object' and col != 'startdate':
        categorical_cols.append(col)

print(f"Categorical features count: {len(categorical_cols)}")
if categorical_cols:
    print("\nCategorical features overview:")
    for col in categorical_cols[:10]:  # 只显示前10个
        print(f"\n{col}:")
        print(f"  Unique values: {df[col].nunique()}")
        print(f"  Missing values: {df[col].isnull().sum()}")
        print(f"  Top 5 values:")
        print(df[col].value_counts().head(5))

    # 对于唯一值较少的分类变量，可视化
    low_cardinality_cols = [col for col in categorical_cols if df[col].nunique() <= 10]
    if low_cardinality_cols:
        fig, axes = plt.subplots(len(low_cardinality_cols), 1,
                                 figsize=(10, 4 * len(low_cardinality_cols)))
        if len(low_cardinality_cols) == 1:
            axes = [axes]

        for i, col in enumerate(low_cardinality_cols[:5]):  # 最多显示5个
            value_counts = df[col].value_counts()
            axes[i].bar(range(len(value_counts)), value_counts.values)
            axes[i].set_xticks(range(len(value_counts)))
            axes[i].set_xticklabels(value_counts.index, rotation=45)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'{col} Distribution (Unique values: {df[col].nunique()})')
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

# 10. 季节性分析
print("\n" + "=" * 80)
print("Seasonal Analysis")
print("=" * 80)

if target_col in df.columns and 'month' in df.columns:
    # 按月分组统计
    monthly_stats = df.groupby('month')[target_col].agg(['mean', 'std', 'min', 'max', 'count'])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 月平均温度
    axes[0, 0].plot(monthly_stats.index, monthly_stats['mean'], marker='o', linewidth=2, color='red')
    axes[0, 0].fill_between(monthly_stats.index,
                            monthly_stats['mean'] - monthly_stats['std'],
                            monthly_stats['mean'] + monthly_stats['std'],
                            alpha=0.3, color='red')
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('Average Temperature (°C)')
    axes[0, 0].set_title('Monthly Average Temperature (±Std Dev)')
    axes[0, 0].set_xticks(range(1, 13))
    axes[0, 0].grid(True, alpha=0.3)

    # 温度范围
    axes[0, 1].bar(monthly_stats.index, monthly_stats['max'] - monthly_stats['min'],
                   alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Temperature Range (°C)')
    axes[0, 1].set_title('Monthly Temperature Range (Max-Min)')
    axes[0, 1].set_xticks(range(1, 13))
    axes[0, 1].grid(True, alpha=0.3)

    # 温度分布箱线图
    monthly_data = []
    month_labels = []
    for month in range(1, 13):
        month_data = df[df['month'] == month][target_col].dropna()
        if len(month_data) > 0:
            monthly_data.append(month_data)
            month_labels.append(f'Month {month}')

    axes[1, 0].boxplot(monthly_data)
    axes[1, 0].set_xticklabels(month_labels)
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Temperature (°C)')
    axes[1, 0].set_title('Monthly Temperature Distribution Boxplot')
    axes[1, 0].grid(True, alpha=0.3)

    # 数据点数量
    axes[1, 1].bar(monthly_stats.index, monthly_stats['count'], alpha=0.7, color='green')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Data Points Count')
    axes[1, 1].set_title('Monthly Data Points Count')
    axes[1, 1].set_xticks(range(1, 13))
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# 11. 预测模型特征分析 (NMME)
print("\n" + "=" * 80)
print("NMME Prediction Model Features Analysis")
print("=" * 80)

# 找出所有NMME相关的特征
nmme_features = [col for col in df.columns if 'nmme' in col.lower() and col in numeric_cols]
print(f"NMME numerical features count: {len(nmme_features)}")
if nmme_features:
    print("\nFirst 10 NMME features:")
    for feature in nmme_features[:10]:
        print(f"  {feature}")

    # 分析NMME特征统计
    print("\nNMME features statistics:")
    print(df[nmme_features[:5]].describe().T)

    # NMME特征与目标变量的关系
    if target_col in df.columns and nmme_features:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, feature in enumerate(nmme_features[:6]):
            valid_data = df[[feature, target_col]].dropna().iloc[::100]  # 采样
            axes[i].scatter(valid_data[feature], valid_data[target_col],
                            alpha=0.3, s=10)
            axes[i].set_xlabel(feature[:20] + '...' if len(feature) > 20 else feature)
            axes[i].set_ylabel('Actual Temperature')
            axes[i].set_title(f'NMME Prediction vs Actual Temperature\n{feature[:20]}...')
            axes[i].grid(True, alpha=0.3)

            # 添加对角线 (理想预测线)
            x_range = axes[i].get_xlim()
            axes[i].plot(x_range, x_range, 'r--', alpha=0.5, label='Ideal Prediction')
            axes[i].legend()

        plt.tight_layout()
        plt.show()

# 12. 气候区域分析
print("\n" + "=" * 80)
print("Climate Regions Analysis")
print("=" * 80)

# 找出气候区域相关的特征
climate_features = [col for col in df.columns if 'climateregion' in col.lower()]
print(f"Climate region features count: {len(climate_features)}")

if climate_features:
    # 检查特征类型
    climate_types = {}
    for col in climate_features:
        climate_types[col] = df[col].dtype

    print("\nClimate region features types:")
    for col, dtype in list(climate_types.items())[:10]:
        print(f"  {col}: {dtype}")

    # 分析数值型气候区域特征
    numeric_climate_features = [col for col in climate_features if col in numeric_cols]
    print(f"\nNumerical climate region features: {len(numeric_climate_features)}")

    if numeric_climate_features:
        # 找出最常见的几个气候区域
        climate_data = {}
        for col in numeric_climate_features:
            # 确保是数值型才能求和
            if pd.api.types.is_numeric_dtype(df[col]):
                total = df[col].sum()
                if total > 0:
                    climate_data[col] = total

        if climate_data:
            climate_series = pd.Series(climate_data).sort_values(ascending=False)

            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            # 气候区域分布
            top_climates = climate_series.head(10)
            axes[0].barh(range(len(top_climates)), top_climates.values)
            axes[0].set_yticks(range(len(top_climates)))
            # 简化列名显示
            labels = [col.split('__')[-1] if '__' in col else col[-10:] for col in top_climates.index]
            axes[0].set_yticklabels(labels)
            axes[0].set_xlabel('Sample Count')
            axes[0].set_title('Most Common Climate Regions')
            axes[0].grid(True, alpha=0.3)

            # 气候区域与温度的关系
            if target_col in df.columns:
                climate_temps = {}
                for col in climate_series.index[:5]:  # 前5个气候区域
                    # 对于数值型特征，我们假设1表示属于该气候区域
                    if pd.api.types.is_numeric_dtype(df[col]):
                        mask = df[col] == 1
                        if mask.any():
                            region_name = col.split('__')[-1] if '__' in col else col[-10:]
                            climate_temps[region_name] = df.loc[mask, target_col].mean()

                if climate_temps:
                    axes[1].bar(range(len(climate_temps)), list(climate_temps.values()), color='orange')
                    axes[1].set_xticks(range(len(climate_temps)))
                    axes[1].set_xticklabels(list(climate_temps.keys()), rotation=45, ha='right')
                    axes[1].set_ylabel('Average Temperature (°C)')
                    axes[1].set_title('Average Temperature by Climate Region')
                    axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

# 13. 异常值检测
print("\n" + "=" * 80)
print("Outliers Detection")
print("=" * 80)

# 对关键数值特征进行异常值检测
key_features_for_outliers = [target_col, 'lat', 'lon'] + nmme_features[:5]  # 目标变量和几个关键特征

# 只保留实际存在的特征
key_features_for_outliers = [f for f in key_features_for_outliers if f in df.columns and f in numeric_cols]

outliers_info = {}

for col in key_features_for_outliers:
    if df[col].notna().sum() > 0:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        if IQR > 0:  # 避免除零
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_percentage = (len(outliers) / len(df)) * 100

            if outlier_percentage > 0:
                outliers_info[col] = {
                    'Outliers Count': len(outliers),
                    'Outliers Percentage': outlier_percentage,
                    'Min Value': df[col].min(),
                    'Max Value': df[col].max(),
                    'Lower Bound': lower_bound,
                    'Upper Bound': upper_bound,
                    'Q1': Q1,
                    'Q3': Q3
                }

if outliers_info:
    outliers_df = pd.DataFrame(outliers_info).T
    print("Outliers analysis:")
    print(outliers_df)

    # 可视化异常值
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (col, info) in enumerate(list(outliers_info.items())[:6]):
        axes[i].boxplot(df[col].dropna())
        axes[i].set_title(f'{col[:15]}...\nOutliers: {info["Outliers Percentage"]:.1f}%')
        axes[i].set_ylabel('Value')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
else:
    print("No significant outliers detected in key features")

# 14. 数据质量总结
print("\n" + "=" * 80)
print("Data Quality Summary")
print("=" * 80)

print(f"1. Dataset size: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"2. Time range: {df['startdate'].min().date()} to {df['startdate'].max().date()}")
print(f"3. Target variable '{target_col}':")
if target_col in df.columns:
    print(f"   - Mean: {df[target_col].mean():.2f} °C")
    print(f"   - Std Dev: {df[target_col].std():.2f} °C")
    print(f"   - Range: {df[target_col].min():.2f} to {df[target_col].max():.2f} °C")
    print(
        f"   - Missing values: {df[target_col].isnull().sum()} ({df[target_col].isnull().sum() / len(df) * 100:.1f}%)")

missing_cols = missing_df[missing_df['Missing Percentage'] > 0]
print(f"4. Missing values:")
print(f"   - Features with missing values: {len(missing_cols)}")
if len(missing_cols) > 0:
    print(f"   - Most missing values: {missing_cols.index[0]} ({missing_cols.iloc[0]['Missing Percentage']:.1f}%)")

print(f"5. Feature types:")
print(f"   - Numerical features: {len(numeric_cols)}")
print(f"   - Categorical features: {len(categorical_cols)}")
print(f"   - Date feature: 1 (startdate)")

print(f"6. Geographic information:")
if 'lat' in df.columns and 'lon' in df.columns:
    print(f"   - Latitude range: {df['lat'].min():.2f} to {df['lat'].max():.2f}")
    print(f"   - Longitude range: {df['lon'].min():.2f} to {df['lon'].max():.2f}")
    print(f"   - Unique locations: {df[['lat', 'lon']].drop_duplicates().shape[0]}")

print(f"7. Seasonal patterns:")
if 'month' in df.columns and target_col in df.columns:
    monthly_means = df.groupby('month')[target_col].mean()
    coldest_month = monthly_means.idxmin()
    warmest_month = monthly_means.idxmax()
    print(f"   - Coldest month: {coldest_month} ({monthly_means[coldest_month]:.1f}°C)")
    print(f"   - Warmest month: {warmest_month} ({monthly_means[warmest_month]:.1f}°C)")

print(f"8. Prediction features:")
print(f"   - NMME model features: {len(nmme_features)}")
if climate_features:
    print(f"   - Climate region features: {len(climate_features)}")
    print(
        f"   - Numerical climate region features: {len(numeric_climate_features) if 'numeric_climate_features' in locals() else 0}")

print(f"9. Outliers:")
if outliers_info:
    avg_outlier_percentage = np.mean([info['Outliers Percentage'] for info in outliers_info.values()])
    print(f"   - Average outlier percentage: {avg_outlier_percentage:.1f}%")
else:
    print(f"   - No significant outliers found in key features")



# 15. 保存EDA结果
print("\n" + "=" * 80)
print("Saving EDA Results")
print("=" * 80)

# 保存关键统计信息
eda_summary = {
    'Dataset Shape': df.shape,
    'Time Range': f"{df['startdate'].min().date()} to {df['startdate'].max().date()}",
    'Target Variable Statistics': df[target_col].describe().to_dict() if target_col in df.columns else {},
    'Top 5 Features with Most Missing Values': missing_df.head(5).to_dict(),
    'Top 10 Features Correlated with Target': corr_df.head(10).to_dict() if 'corr_df' in locals() else {},
}

import json

with open('C:/Users/DELL/PycharmProjects/pythonProject/529/group_project/eda_summary.json', 'w') as f:
    json.dump(eda_summary, f, indent=2, default=str)

print("EDA summary saved to C:/Users/DELL/PycharmProjects/pythonProject/529/group_project/eda_summary.json")

print("\nEDA analysis completed!")