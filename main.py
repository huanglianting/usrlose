from sklearn.feature_selection import chi2, mutual_info_classif
from scipy.stats import chi2_contingency
import pandas as pd
import numpy as np
from feature_engineering import *


def feature_selection_analysis(data_path, target_col='label', alpha=0.05):
    """
    对所有特征进行卡方检验和互信息分析，筛选与目标变量相关的特征
    参数:
        data_path: 数据文件路径
        target_col: 目标变量列名，默认为'label'
        alpha: 显著性水平，默认为0.05
    返回:
        通过筛选的特征列表和详细结果
    """
    # 读取数据
    df = pd.read_csv(data_path)

    # 获取所有特征列（排除目标变量和user_id）
    all_features = [col for col in df.columns if col not in [
        target_col, 'user_id']]

    # 存储结果
    results = []
    selected_features = []

    print(f"特征选择分析 (显著性水平 α = {alpha})")
    print("=" * 60)

    for feature in all_features:
        # 确保特征是数值型
        df[feature] = pd.to_numeric(df[feature], errors='coerce')

        # 检查缺失值
        missing_count = df[feature].isna().sum()
        if missing_count > 0:
            print(f"警告: 特征 '{feature}' 存在 {missing_count} 个缺失值")
            clean_data = df[[feature, target_col]].dropna()
        else:
            clean_data = df[[feature, target_col]]

        if len(clean_data) == 0:
            print(f"警告: 特征 '{feature}' 没有有效数据")
            continue

        # 检查特征是否只有一个唯一值
        if len(clean_data[feature].unique()) <= 1:
            print(f"警告: 特征 '{feature}' 只有一个唯一值，跳过")
            continue

        # 卡方检验
        try:
            # 创建列联表
            contingency_table = pd.crosstab(
                clean_data[feature], clean_data[target_col])

            # 如果列联表维度太小，跳过
            if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                print(f"特征 '{feature}': 列联表维度不足，跳过")
                continue

            chi2_stat, p_value, dof, expected = chi2_contingency(
                contingency_table)

            # 互信息
            mi_score = mutual_info_classif(
                clean_data[feature].values.reshape(-1, 1),
                clean_data[target_col],
                random_state=42
            )[0]

            # 判断是否显著
            is_significant = p_value < alpha

            results.append({
                'feature': feature,
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'mi_score': mi_score,
                'significant': is_significant
            })

            if is_significant:
                selected_features.append(feature)
                print(
                    f"✓ {feature:30s} | χ²={chi2_stat:8.2f} | p={p_value:.4f} | MI={mi_score:.4f}")
            else:
                print(
                    f"✗ {feature:30s} | χ²={chi2_stat:8.2f} | p={p_value:.4f} | MI={mi_score:.4f}")

        except Exception as e:
            print(f"特征 '{feature}' 计算出错: {str(e)}")
            continue

    # 排序结果
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('mi_score', ascending=False)

        print(f"\n通过筛选的特征数量: {len(selected_features)}")
        print(f"总分析特征数量: {len(all_features)}")
        print(
            f"筛选比例: {len(selected_features)/len(all_features)*100:.1f}%")

        print("\n通过筛选的特征 (按互信息得分排序):")
        significant_results = results_df[results_df['significant']]
        for idx, row in significant_results.iterrows():
            print(
                f"  {row['feature']:30s} | MI={row['mi_score']:.4f} | p={row['p_value']:.4f}")

    return selected_features, results_df


if __name__ == "__main__":
    """
    # 构建时间特征
    time_features_df = build_time_features("./data/user_time_base.csv")
    time_features_df.to_csv("./data/time_features.csv", index=False)

    # 构建静态特征
    static_features_df = build_static_features("./data/user_static_base.csv")
    static_features_df.to_csv("./data/static_features.csv", index=False)

    # 合并最终数据
    feature_df = merge_data(
        time_features_path="./data/time_features.csv",
        static_features_path="./data/static_features.csv",
        label_path="./data/user_label.csv"
    )
    print("\n最终合并数据预览:")
    print(final_df.head(3))

    # 分析特征类型
    print("\n=== 特征类型分析 ===")
    numeric_features, string_features = analyze_feature_types(feature_df)
    # 没有字符串型特征，均为数值型特征
    """

    feature_df = pd.read_csv("./data/user_merged.csv")

    # 进行特征选择分析
    print("开始特征选择分析...")
    selected_features, analysis_results = feature_selection_analysis(
        "./data/user_merged.csv")

    # 保存分析结果
    analysis_results.to_csv(
        "./data/feature_selection_results.csv", index=False)
    print(f"\n特征选择结果已保存至: ./data/feature_selection_results.csv")
