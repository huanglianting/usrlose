from sklearn.feature_selection import chi2, mutual_info_classif
from scipy.stats import chi2_contingency
import pandas as pd
import numpy as np
from feature_engineering import *


def feature_selection_analysis(data_path, target_col='label', alpha=0.1, mi_threshold=0.01):
    """
    对所有特征进行卡方检验和互信息分析，筛选与目标变量相关的特征
    参数:
        data_path: 数据文件路径
        target_col: 目标变量列名，默认为'label'
        alpha: 显著性水平阈值，默认为0.1 (p值大于此值的特征将被剔除)
        mi_threshold: 互信息阈值，默认为0.01 (MI小于此值的特征将被剔除)
    返回:
        筛选后的特征列表和筛选后的数据
    """
    # 读取数据
    df = pd.read_csv(data_path)

    # 获取所有特征列（排除目标变量和user_id）
    all_features = [col for col in df.columns if col not in [
        target_col, 'user_id']]

    # 存储结果
    results = []
    selected_features = []

    print(f"特征选择分析 (显著性水平 α = {alpha}, MI阈值 = {mi_threshold})")
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

        # 根据条件进一步筛选特征
        # 1. p值 < alpha (显著性检验通过)
        # 2. MI > mi_threshold (互信息足够大)
        final_selected = results_df[
            (results_df['p_value'] < alpha) &
            (results_df['mi_score'] > mi_threshold)
        ]['feature'].tolist()

        print(f"\n最终筛选结果:")
        print(f"原始特征数: {len(all_features)}")
        print(f"显著性筛选后特征数: {len(selected_features)}")
        print(f"最终筛选后特征数: {len(final_selected)}")
        print(f"最终保留特征:")
        for feature in final_selected:
            feature_info = results_df[results_df['feature'] == feature].iloc[0]
            print(
                f"  {feature:30s} | MI={feature_info['mi_score']:.4f} | p={feature_info['p_value']:.4f}")

        # 创建筛选后的数据集（保留所有原始数据，只筛选特征列）
        selected_columns = [target_col, 'user_id'] + final_selected
        filtered_df = df[selected_columns].copy()

        return final_selected, filtered_df


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
    selected_features, filtered_data = feature_selection_analysis(
        "./data/user_merged.csv",
        alpha=0.1,      # p < 0.1的特征保留
        mi_threshold=0.01  # MI > 0.01的特征保留
    )

    # 保存筛选后的数据
    filtered_data.to_csv("./data/filtered_user_data.csv", index=False)
    print(f"\n筛选后的数据已保存至: ./data/filtered_user_data.csv")
