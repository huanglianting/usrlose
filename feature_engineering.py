from sklearn.feature_selection import chi2, mutual_info_classif
from scipy.stats import chi2_contingency
import pandas as pd
import numpy as np


def merge_data(time_features_path, static_features_path, label_path):
    """
    根据用户id合并时间特征、静态特征和标签表并保存结果
    参数:
        time_features_path: 时间特征表路径
        static_features_path: 静态特征表路径
        label_path: 用户标签表路径
    返回:
        合并后的DataFrame
    """
    # 读取标签表
    label_columns = ['user_id', 'label']
    label_df = pd.read_csv(label_path, header=None, names=label_columns)

    # 读取时间特征表
    time_features_df = pd.read_csv(time_features_path)

    # 读取静态特征表
    static_features_df = pd.read_csv(static_features_path)

    print(f"标签表维度: {label_df.shape}")
    print(f"时间特征表维度: {time_features_df.shape}")
    print(f"静态特征表维度: {static_features_df.shape}")

    # 合并数据 (基于user_id)，确保每个用户都有标签
    merged_df = label_df.merge(
        time_features_df,
        on='user_id',
        how='left',
    ).merge(
        static_features_df,
        on='user_id',
        how='left',
    )

    print(f"合并完成: 总记录数 {len(merged_df)}")
    print("数据列预览:", merged_df.columns.tolist())

    # 处理缺失值
    merged_df = handle_missing_values(merged_df)

    # 保存处理后的数据
    merged_df.to_csv("./data/user_merged.csv", index=False)
    print(f"处理后的合并数据已保存至: ./data/user_merged.csv")

    return merged_df


def handle_missing_values(df, save_path=None):
    """
    处理缺失值并可选保存结果
    参数:
        df: 原始DataFrame
        save_path: 保存路径，None表示不保存
    返回:
        处理后的DataFrame
    """
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # 数值型用中位数填充
            median = df[col].median()
            df[col].fillna(median, inplace=True)
        else:
            # 类别型用众数填充，效果不好的话可以改成 'Missing'
            mode_series = df[col].mode()
            if len(mode_series) > 0:
                # 如果存在众数，则使用众数填充
                mode_value = mode_series[0]
                df[col].fillna(mode_value, inplace=True)
            else:
                # 如果没有众数（所有值都是NaN），则用 'Missing' 填充
                df[col].fillna('Missing', inplace=True)
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"处理后的数据已保存至: {save_path}")

    return df


def build_time_features(time_path):
    """
    从时间行为表构建完整的活跃度和时间偏好特征
    参数:
        time_path: 时间行为表路径
    返回:
        包含所有时间特征的DataFrame (每个用户一行)
    """
    # 读取时间行为表
    time_columns = [
        'user_id', 'date', 'a_page', 'other_page', 'b_page', 'b_promotion', 'c_active',
        'd_related_x', 'd_related_y', 'd_related_u', 'e_related_x', 'e_related_y', 'e_related_u', 'e_related_v',
        'd_related_v',
        'f_page', 'f_page_m', 'f_page_n', 'h_used',
        'rest', 'job', 'night', 'weekend', 'weekday'
    ]
    # 读取原始数据
    time_df = pd.read_csv(time_path, header=None, names=time_columns)
    # 先处理缺失值
    processed_time_df = handle_missing_values(time_df)
    # 按用户和日期去重 (避免同一天多条记录)
    daily_df = processed_time_df.drop_duplicates(['user_id', 'date'])
    # 按用户分组计算
    grouped = daily_df.groupby('user_id')
    # 创建特征DataFrame
    features_df = pd.DataFrame()
    features_df['user_id'] = daily_df['user_id'].unique()

    # 1. 活跃度特征
    # (1) 各服务活跃率
    service_columns = ['a_page', 'other_page', 'b_page', 'b_promotion', 'c_active',
                       'd_related_x', 'd_related_y', 'd_related_u', 'd_related_v',
                       'e_related_x', 'e_related_y', 'e_related_u', 'e_related_v',
                       'f_page', 'f_page_m', 'f_page_n', 'h_used']
    for col in service_columns:
        features_df[f'{col}_active_ratio'] = grouped[col].apply(
            lambda x: (x == 1).sum() / len(x)
        ).values

    # (2) 关联服务活跃度
    # d_related系列
    d_related_cols = ['d_related_x', 'd_related_y',
                      'd_related_u', 'd_related_v']
    features_df['d_related_active_ratio'] = grouped[d_related_cols].apply(
        lambda x: (x == 1).any(axis=1).sum() / len(x)
    ).values
    # e_related系列
    e_related_cols = ['e_related_x', 'e_related_y',
                      'e_related_u', 'e_related_v']
    features_df['e_related_active_ratio'] = grouped[e_related_cols].apply(
        lambda x: (x == 1).any(axis=1).sum() / len(x)
    ).values

    # 2. 时间偏好特征
    # (1) 夜间活跃占比
    features_df['night_ratio'] = grouped['night'].apply(
        lambda x: (x == 1).sum() / len(x)
    ).values
    # (2) 休息时间活跃度
    features_df['rest_ratio'] = grouped['rest'].apply(
        lambda x: (x == 1).sum() / len(x)
    ).values
    # (3) 工作时间活跃度
    features_df['job_ratio'] = grouped['job'].apply(
        lambda x: (x == 1).sum() / len(x)
    ).values
    # (4) 工作日和休息日活跃差
    features_df['weekday_weekend_diff'] = grouped.apply(
        lambda x: (x['weekday'] == 1).sum() - (x['weekend'] == 1).sum()
    ).values

    return features_df

    """
    构建时间偏好行为特征(返回单独的特征DataFrame)
    参数:
        df: 包含原始时间数据的DataFrame
    返回:
        添加了新特征的DataFrame
    """
    # 按user_id分组计算各项特征
    grouped = df.groupby('user_id')

    # 创建特征DataFrame
    features_df = pd.DataFrame()
    features_df['user_id'] = df['user_id'].unique()

    # (1) 夜间活跃占比
    features_df['night_ratio'] = grouped['night'].apply(
        lambda x: x.sum() / len(x)
    ).values

    # (2) 休息时间活跃度
    features_df['rest_ratio'] = grouped['rest'].apply(
        lambda x: x.sum() / len(x)
    ).values

    # (3) 工作时间活跃度
    features_df['job_ratio'] = grouped['job'].apply(
        lambda x: x.sum() / len(x)
    ).values

    # (4) 工作日和休息日活跃差
    features_df['weekday_weekend_diff'] = grouped.apply(
        lambda x: (x['weekday'] == 1).sum() - (x['weekend'] == 1).sum()
    ).values

    return features_df.drop_duplicates()


def build_static_features(static_path):
    """
    构建静态属性特征并扩展原静态表
    参数:
        static_path: 静态属性表路径
    返回:
        包含新增特征的静态属性DataFrame
    """
    # 读取静态属性表
    static_columns = [
        'user_id', 'p_eff_days', 'p_exp_days', 'p_pur_status',
        'p_fee_type', 'p_fee_cycle', 'p_inc', 'p_capacity', 'p_pur_type',
        'p_source_code_1', 'p_source_level', 'p_source_code_2', 'p_package_type',
        'p_hard_type', 'p_pard_fomt', 'q_used', 'q_type', 'q_scale',
        'q_capacity', 'q_open_days', 'q_exp_days', 'q_open_months', 'q_exp_months',
        'q_status', 'a_pur', 'a_pur_type', 'a_state', 'a_source_type',
        'a_bind', 'e_bind', 'd_bind', 'b_bind'
    ]
    static_df = pd.read_csv(static_path, header=None, names=static_columns)
    # 先处理缺失值
    static_df = handle_missing_values(static_df)

    # 1. 服务有效期特征
    # (1) 服务总时长
    static_df['p_total_duration'] = static_df['p_eff_days'] + \
        static_df['p_exp_days']
    static_df['q_total_duration'] = static_df['q_open_days'] + static_df['q_exp_days'] + \
        30 * static_df['q_open_months'] + 30 * static_df['q_exp_months']

    # (2) 剩余有效期占比
    # 如果p_total_duration不为0，则计算实际占比
    static_df['p_remaining_ratio'] = 0  # 初始化
    p_valid_mask = static_df['p_total_duration'] != 0
    static_df.loc[p_valid_mask, 'p_remaining_ratio'] = (
        static_df['p_exp_days'] / static_df['p_total_duration']
    )[p_valid_mask]
    # 如果q_total_duration不为0，则计算实际占比
    static_df['q_remaining_ratio'] = 0  # 初始化
    q_valid_mask = static_df['q_total_duration'] != 0
    static_df.loc[q_valid_mask, 'q_remaining_ratio'] = (
        (static_df['q_exp_days'] + 30 * static_df['q_exp_months']) /
        static_df['q_total_duration']
    )[q_valid_mask]

    # (3) 服务是否已过期
    static_df['p_is_expired'] = (static_df['p_exp_days'] <= 0).astype(int)
    static_df['q_is_expired'] = ((static_df['q_exp_days'] <= 0) & (
        static_df['q_exp_months'] <= 0)).astype(int)

    # 2. 服务活跃状态
    static_df['p_used'] = ((static_df['p_pur_status'] == 1) | (
        static_df['p_pur_status'] == 2)).astype(int)
    static_df['active_service_count'] = static_df['p_used'] + \
        static_df['a_pur'] + static_df['q_used']

    # 3. 服务覆盖度
    static_df['total_service_count'] = (
        static_df['p_used'] + static_df['q_used'] + static_df['a_pur'] +
        static_df['a_bind'] + static_df['e_bind'] +
        static_df['d_bind'] + static_df['b_bind']
    )
    static_df['bind_service_count'] = (
        static_df['a_bind'] + static_df['e_bind'] +
        static_df['d_bind'] + static_df['b_bind']
    )

    # 处理可能的无穷值或NaN
    static_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    static_df.fillna(0, inplace=True)  # 将无效值替换为0

    return static_df


def analyze_feature_types(df):
    """
    分析DataFrame中的特征类型
    参数:
        df: 要分析的DataFrame
    """
    numeric_features = []
    string_features = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)
        else:
            string_features.append(col)

    print(f"数值型特征数量: {len(numeric_features)}")
    print(f"字符串型特征数量: {len(string_features)}")
    print(f"总特征数量: {len(df.columns)}")

    print("\n数值型特征列表:")
    for i, feature in enumerate(numeric_features, 1):
        print(f"{i:2d}. {feature}")

    print("\n字符串型特征列表:")
    for i, feature in enumerate(string_features, 1):
        print(f"{i:2d}. {feature}")

    return numeric_features, string_features


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
