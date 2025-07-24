import pandas as pd
import numpy as np


def merge_user_data(time_path, static_path, label_path):
    """
    根据用户id合并三个表格并保存结果
    参数:
        time_path: 时间行为表路径
        static_path: 静态属性表路径
        label_path: 用户标签表路径
    返回:
        合并后的DataFrame
    """
    # ===== 时间行为表字段（共24列）=====
    time_columns = [
        'user_id', 'date', 'a_page', 'other_page', 'b_page', 'b_promotion', 'c_active',
        'd_related_x', 'd_related_y', 'd_related_u', 'e_related_x', 'e_related_y', 'e_related_u', 'e_related_v',
        'd_related_v',
        'f_page', 'f_page_m', 'f_page_n', 'h_used',
        'rest', 'job', 'night', 'weekend', 'weekday'
    ]
    # ===== 静态属性表字段（共32列）=====
    static_columns = [
        'user_id', 'p_eff_days', 'p_exp_days', 'p_pur_status',
        'p_fee_type', 'p_fee_cycle', 'p_inc', 'p_capacity', 'p_pur_type',
        'p_source_code_1', 'p_source_level', 'p_source_code_2', 'p_package_type',
        'p_hard_type', 'p_pard_fomt', 'q_used', 'q_type', 'q_scale',
        'q_capacity', 'q_open_days', 'q_exp_days', 'q_open_months', 'q_exp_months',
        'q_status', 'a_pur', 'a_pur_type', 'a_state', 'a_source_type',
        'a_bind', 'e_bind', 'd_bind', 'b_bind'
    ]
    # ===== 用户流失标签表字段（共2列）=====
    label_columns = ['user_id', 'label']

    # 读取无列名文件
    time_df = pd.read_csv(
        time_path,
        header=None,  # 文件无列名标题行
        names=time_columns  # 使用自定义列名
    )
    static_df = pd.read_csv(
        static_path,
        header=None,
        names=static_columns
    )
    label_df = pd.read_csv(
        label_path,
        header=None,
        names=label_columns
    )
    print(f"时间行为表维度: {time_df.shape}")
    print(f"静态属性表维度: {static_df.shape}")
    print(f"用户流失标签表维度: {label_df.shape}")

    # 合并数据 (基于user_id)
    merged_df = label_df.merge(
        time_df,
        on='user_id',
        how='left',
    ).merge(
        static_df,
        on='user_id',
        how='left',
    )
    print(f"合并完成: 总记录数 {len(merged_df)}")
    print("数据列预览:", merged_df.columns.tolist())
    merged_df.to_csv("./data/user_merged.csv", index=False)

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
            # 类别型用 'Missing' 填充，效果不好的话可以改成众数
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


if __name__ == "__main__":
    """
    result_df = merge_user_data(
        time_path="./data/user_time_base.csv",
        static_path="./data/user_static_base.csv",
        label_path='./data/user_label.csv'
    )
    result_df = pd.read_csv("./data/user_merged.csv")
    print("\n数据预览:")
    print(result_df.head(3))

    # 构建时间特征
    time_features_df = build_time_features("./data/user_time_base.csv")
    time_features_df.to_csv("./data/time_features.csv", index=False)
    print(time_features_df.head())
    """

    # 构建静态特征
    static_features_df = build_static_features("./data/user_static_base.csv")
    static_features_df.to_csv("./data/static_features.csv", index=False)
    print("\n静态特征预览:")
    print(static_features_df[['user_id', 'p_total_duration', 'q_total_duration',
                             'p_remaining_ratio', 'q_remaining_ratio']].head())
