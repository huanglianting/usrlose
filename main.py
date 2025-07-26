import pandas as pd
import numpy as np
# from feature_engineering import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import os


# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置统一的保存路径
save_path = './result'
os.makedirs(save_path, exist_ok=True)  # 创建文件夹（如果不存在）

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

    # 进行特征选择分析
    selected_features, filtered_data = feature_selection_analysis(
        "./data/user_merged.csv",
        alpha=0.1,      # p < 0.1的特征保留
        mi_threshold=0.01  # MI > 0.01的特征保留
    )

    # 保存筛选后的数据
    filtered_data.to_csv("./data/filtered_user_data.csv", index=False)
    print(f"\n筛选后的数据已保存至: ./data/filtered_user_data.csv")
    """

    feature_df = pd.read_csv("./data/filtered_user_data.csv")

    # 按照user_id顺序划分训练集和测试集
    train_df = feature_df[feature_df['user_id'] <= 640000]
    test_df = feature_df[feature_df['user_id'] > 640000]
    print(f"训练集样本数: {len(train_df)}")
    print(f"测试集样本数: {len(test_df)}")

    # 使用LightGBM建立预测模型
    print("\n开始使用LightGBM建立用户流失预测模型...")

    # 准备数据
    X_train = train_df.drop(['label', 'user_id'], axis=1)
    y_train = train_df['label']
    X_test = test_df.drop(['label', 'user_id'], axis=1)
    y_test = test_df['label']

    print(f"训练数据形状: {X_train.shape}")
    print(f"测试数据形状: {X_test.shape}")
    print(f"训练集标签分布:\n{y_train.value_counts()}")
    print(f"测试集标签分布:\n{y_test.value_counts()}")

    # 计算类别不平衡权重
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    print(
        f"正负样本比例 (0:1): {neg_count}:{pos_count} = 1:{pos_count/neg_count:.2f}")
    print(f"scale_pos_weight 设置为: {scale_pos_weight:.2f}")

    # 创建LightGBM数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # 设置参数
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,           # 回到原始设置
        'learning_rate': 0.05,      # 保持适中的学习率
        'feature_fraction': 0.9,    # 回到原始设置
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 20,     # 添加适度的约束
        'scale_pos_weight': scale_pos_weight,  # 处理类别不平衡问题
        'verbose': 0,
        'random_state': 42
    }

    # 训练模型
    print("训练LightGBM模型...")
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'eval'],
        num_boost_round=1500,       # 增加训练轮数
        callbacks=[lgb.early_stopping(
            stopping_rounds=100), lgb.log_evaluation(100)]  # 增加早停轮数
    )

    # 预测
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # 评估模型
    print("\n=== 模型评估结果 ===")
    print(f"AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Churn', 'Churn'],
                yticklabels=['Not Churn', 'Churn'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    # ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2,
             linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'roc_curve.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    print("\n=== 特征重要性 (前10个) ===")
    print(feature_importance.head(10))

    # 绘制特征重要性图
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(15)
    sns.barplot(data=top_features, y='feature',
                x='importance', palette='viridis')
    plt.title('LightGBM Feature Importance (Top 15)')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'feature_importance.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    # 保存模型
    model.save_model(os.path.join(save_path, 'lgbm_churn_model.txt'))
    print(f"\n模型已保存至: {os.path.join(save_path, 'lgbm_churn_model.txt')}")

    # 保存特征重要性
    feature_importance.to_csv(os.path.join(
        save_path, 'feature_importance.csv'), index=False)
    print(f"特征重要性已保存至: {os.path.join(save_path, 'feature_importance.csv')}")
