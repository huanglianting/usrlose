import pandas as pd
import numpy as np
from feature_engineering import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns


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

    # 使用LightGBM建立预测模型
    print("\n开始使用LightGBM建立用户流失预测模型...")

    # 准备数据
    X = feature_df.drop(['label', 'user_id'], axis=1)
    y = feature_df['label']

    print(f"训练数据形状: {X.shape}")
    print(f"标签分布:\n{y.value_counts()}")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 创建LightGBM数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # 设置参数
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
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
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(
            stopping_rounds=50), lgb.log_evaluation(100)]
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
                xticklabels=['未流失', '流失'],
                yticklabels=['未流失', '流失'])
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig('./data/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC曲线 (AUC = {roc_auc_score(y_test, y_pred_proba):.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (False Positive Rate)')
    plt.ylabel('真正率 (True Positive Rate)')
    plt.title('ROC曲线')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./data/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    print("\n=== 特征重要性 (前10个) ===")
    print(feature_importance.head(10))

    # 绘制特征重要性图
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(15)
    sns.barplot(data=top_features, y='feature',
                x='importance', palette='viridis')
    plt.title('LightGBM特征重要性 (前15个)')
    plt.xlabel('重要性得分')
    plt.tight_layout()
    plt.savefig('./data/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 保存模型
    model.save_model('./data/lgbm_churn_model.txt')
    print(f"\n模型已保存至: ./data/lgbm_churn_model.txt")

    # 保存特征重要性
    feature_importance.to_csv('./data/feature_importance.csv', index=False)
    print(f"特征重要性已保存至: ./data/feature_importance.csv")
