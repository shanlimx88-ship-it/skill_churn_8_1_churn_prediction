#!/usr/bin/env python3
"""
Skill 8.1: 流失预警模型 - 深入指标特征版

基于早期行为预测用户流失风险
包含：使用模式、活跃趋势、功能深度、使用广度、时间偏好、参与质量、社交信号、付费信号
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# 0. 特征定义表（中英文说明）
# ============================================================

FEATURE_DEFINITIONS = {
    'active_days_14d': {
        '中文': '前14天活跃天数',
        'English': 'Active days in first 14 days'
    },
    'sessions_14d': {
        '中文': '前14天会话总数',
        'English': 'Total sessions in first 14 days'
    },
    'avg_depth_14d': {
        '中文': '平均会话深度（轮次）',
        'English': 'Average session depth (turns)'
    },
    'has_hva_14d': {
        '中文': '是否完成高价值行为',
        'English': 'Completed High-Value Action'
    },
    'adoption_rate_14d': {
        '中文': '采纳率（AI输出直接使用比例）',
        'English': 'Adoption rate (direct AI output usage)'
    },
    'feature_count_14d': {
        '中文': '使用的高级功能数量',
        'English': 'Number of advanced features used'
    },
    'day1_return': {
        '中文': '第2天是否回访',
        'English': 'Returned on day 2'
    },
    'day7_return': {
        '中文': '第7天是否回访',
        'English': 'Returned on day 7'
    },
    'day14_return': {
        '中文': '第14天是否回访',
        'English': 'Returned on day 14'
    },
    'num_scenarios': {
        '中文': '使用场景数量',
        'English': 'Number of usage scenarios'
    },
    'interval_std': {
        '中文': '使用间隔标准差（使用规律性）',
        'English': 'Standard deviation of usage intervals (regularity)'
    },
    'activity_trend': {
        '中文': '活跃度变化趋势（后7天-前7天）',
        'English': 'Activity trend (day 8-14 minus day 1-7)'
    },
    'first_hva_days': {
        '中文': '首次高价值行为出现天数',
        'English': 'Days to first High-Value Action'
    },
    'scenario_diversity': {
        '中文': '场景多样性指数（0-1，越高越多样）',
        'English': 'Scenario diversity index (0-1, higher = more diverse)'
    },
    'weekday_ratio': {
        '中文': '工作日/周末使用强度比',
        'English': 'Weekday vs weekend usage ratio'
    },
    'avg_wait_time': {
        '中文': '平均响应等待时间（秒）',
        'English': 'Average response wait time (seconds)'
    },
    'has_shared': {
        '中文': '是否有分享/导出行为',
        'English': 'Has shared or exported content'
    },
    'viewed_pricing': {
        '中文': '是否查看过定价页面',
        'English': 'Viewed pricing page'
    }
}


# ============================================================
# 1. 数据生成
# ============================================================

def generate_sample_data(n_users: int = 5000):
    """生成用户行为数据（含深入特征）"""
    np.random.seed(42)
    start_date = datetime(2024, 1, 1)

    users = []

    for i in range(n_users):
        user_id = f"user_{i:05d}"
        signup_date = start_date + timedelta(days=np.random.randint(0, 60))

        # ===== 基础行为 =====
        active_days_14d = np.random.choice(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            p=[0.02, 0.03, 0.05, 0.06, 0.08, 0.10, 0.12, 0.12, 0.11, 0.10, 0.08, 0.06, 0.04, 0.03]
        )
        sessions_14d = active_days_14d + np.random.poisson(2)
        avg_depth_14d = round(1.0 + active_days_14d * 0.3 + np.random.gamma(1, 0.5), 1)
        adoption_rate_14d = round(0.1 + active_days_14d * 0.04 + np.random.beta(1, 2) * 0.2, 2)
        adoption_rate_14d = min(0.95, adoption_rate_14d)
        feature_count_14d = min(4, int(active_days_14d / 2 + np.random.poisson(0.5)))
        day1_return = 1 if np.random.random() < (0.1 + active_days_14d * 0.03) else 0
        day7_return = 1 if np.random.random() < (0.05 + active_days_14d * 0.03) else 0
        day14_return = 1 if np.random.random() < (0.03 + active_days_14d * 0.03) else 0

        # ===== HVA =====
        hva_prob = 0.02 + active_days_14d * 0.04 + sessions_14d * 0.005 + avg_depth_14d * 0.02
        hva_prob = min(0.90, hva_prob)
        has_hva_14d = 1 if np.random.random() < hva_prob else 0

        num_scenarios = min(4, 1 + int(active_days_14d / 3))

        # ===== 深入特征 =====
        if active_days_14d > 1:
            intervals = np.random.exponential(2, active_days_14d - 1) + 0.5
            interval_std = round(np.std(intervals), 2)
        else:
            interval_std = 0.0

        if active_days_14d >= 7:
            first_week_days = min(7, active_days_14d)
            second_week_days = max(0, active_days_14d - 7)
            activity_trend = round((second_week_days / 7 - first_week_days / 7) * 100, 1)
        else:
            activity_trend = 0.0

        if has_hva_14d == 1:
            first_hva_days = np.random.randint(1, 8)
        else:
            first_hva_days = 99

        scenario_diversity = round(np.random.beta(1 + num_scenarios, 5 - num_scenarios), 2)
        weekday_ratio = round(np.random.uniform(0.3, 1.5), 2) if active_days_14d > 3 else 0.0
        avg_wait_time = round(np.random.gamma(2, 2), 1) if active_days_14d > 0 else 0.0
        has_shared = 1 if np.random.random() < (0.05 + active_days_14d * 0.02) else 0
        viewed_pricing = 1 if np.random.random() < (0.02 + active_days_14d * 0.01 + has_hva_14d * 0.1) else 0

        # ===== 流失标签 =====
        churn_score = 0
        churn_score += max(0, 5 - active_days_14d) * 2.0
        churn_score += max(0, 10 - sessions_14d) * 0.5
        churn_score += (1 - has_hva_14d) * 6
        churn_score += (1 - day7_return) * 4
        churn_score += (1 - day14_return) * 3
        churn_score += (1 - adoption_rate_14d) * 8
        churn_score += max(0, 2 - feature_count_14d) * 2
        churn_score += interval_std * 0.3
        churn_score += max(0, -activity_trend) * 0.1
        churn_score += (1 if first_hva_days > 7 else 0) * 3
        churn_score += (1 - scenario_diversity) * 5
        churn_score += (1 if avg_wait_time > 5 else 0) * 2
        churn_score += (1 - has_shared) * 2

        churn_prob = min(0.95, churn_score / 35)
        is_churned = 1 if np.random.random() < churn_prob else 0

        users.append({
            'user_id': user_id,
            'signup_date': signup_date.strftime('%Y-%m-%d'),
            'active_days_14d': active_days_14d,
            'sessions_14d': sessions_14d,
            'avg_depth_14d': avg_depth_14d,
            'has_hva_14d': has_hva_14d,
            'adoption_rate_14d': adoption_rate_14d,
            'feature_count_14d': feature_count_14d,
            'day1_return': day1_return,
            'day7_return': day7_return,
            'day14_return': day14_return,
            'num_scenarios': num_scenarios,
            'interval_std': interval_std,
            'activity_trend': activity_trend,
            'first_hva_days': first_hva_days,
            'scenario_diversity': scenario_diversity,
            'weekday_ratio': weekday_ratio,
            'avg_wait_time': avg_wait_time,
            'has_shared': has_shared,
            'viewed_pricing': viewed_pricing,
            'is_churned': is_churned
        })

    df = pd.DataFrame(users)
    return df


# ============================================================
# 2. 流失预测模型
# ============================================================

def build_churn_model(df):
    """构建流失预测模型"""

    feature_groups = {
        '基础活跃特征': ['active_days_14d', 'sessions_14d', 'avg_depth_14d'],
        '价值特征': ['has_hva_14d', 'adoption_rate_14d', 'first_hva_days'],
        '留存信号': ['day1_return', 'day7_return', 'day14_return'],
        '使用广度': ['num_scenarios', 'scenario_diversity', 'feature_count_14d'],
        '使用模式': ['interval_std', 'activity_trend', 'weekday_ratio'],
        '参与质量': ['avg_wait_time', 'has_shared', 'viewed_pricing']
    }

    all_features = []
    for features in feature_groups.values():
        all_features.extend(features)

    X = df[all_features].copy()
    y = df['is_churned'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=15,
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(X_train_scaled, y_train)

    y_pred = rf.predict(X_test_scaled)
    y_prob = rf.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    importance_df = pd.DataFrame({
        'feature': all_features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    # 添加中英文说明到特征重要性表
    importance_df['中文说明'] = importance_df['feature'].map(lambda x: FEATURE_DEFINITIONS.get(x, {}).get('中文', ''))
    importance_df['English'] = importance_df['feature'].map(lambda x: FEATURE_DEFINITIONS.get(x, {}).get('English', ''))

    group_importance = {}
    for group, features in feature_groups.items():
        group_imp = sum(rf.feature_importances_[all_features.index(f)] for f in features if f in all_features)
        group_importance[group] = group_imp * 100

    X_all_scaled = scaler.transform(X)
    df['churn_probability'] = rf.predict_proba(X_all_scaled)[:, 1]
    df['risk_level'] = pd.cut(
        df['churn_probability'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['低风险', '中风险', '高风险']
    )

    return {
        'model': rf,
        'scaler': scaler,
        'feature_importance': importance_df,
        'feature_groups': feature_groups,
        'group_importance': group_importance,
        'auc': auc,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'df_with_predictions': df
    }


# ============================================================
# 3. HTML 报告生成
# ============================================================

def generate_html_report(results, output_path="output/churn_report.html"):
    """生成流失预警报告"""

    # ===== 特征重要性（含中英文说明） =====
    importance_rows = ""
    max_imp = results['feature_importance']['importance'].max()
    for _, row in results['feature_importance'].iterrows():
        pct = row['importance'] * 100
        bar_width = (row['importance'] / max_imp) * 100 if max_imp > 0 else 0
        importance_rows += f"""
        <tr>
            <td><strong>{row['feature']}</strong></td>
            <td style="font-size:0.85rem;color:#555;">{row['中文说明']}</td>
            <td style="font-size:0.8rem;color:#888;">{row['English']}</td>
            <td>{pct:.1f}%</td>
            <td>
                <div style="width:100%;background:#e8eaed;border-radius:4px;height:16px;">
                    <div style="width:{bar_width}%;background:#e74c3c;border-radius:4px;height:16px;"></div>
                </div>
            </td>
        </tr>
        """

    # ===== 特征组重要性 =====
    group_rows = ""
    for group, imp in results['group_importance'].items():
        group_rows += f"""
        <tr>
            <td><strong>{group}</strong></td>
            <td>{imp:.1f}%</td>
            <td>
                <div style="width:100%;background:#e8eaed;border-radius:4px;height:16px;">
                    <div style="width:{imp}%;background:#667eea;border-radius:4px;height:16px;"></div>
                </div>
            </td>
        </tr>
        """

    # ===== 混淆矩阵 =====
    cm = results['confusion_matrix']
    cm_rows = ""
    labels = ['留存', '流失']
    for i in range(2):
        cm_rows += f"""
        <tr>
            <td><strong>{labels[i]}</strong></td>
            <td>{cm[i][0]}</td>
            <td>{cm[i][1]}</td>
            <td>{cm[i].sum()}</td>
        </tr>
        """

    # ===== 模型指标 =====
    report = results['classification_report']
    metrics_rows = f"""
    <tr><td>准确率 (Accuracy)</td><td>{report['accuracy']:.3f}</td></tr>
    <tr><td>精确率 (Precision)</td><td>{report['weighted avg']['precision']:.3f}</td></tr>
    <tr><td>召回率 (Recall)</td><td>{report['weighted avg']['recall']:.3f}</td></tr>
    <tr><td>F1 分数</td><td>{report['weighted avg']['f1-score']:.3f}</td></tr>
    <tr><td>AUC</td><td>{results['auc']:.3f}</td></tr>
    """

    # ===== 风险分布 =====
    risk_counts = results['df_with_predictions']['risk_level'].value_counts().sort_index().to_dict()
    risk_rows = ""
    risk_definitions = {
        '低风险': '流失概率 0-30% → 持续监控，正常运营',
        '中风险': '流失概率 30-60% → 积极触达，推送使用提醒',
        '高风险': '流失概率 60-100% → 立即干预，人工跟进'
    }
    risk_colors = {'低风险': '#2ecc71', '中风险': '#f39c12', '高风险': '#e74c3c'}
    for level in ['低风险', '中风险', '高风险']:
        count = risk_counts.get(level, 0)
        pct = count / len(results['df_with_predictions']) * 100
        risk_rows += f"""
        <tr>
            <td><span style="color:{risk_colors[level]};font-weight:bold;">●</span> {level}</td>
            <td>{count:,}</td>
            <td>{pct:.1f}%</td>
            <td style="font-size:0.85rem;color:#555;">{risk_definitions[level]}</td>
        </tr>
        """

    # ===== 全部特征中英文对照表 =====
    feature_table_rows = ""
    for feature, defs in FEATURE_DEFINITIONS.items():
        feature_table_rows += f"""
        <tr>
            <td><code>{feature}</code></td>
            <td>{defs['中文']}</td>
            <td>{defs['English']}</td>
        </tr>
        """

    html = f'''<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>流失预警模型 | Skill 8.1</title>
    <style>
        *{{margin:0;padding:0;box-sizing:border-box}}
        body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f0f2f5;padding:40px 20px}}
        .container{{max-width:1400px;margin:0 auto}}

        .header{{background:white;border-radius:20px;padding:30px;margin-bottom:30px;text-align:center;box-shadow:0 2px 10px rgba(0,0,0,0.08)}}
        .header h1{{color:#1a1a2e;font-size:2rem}}
        .header .subtitle{{color:#666;margin-top:8px}}

        .section{{background:white;border-radius:20px;padding:25px;margin-bottom:30px;box-shadow:0 2px 10px rgba(0,0,0,0.08)}}
        .section h2{{color:#1a1a2e;font-size:1.2rem;margin-bottom:20px;padding-bottom:10px;border-bottom:2px solid #667eea}}

        .stat-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:16px}}
        .stat-card{{background:#f8f9fa;padding:20px;border-radius:16px;text-align:center;border-top:4px solid #667eea}}
        .stat-card .label{{font-size:0.8rem;color:#888}}
        .stat-card .value{{font-size:1.3rem;font-weight:bold;color:#1a1a2e}}

        .table-wrapper{{overflow-x:auto}}
        table{{width:100%;border-collapse:collapse;font-size:0.9rem}}
        th,td{{padding:10px 14px;text-align:center;border-bottom:1px solid #eee}}
        th{{background:#f8f9fa;font-weight:600;color:#667eea}}
        tr:hover td{{background:#fafafa}}

        .metrics-definition{{
            display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;margin-top:10px;
        }}
        .def-item{{
            background:#f8f9fa;padding:12px 16px;border-radius:10px;border-left:3px solid #667eea;
        }}
        .def-item .def-name{{font-weight:600;color:#1a1a2e;display:block;margin-bottom:4px}}
        .def-item .def-desc{{font-size:0.85rem;color:#555;line-height:1.5}}

        .insight-box{{
            background:#f0f4ff;padding:16px 20px;border-radius:12px;border-left:4px solid #667eea;margin-top:10px;line-height:2
        }}

        .footer{{text-align:center;padding:20px;color:#aaa;font-size:0.8rem}}

        @media(max-width:768px){{.stat-grid,.metrics-definition{{grid-template-columns:1fr}}}}
    </style>
</head>
<body>
<div class="container">

    <div class="header">
        <h1>⚠️ 流失预警模型</h1>
        <div class="subtitle">基于18+深入行为特征预测用户流失风险</div>
    </div>

    <div class="section">
        <h2>📐 指标说明</h2>
        <div class="metrics-definition">
            <div class="def-item">
                <span class="def-name">📊 特征重要性</span>
                <span class="def-desc">反映每个行为特征对预测流失的贡献度</span>
            </div>
            <div class="def-item">
                <span class="def-name">📊 AUC</span>
                <span class="def-desc">模型区分能力，0.7-0.8 良好，>0.8 优秀</span>
            </div>
            <div class="def-item">
                <span class="def-name">📊 风险等级</span>
                <span class="def-desc">低风险(0-30%) / 中风险(30-60%) / 高风险(60-100%)</span>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>📋 全部特征中英文对照</h2>
        <div class="table-wrapper">
            <table>
                <thead><tr><th>特征名</th><th>中文说明</th><th>English</th></tr></thead>
                <tbody>{feature_table_rows}</tbody>
            </table>
        </div>
    </div>

    <div class="section">
        <h2>📊 模型性能</h2>
        <div class="stat-grid">
            <div class="stat-card"><div class="label">AUC</div><div class="value">{results['auc']:.3f}</div></div>
            <div class="stat-card"><div class="label">准确率</div><div class="value">{results['classification_report']['accuracy']:.3f}</div></div>
            <div class="stat-card"><div class="label">精确率</div><div class="value">{results['classification_report']['weighted avg']['precision']:.3f}</div></div>
            <div class="stat-card"><div class="label">召回率</div><div class="value">{results['classification_report']['weighted avg']['recall']:.3f}</div></div>
            <div class="stat-card"><div class="label">F1 分数</div><div class="value">{results['classification_report']['weighted avg']['f1-score']:.3f}</div></div>
        </div>
    </div>

    <div class="section">
        <h2>📈 特征重要性排名</h2>
        <div class="table-wrapper">
            <table>
                <thead><tr><th>特征</th><th>中文说明</th><th>English</th><th>重要性</th><th>贡献</th></tr></thead>
                <tbody>{importance_rows}</tbody>
            </table>
        </div>
        <div class="insight-box">
            💡 <strong>关键洞察</strong>：<br>
            • <strong>{results['feature_importance'].iloc[0]['feature']}</strong> ({results['feature_importance'].iloc[0]['中文说明']}) 是最强的流失预测因子<br>
            • 前 3 大特征贡献了 {results['feature_importance'].head(3)['importance'].sum()*100:.0f}% 的预测能力
        </div>
    </div>

    <div class="section">
        <h2>📊 特征组重要性</h2>
        <div class="table-wrapper">
            <table>
                <thead><tr><th>特征组</th><th>重要性</th><th>贡献</th></tr></thead>
                <tbody>{group_rows}</tbody>
            </table>
        </div>
    </div>

    <div class="section">
        <h2>📋 混淆矩阵</h2>
        <div class="table-wrapper">
            <table>
                <thead><tr><th>实际 \\ 预测</th><th>留存</th><th>流失</th><th>合计</th></tr></thead>
                <tbody>{cm_rows}</tbody>
            </table>
        </div>
    </div>

    <div class="section">
        <h2>📊 风险分布</h2>
        <div class="table-wrapper">
            <table>
                <thead><tr><th>风险等级</th><th>用户数</th><th>占比</th><th>定义与建议</th></tr></thead>
                <tbody>{risk_rows}</tbody>
            </table>
        </div>
    </div>

    <div class="footer">
        <p>Generated by Skill 8.1: 流失预警模型</p>
    </div>
</div>
</body>
</html>'''

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✅ Report saved: {output_path}")
    return output_path


# ============================================================
# 4. 主函数
# ============================================================

def main():
    print("=" * 60)
    print("Skill 8.1: 流失预警模型 (深入特征版)")
    print("=" * 60)

    print("\n📊 Generating sample data...")
    df = generate_sample_data(5000)

    print(f"   Total users: {len(df)}")
    print(f"   Churned: {df['is_churned'].sum()} ({df['is_churned'].mean()*100:.1f}%)")

    print("\n🤖 Building churn prediction model...")
    results = build_churn_model(df)

    print("\n" + "-" * 50)
    print("MODEL PERFORMANCE")
    print("-" * 50)
    print(f"  AUC: {results['auc']:.3f}")
    print(f"  Accuracy: {results['classification_report']['accuracy']:.3f}")

    print("\n  Feature Importance:")
    for _, row in results['feature_importance'].head(5).iterrows():
        print(f"    {row['feature']}: {row['importance']*100:.1f}%")

    print("\n📄 Generating HTML report...")
    report_path = generate_html_report(results)

    print("\n" + "=" * 60)
    print("✅ DONE!")
    print("=" * 60)
    print(f"\n📁 Open report: {report_path}")


if __name__ == "__main__":
    main()
