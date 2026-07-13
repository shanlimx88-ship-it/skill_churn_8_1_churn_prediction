# Skill 8.1: 流失预警模型

## Overview
Predict user churn risk using deep behavioral features. Identify high-risk users before they churn.

## Core Questions

| Question | Value |
|----------|-------|
| Which users are at high risk of churning? | Risk level classification (Low/Medium/High) |
| What behaviors predict churn best? | Feature importance ranking |
| How accurate is the prediction? | AUC, Accuracy, Precision, Recall, F1 |

## Features (18+ Behavioral Signals)

| Category | Features | Description |
|----------|----------|-------------|
| Basic Activity | active_days_14d, sessions_14d, avg_depth_14d | Usage frequency and depth |
| Value Signals | has_hva_14d, adoption_rate_14d, first_hva_days | User value realization |
| Retention Signals | day1_return, day7_return, day14_return | Early return behavior |
| Usage Breadth | num_scenarios, scenario_diversity, feature_count_14d | Exploration range |
| Usage Patterns | interval_std, activity_trend, weekday_ratio | Behavioral regularity |
| Engagement Quality | avg_wait_time, has_shared, viewed_pricing | Deep engagement signals |

## Methodology

### 1. Feature Engineering
- Feature Window: First 14 days of user activity
- Prediction Window: Day 15-45 (30-day churn detection)
- Target Variable: is_churned (1 = churned, 0 = retained)

### 2. Model
- Algorithm: Random Forest Classifier
- Class Weight: Balanced (handles imbalanced data)
- Validation: 70/30 train-test split with stratification

### 3. Risk Levels

| Level | Probability | Action |
|-------|-------------|--------|
| Low Risk | 0-30% | Monitor, normal operations |
| Medium Risk | 30-60% | Engage, send usage reminders |
| High Risk | 60-100% | Immediate intervention |

## Quick Start

bash:
pip install pandas numpy scikit-learn jinja2 matplotlib seaborn
python3 churn_prediction.py
open output/churn_report.html

## File Structure

skill_churn_8_1_churn_prediction/
├── churn_prediction.py      # Main script
├── requirements.txt         # Dependencies
├── README.md                # This file
└── output/
    └── churn_report.html    # HTML report

## License
Internal use only