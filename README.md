# Survey Participation & Incentive Optimization

An end-to-end machine learning and decision-optimization framework for improving survey participation and optimizing respondent incentives.

The project combines:

1. **Survey participation propensity modeling** — identify respondents most likely to complete a survey.
2. **Incentive response modeling** — estimate completion probability under different incentive levels.
3. **Economic value modeling** — assign business value to survey completions based on survey category and length.
4. **Economic optimization** — select the incentive that maximizes expected economic value.
5. **Next-best-action generation** — produce a respondent-level recommendation consisting of audience eligibility, incentive, and survey context.

> **Important:** The optimized economic outcomes are model-based counterfactual estimates, not realized savings or causal effects. Historical incentive assignment is observational and should be validated through randomized experimentation before production deployment.

---

## 1. Project Objective

The objective is to move survey outreach from a largely uniform incentive strategy toward an AI-assisted **next-best-action** approach.

The target decision flow is:

```text
Respondent + Survey Context
            │
            ▼
   Propensity Model
            │
            ▼
    Candidate Audience
            │
            ▼
 Incentive Response Model
            │
            ▼
   Candidate Incentives
            │
            ▼
     Economic Value
            │
            ▼
 Economic Optimization
            │
            ▼
       Next Best Action
```

The optimization objective is:

```text
Expected Value =
    P(completion | respondent, survey, incentive)
    × Survey Economic Value
    − Incentive Cost
```

---

## 2. Data Architecture

The project deliberately uses **Excel files rather than a database**.

The source data is logically separated into four files and merged at runtime:

```text
data/raw/
├── respondents/
│   └── respondents_v001.xlsx
├── surveys/
│   └── surveys_v001.xlsx
├── invitations/
│   └── invitations_v001.xlsx
└── responses/
    └── responses_v001.xlsx
```

This keeps the source datasets independent while allowing the modeling pipeline to construct an invitation-level analytical dataset.

The project does **not** store train/test/validation labels in Excel. Splits are generated programmatically using invitation date.

---

# 3. Repository Structure

```text
survey-participation-optimizer/
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   │   ├── respondents/
│   │   │   └── respondents_v001.xlsx
│   │   ├── surveys/
│   │   │   └── surveys_v001.xlsx
│   │   ├── invitations/
│   │   │   └── invitations_v001.xlsx
│   │   └── responses/
│   │       └── responses_v001.xlsx
│   │
│   ├── interim/
│   └── processed/
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── assembler.py
│   │   └── validation.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── propensity_features.py
│   │   ├── incentive_response_features.py
│   │   └── feature_audit.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── propensity.py
│   │   └── incentive_response.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── classification.py
│   │   ├── ranking.py
│   │   ├── thresholds.py
│   │   ├── plots.py
│   │   └── evaluation.py
│   │
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── economic_value.py
│   │   └── incentive_optimizer.py
│   │
│   └── pipelines/
│       └── __init__.py
│
├── models/
│   ├── final_propensity_hgb.joblib
│   └── final_incentive_response_hgb.joblib
│
├── evaluation/
│   ├── final/
│   ├── phase3_incentive/
│   └── final_propensity_test_results.csv
│
├── tests/
│   ├── __init__.py
│   ├── test_data_pipeline.py
│   ├── test_features.py
│   ├── test_feature_audit.py
│   ├── test_logistic_regression.py
│   ├── test_model_comparison.py
│   ├── test_threshold_analysis.py
│   ├── test_model_diagnostics.py
│   ├── test_phase2_tuning.py
│   ├── test_incentive_assignment.py
│   └── test_final_system.py
│
├── deployment/
│   └── azure/
│       ├── aml/
│       ├── api/
│       └── pipelines/
│
├── deployment/
│   └── docker/
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

# 4. Configuration

## `configs/config.yaml`

Central configuration for the project.

It defines:

- Project name and version
- Locations of the four source Excel files
- Target variable
- Temporal split configuration
- Date column used for splitting
- Train/validation/test proportions

Current split:

```text
Train       70%
Validation  15%
Test        15%
```

The temporal split is important because survey participation is a time-dependent business problem. Random splitting could allow future behavior to influence the training set.

---

# 5. Data Layer

The data layer is responsible for loading, assembling and validating the source data.

## `src/data/loader.py`

Responsible for reading the source data.

Main responsibilities:

- Load `config.yaml`
- Read respondent Excel data
- Read survey Excel data
- Read invitation Excel data
- Read response Excel data
- Convert relevant date fields to pandas datetime
- Return the source datasets as a dictionary

Conceptually:

```text
Excel files
    ↓
loader.py
    ↓
source_data
```

---

## `src/data/assembler.py`

Builds the modeling dataset by joining the four source datasets.

The primary modeling grain is:

```text
One row = One survey invitation
```

The main joins are:

```text
Invitations
    +
Respondents
    +
Surveys
    +
Response outcome
```

Important validation assumptions include:

- One respondent can have many invitations.
- One survey can have many invitations.
- One invitation should have at most one response outcome.

The resulting dataset contains the modeling features and the `completed` target.

---

## `src/data/validation.py`

Provides structural and modeling-safety validation.

Responsibilities include:

- Checking required columns
- Checking missing identifiers
- Checking duplicate invitation IDs
- Checking valid binary target values
- Checking date availability
- Checking forbidden/leaky features
- Creating the temporal train/validation/test split

The module explicitly protects against target leakage.

Examples of forbidden fields include:

```text
completed
click_date
completion_date
invitation_id
user_id
survey_id
invitation_date
date_of_join
base_engagement
incentive_sensitivity
fatigue_sensitivity
```

---

# 6. Feature Engineering

## `src/features/propensity_features.py`

Defines the feature set and preprocessing pipeline for the propensity model.

### Numerical features

```text
age
survey_length_minutes
survey_complexity
incentive_amount
prior_invitations
prior_clicks
prior_completions
prior_click_rate
prior_completion_rate
days_since_last_invite
days_since_last_completion
```

### Categorical features

```text
gender
loc_india
survey_category
device
channel
```

Preprocessing includes:

- Median imputation for numerical variables
- StandardScaler for numerical variables
- Most-frequent imputation for categorical variables
- One-hot encoding for categorical variables

---

## `src/features/incentive_response_features.py`

Defines the feature pipeline for the incentive-response model.

It uses the propensity features plus:

```text
incentive_amount_squared
```

The squared incentive term allows the model to represent nonlinear relationships between incentive level and completion probability.

For example:

```text
Completion probability
          ^
          |       ______
          |     /
          |   /
          | /
          +----------------> Incentive
```

The actual relationship is learned by the model rather than imposed as a specific curve.

---

## `src/features/feature_audit.py`

Provides feature-level checks designed to prevent leakage and inappropriate variables from entering the model.

The audit is particularly important because the original modeling approach had severe target leakage.

The previous approach produced ROC-AUC close to 1.0 because outcome-derived variables were available to the model.

The new pipeline explicitly separates:

```text
Information available at decision time
```

from:

```text
Information generated after the decision
```

---

# 7. Propensity Models

## `src/models/propensity.py`

Contains the candidate propensity models.

Models evaluated included:

```text
Logistic Regression
Decision Tree
Random Forest
Gradient Boosting
HistGradientBoosting
```

The final model selected was:

```text
HistGradientBoostingClassifier
```

Logistic Regression was retained as an interpretable benchmark.

The final propensity model is stored as:

```text
models/final_propensity_hgb.joblib
```

---

# 8. Incentive Response Model

## `src/models/incentive_response.py`

Defines the models used to estimate completion probability under different incentive levels.

Two candidate models were evaluated:

```text
Logistic Regression
HistGradientBoosting
```

HGB was selected as the final response model.

The final model is stored as:

```text
models/final_incentive_response_hgb.joblib
```

The response model is evaluated on historical incentives for predictive performance and then used to generate counterfactual predictions across candidate incentive values.

---

# 9. Evaluation Layer

## `src/evaluation/classification.py`

Contains classification metrics used to assess model performance.

Important metrics include:

- ROC-AUC
- PR-AUC
- Log Loss
- Brier Score
- Precision
- Recall
- F1

---

## `src/evaluation/ranking.py`

Focuses on the business use case of ranking respondents.

Important outputs include:

- Propensity deciles
- Completion rate by decile
- Lift
- Cumulative gain
- Top-decile performance
- Top-20% / top-30% audience performance

This is particularly important because the model is primarily used to prioritize respondents rather than simply classify them at a fixed threshold.

---

## `src/evaluation/thresholds.py`

Supports analysis of classification thresholds.

The project ultimately avoided relying on a single fixed threshold such as:

```text
P(completion) >= 0.50
```

because threshold-based classification was not the primary business objective.

---

## `src/evaluation/plots.py`

Contains plotting utilities for model evaluation and diagnostics.

Typical visualizations include:

- Model comparison
- Calibration curves
- Propensity decile performance
- Incentive-response curves
- Economic comparisons
- Incentive distributions

---

## `src/evaluation/evaluation.py`

Provides higher-level evaluation functionality and consolidates model evaluation outputs.

This module is intended to keep evaluation logic separate from model construction.

---

# 10. Economic Optimization

## `src/optimization/economic_value.py`

Defines the business-derived economic value of a completed survey.

Current segmentation:

| Survey Category | Economic Segment | Value Range |
|---|---|---:|
| Consumer | B2C | $2–$5 |
| Retail | B2C | $2–$5 |
| B2B | B2B | $15–$30 |
| Finance | B2B | $15–$30 |
| Technology | B2B | $15–$30 |
| Healthcare | Healthcare | $50–$100 |

Within each category, economic value is scaled according to survey length.

This is a **business assumption**, not a machine-learning prediction.

---

## `src/optimization/incentive_optimizer.py`

This is the decision engine.

It evaluates candidate incentive levels for each respondent/survey combination.

Current production candidate grid:

```text
$0.50
$1.00
$1.50
$2.00
$2.50
$3.00
$3.50
$4.00
$5.00
$6.00
$8.00
$10.00
$12.00
$15.00
```

For each candidate:

```text
Predicted Completion Probability
                ×
        Survey Economic Value
                −
          Incentive Cost
                =
        Expected Economic Value
```

The optimizer selects the incentive with the highest expected value.

If multiple incentives have the same expected value, the lower incentive is preferred.

---

# 11. Test and Experiment Scripts

The `tests/` directory contains executable validation and experiment scripts.

These are not limited to unit tests; several are analytical experiment runners.

## `tests/test_data_pipeline.py`

Tests the data loading, assembly and validation pipeline.

---

## `tests/test_features.py`

Tests feature creation and preprocessing.

---

## `tests/test_feature_audit.py`

Tests leakage and forbidden-feature controls.

---

## `tests/test_logistic_regression.py`

Evaluates the Logistic Regression propensity baseline.

---

## `tests/test_model_comparison.py`

Runs the Phase I comparison of candidate propensity models.

---

## `tests/test_threshold_analysis.py`

Evaluates model behavior at different classification thresholds.

---

## `tests/test_model_diagnostics.py`

Runs calibration and diagnostic analysis.

---

## `tests/test_phase2_tuning.py`

Runs Phase II hyperparameter tuning using time-aware cross-validation.

The tuning framework compares:

- Logistic Regression
- HistGradientBoosting

The final conclusion was that additional tuning produced only marginal improvement.

---

## `tests/test_incentive_assignment.py`

Performs Phase IV-A analysis of historical incentive assignment.

It examines whether incentives vary materially with:

- Respondent characteristics
- Survey characteristics
- Engagement history
- Channel
- Device
- Time

It also produces incentive quantile and historical trend analyses.

---

## `tests/test_final_system.py`

Runs the final end-to-end evaluation.

The final workflow is:

```text
Load data
   ↓
Build modeling dataset
   ↓
Temporal development/test split
   ↓
Train final propensity model
   ↓
Evaluate propensity on test
   ↓
Train incentive response model
   ↓
Evaluate response model
   ↓
Calculate economic value
   ↓
Generate candidate incentives
   ↓
Optimize incentive
   ↓
Compare historical vs optimized policy
   ↓
Save final outputs
```

---

# 12. Model Outputs

The `models/` directory contains the final serialized models.

```text
models/
├── final_propensity_hgb.joblib
└── final_incentive_response_hgb.joblib
```

These can be loaded later without retraining the models.

---

# 13. Evaluation Outputs

The `evaluation/` directory stores analytical outputs.

Typical structure:

```text
evaluation/
├── phase3_incentive/
│   ├── incentive_quantile_summary.csv
│   ├── monthly_incentive_summary.csv
│   ├── incentive_quantile_completion.png
│   ├── historical_incentive_scatter.png
│   ├── historical_mean_incentive_over_time.png
│   └── historical_completion_rate_over_time.png
│
└── final/
    ├── final_system_test_summary.csv
    ├── final_test_optimal_incentive_distribution.csv
    ├── final_test_incentive_by_category.csv
    ├── final_test_optimal_decisions.csv
    ├── final_test_historical_policy.csv
    └── final_propensity_test_deciles.csv
```

These outputs provide both model-level and business-level evidence.

---

# 14. Modeling Phases

## Phase I — Model Development

Candidate models were compared.

Final direction:

```text
HGB → performance candidate
Logistic Regression → interpretable benchmark
```

---

## Phase II — Hyperparameter Tuning

Time-aware cross-validation was used.

Result:

```text
Further tuning provided negligible improvement.
```

The project therefore avoided unnecessary hyperparameter optimization.

---

## Phase III — Calibration & Propensity Evaluation

Calibration methods tested:

```text
Raw HGB
HGB + Sigmoid
HGB + Isotonic
```

Raw HGB performed best and was retained.

Final propensity performance:

```text
ROC-AUC       0.6977
PR-AUC        0.3228
Log Loss      0.4187
Brier Score   0.1297
Top-decile lift ≈ 2.34x
```

---

## Phase IV — Incentive Modeling

Historical incentive assignment was analyzed before building the response model.

The response model then estimated:

```text
P(completion | respondent, survey, incentive)
```

The model was evaluated on historical incentives and subsequently used for counterfactual incentive scenarios.

---

## Phase V — End-to-End Optimization

The final system combined:

```text
Propensity
+
Response Model
+
Economic Value
+
Incentive Optimizer
```

to produce respondent-level incentive recommendations.

---

# 15. Final Business Outcome

On the untouched test period:

| Metric | Historical Policy | Optimized Policy |
|---|---:|---:|
| Incentive expenditure | $90,486.54 | $17,153.50 |
| Modeled expected economic value | $6,728.61 | $77,305.89 |
| Change in incentive cost | — | -$73,333.04 |
| Incremental modeled EV | — | +$70,577.28 |

The modeled incentive expenditure reduction is approximately:

```text
81.0%
```

The optimized policy selected:

```text
$0.50 for 92.77% of test decisions
```

with the remaining decisions receiving higher incentives where the model estimated sufficient incremental economic value.

---

# 16. Important Methodological Caveats

### 1. Counterfactual incentive predictions are not causal

Historical incentives were not assigned through a randomized experiment.

Therefore:

```text
Model prediction ≠ causal incentive effect
```

The optimizer should be viewed as a decision hypothesis until experimentally validated.

### 2. $0 incentive is outside historical support

A sensitivity analysis showed that the optimizer strongly preferred $0 when it was introduced into the candidate grid.

However, the historical dataset contained no $0 incentive observations.

Therefore, $0 recommendations represent model extrapolation and should be tested experimentally rather than treated as established evidence.

### 3. Economic value is business-derived

The $2–$5, $15–$30 and $50–$100 ranges were defined as business assumptions.

They were not learned from historical revenue or profitability data.

### 4. Propensity/incentive sequencing

The current propensity model includes historical `incentive_amount`.

If the production workflow must select the audience **before** assigning an incentive, this feature should be removed or the propensity model should explicitly be defined as conditional on a planned baseline incentive.

---

# 17. Recommended Production Experiment

Before deploying the optimized policy broadly, run a randomized incentive experiment.

A potential experimental design is:

```text
Control / low incentive
$0.50
$2.00
$4.00
$6.00
```

Randomization should be appropriately stratified by factors such as:

- Survey category
- Respondent segment
- Propensity band
- Survey length

The experiment should measure:

```text
Completion rate
Cost per completion
Incremental completions
Incentive spend
Economic value
Net economic value
```

The experiment will provide the causal evidence required to validate the optimization policy.

---

# 18. Technology Stack

Primary technologies:

```text
Python
pandas
NumPy
scikit-learn
Jupyter Notebook
PyYAML
joblib
Excel
Matplotlib
```

The architecture has been structured so that the modeling and optimization components can later be exposed through an API and deployed on Azure.

Potential future production components include:

```text
Azure Machine Learning
Azure API
Docker
CI/CD
Model monitoring
Feature/data quality monitoring
Experimentation platform
```

---

# 19. Running the Project

Create and activate the virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the final system:

```bash
python tests/test_final_system.py
```

Run individual analytical phases as required:

```bash
python tests/test_data_pipeline.py
python tests/test_features.py
python tests/test_feature_audit.py
python tests/test_model_comparison.py
python tests/test_phase2_tuning.py
python tests/test_incentive_assignment.py
python tests/test_final_system.py
```

---

# 20. Design Philosophy

The project follows several principles:

### Temporal integrity

Training data must precede evaluation data.

### Leakage prevention

Variables generated after the invitation decision cannot be used as predictors.

### Ranking over arbitrary classification thresholds

The business problem is primarily audience prioritization.

### Separation of prediction and optimization

The ML model predicts behavior; the optimization layer makes the economic decision.

### Business-aware ML

Model performance is evaluated alongside economic outcomes.

### Experimental validation

Counterfactual optimization recommendations must ultimately be validated through randomized experiments.

---

## 21. Final Architecture

```text
                     ┌─────────────────────┐
                     │   Respondent Data   │
                     └──────────┬──────────┘
                                │
                     ┌──────────▼──────────┐
                     │     Survey Data     │
                     └──────────┬──────────┘
                                │
                     ┌──────────▼──────────┐
                     │  Invitation Data   │
                     └──────────┬──────────┘
                                │
                     ┌──────────▼──────────┐
                     │  Response Outcome  │
                     └──────────┬──────────┘
                                │
                       Data Assembly
                                │
                                ▼
                    ┌──────────────────────┐
                    │ Propensity HGB Model │
                    └──────────┬───────────┘
                               │
                       Candidate Audience
                               │
                               ▼
                 ┌──────────────────────────┐
                 │ Incentive Response Model │
                 └────────────┬─────────────┘
                              │
                    Candidate Incentives
                              │
                              ▼
                 ┌──────────────────────────┐
                 │    Economic Value Model  │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │   Economic Optimizer     │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │     Next Best Action     │
                 │ Respondent + Incentive   │
                 │        + Survey          │
                 └──────────────────────────┘
```

---

## 22. Project Status

**Current status: Modeling prototype complete.**

The project has progressed from data assembly and leakage prevention through model selection, tuning, calibration, incentive-response modeling and economic optimization.

The next logical stage is **production hardening and causal validation**, rather than additional model tuning.

Key next steps:

1. Refactor the propensity feature set if incentive is not known at audience-selection time.
2. Add production pipeline orchestration.
3. Add model/data quality monitoring.
4. Build an API serving layer.
5. Design and execute randomized incentive experiments.
6. Compare predicted versus realized economic outcomes.
7. Deploy the validated policy through Azure.
