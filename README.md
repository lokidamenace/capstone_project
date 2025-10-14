# An In-Depth Analysis of Stroke Risk Factors Using Predictive Analytics




<img width="2392" height="1192" alt="pic1" src="https://github.com/user-attachments/assets/e7c0d7f3-7733-4312-83c3-98a2441167d2" />

## Part I:
### Executive summary
According to the World Health Organization (WHO), stroke is the second leading cause of death worldwide, accounting for approximately 11% of all deaths, resulting in high medical costs and significant social and economic impacts. Early detection and prevention are crucial. However, healthcare organizations often lack data-driven tools to identify individuals at high risk before a stroke occurs proactively. This project aims to examine key factors associated with stroke risk using a publicly available stroke dataset to uncover demographic, lifestyle, and health factors related to stroke occurrence. In addition, the study seeks to develop predictive and interpretable models that can identify patients at elevated risk and tailor preventive care strategies accordingly.

### Rationale
The research is crucial because it helps understand and anticipate stroke risk. By identifying the most influential predictors, this question reveals which health, lifestyle, and demographic factors have the most significant impact on the likelihood of experiencing a stroke. For instance, determining whether age, glucose levels, hypertension, or smoking status has a more substantial role offers valuable insights for clinicians, caregivers, and policymakers. These findings can inform preventive strategies, such as targeting high-risk groups for early screening or implementing lifestyle interventions.

### Research Question
Which features are the strongest predictors of stroke occurrence, and how accurately can we predict stroke using machine learning models?

### Data Sources
A dataset sufficient to answer descriptive, diagnostic, and predictive questions should include both medical and behavioral variables commonly associated with stroke. An appropriate stroke dataset should include the following information:

* Demographics: gender, age, residence type, marital status
* Medical history: hypertension, heart disease, BMI, average glucose level
* Lifestyle: work type, smoking status
* Target variable: stroke (yes or no)

Source: kaggle.com

### Exploratory Data Analysis
Before moving into modeling, exploratory data analysis (EDA) was conducted to gain a clear understanding of the stroke dataset. The goal was to understand the data, check for missing values and patterns, and examine how factors such as age, glucose levels, and BMI relate to stroke occurrence. This process helped reveal initial trends and potential key variables, providing a solid foundation for building and interpreting predictive models.

<img width="2390" height="1342" alt="pic2" src="https://github.com/user-attachments/assets/5232d82a-30e4-4a27-8ac1-95e6ab686a1b" />

## Part II:
### Methodology
The Logistic Regression model will serve as the baseline, providing an interpretable way to understand how each variable affects the likelihood of a stroke. It will set a benchmark for both accuracy and explainability. Following this, the Random Forest model will be employed, which utilizes an ensemble of decision trees to capture nonlinear relationships and determine the most critical factors influencing stroke outcomes.

The third model we will implement is the Support Vector Machine (SVM), which uses a margin-based approach to classify cases as either stroke or non-stroke. By employing a kernel function such as the Radial Basis Function (RBF), SVM can effectively handle complex, nonlinear data patterns while addressing class imbalance through class weighting. Lastly, we will develop a Gradient Boosting model, such as XGBoost or LightGBM, to achieve the highest predictive accuracy. This model will build trees sequentially to minimize errors and will provide feature-level explanations using SHAP (Shapley Additive Explanations) values.

All models will be evaluated using consistent performance metrics, accuracy, ROC-AUC, precision, recall, and F1-score, with cross-validation to ensure reliability. The best-performing model will be selected based on a balance of accuracy, interpretability, and ethical fairness.

### Modeling & Analysis
Together, these four techniques create a systematic modeling framework that progresses from a simple, interpretable model to more complex and powerful ones. This multi-model approach ensures a well-rounded analysis—revealing not only how accurately stroke can be predicted, but also which risk factors have the most significant influence, supporting data-driven prevention and decision-making in healthcare.

### Expected Outcomes
* Identification of top risk factors for stroke (e.g., age, glucose level, BMI, hypertension).
* Comparison of predictive model performance to determine the most effective algorithm.
* An interactive dashboard visualizing stroke likelihood by demographic and health attributes.
* Recommendations for using predictive insights in health monitoring or community wellness programs.

### Next steps
* Evaluate the ethical implications of using predictive analytics in healthcare decision-making.
* Ensure fairness by examining model performance across demographic subgroups (e.g., gender, residence).
* Present findings through a business intelligence (BI) dashboard that visualizes patterns and predictions for stakeholder use.


