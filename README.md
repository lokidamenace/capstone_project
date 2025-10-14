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
#### Correlation Heatmap for the Stroke Dataset
The correlation heatmap below shows the relationships between variables in the dataset. Age, hypertension, heart disease, and elevated glucose levels all positively correlate with stroke risk, indicating that older individuals and those with these conditions are more prone to strokes. BMI and smoking status also have mild correlations with stroke risk. In contrast, factors like gender and work type show minimal correlation. However, it is important to keep in mind that correlation does not imply causation. Strokes typically result from multiple interacting factors, leading to moderate correlations. Further statistical analyses can provide deeper insights into these relationships.
<img width="2850" height="2400" alt="heatmap" src="https://github.com/user-attachments/assets/efc140ea-9988-493c-bec7-ba58fa57d10c" />

#### Distribution of Age by Stroke Status
This histogram illustrates the age distribution of individuals, distinguishing between those who experienced a stroke (orange bars) and those who did not (blue bars). The visualization clearly indicates that strokes predominantly occur among older individuals, as shown by the higher concentration of orange bars on the right side of the plot. Younger individuals rarely experience strokes, while the frequency significantly increases with advancing age. This information highlights how age affects stroke risk. As people get older, the chances of a stroke usually increase. The overlap in some age groups shows that while age is significant, it is not the only factor. Other factors, such as high blood pressure, heart disease, and lifestyle choices, should also be considered.
<img width="3000" height="1800" alt="histogram" src="https://github.com/user-attachments/assets/de22d22d-ae90-422a-9715-a1c631c230e7" />

#### Age versus Glucose Level By Stroke Status
The scatter plot illustrates the relationship between age, average glucose levels, and stroke occurrence. Orange dots represent individuals who had a stroke, while blue dots indicate those who did not. The plot shows a trend in which strokes are more common among older adults and are associated with higher average glucose levels. There is a notable overlap between stroke and non-stroke groups, suggesting a hypothesis that both age and elevated glucose levels are associated with an increased risk of stroke. Further analysis is needed to understand the combined impact of these variables on stroke likelihood.
<img width="3000" height="1800" alt="scatterplot (1)" src="https://github.com/user-attachments/assets/296e61e6-498d-4138-a735-3fa044b488b1" />

#### Average Glucose Level by Stroke Status
The boxplot comparing average glucose levels by stroke status shows that individuals who experienced a stroke generally had higher glucose levels than those who did not. The median glucose level is noticeably higher in the stroke group, and the spread of values is also wider, suggesting greater variability among these individuals. A few extremely high values appear as outliers, likely representing cases with very high blood sugar. This pattern indicates a possible link between elevated glucose levels and stroke occurrence, which aligns with research suggesting that high blood sugar and diabetes increase stroke risk. While this visual shows a clear difference between the groups, further analysis is needed to confirm whether glucose level is an independent predictor of stroke when other factors are considered.
<img width="2400" height="1800" alt="boxplot" src="https://github.com/user-attachments/assets/dc0f8760-7508-40ba-9bb5-d3d820ad7383" />

The EDA provided a clear understanding of the stroke dataset and revealed several important patterns. The data showed that stroke cases accounted for a small proportion of the sample, confirming a significant class imbalance. Key variables such as age, average glucose level, and BMI displayed visible differences between individuals who experienced a stroke and those who did not. In particular, higher glucose levels and older age appeared more common among stroke cases. Some missing values were identified, especially in BMI, but these were handled through imputation. Overall, the EDA established a solid foundation for the modeling phase by highlighting which factors may be most influential in predicting stroke occurrence.

<img width="2390" height="1342" alt="pic2" src="https://github.com/user-attachments/assets/5232d82a-30e4-4a27-8ac1-95e6ab686a1b" />

## Part II:
### Methodology
The Logistic Regression model will serve as the baseline, providing an interpretable way to understand how each variable affects the likelihood of a stroke. It will set a benchmark for both accuracy and explainability. Following this, the Random Forest model will be employed, which utilizes an ensemble of decision trees to capture nonlinear relationships and determine the most critical factors influencing stroke outcomes.

The third model we will implement is the Support Vector Machine (SVM), which uses a margin-based approach to classify cases as either stroke or non-stroke. By employing a kernel function such as the Radial Basis Function (RBF), SVM can effectively handle complex, nonlinear data patterns while addressing class imbalance through class weighting. Lastly, we will develop a Gradient Boosting model, such as XGBoost or LightGBM, to achieve the highest predictive accuracy. This model will build trees sequentially to minimize errors and will provide feature-level explanations using SHAP (Shapley Additive Explanations) values.

All models will be evaluated using consistent performance metrics, accuracy, ROC-AUC, precision, recall, and F1-score, with cross-validation to ensure reliability. The best-performing model will be selected based on a balance of accuracy, interpretability, and ethical fairness.

### Modeling & Analysis
Together, these four techniques create a systematic modeling framework that progresses from a simple, interpretable model to more complex and powerful ones. This multi-model approach ensures a well-rounded analysis—revealing not only how accurately stroke can be predicted, but also which risk factors have the most significant influence, supporting data-driven prevention and decision-making in healthcare.

#### I. Predicting Stroke with Decision Tree Analysis
The decision tree is a visualization tool that helps identify the most important factors for predicting stroke risk. Each box represents a decision point, starting with the most critical factor, which branches into secondary factors based on answers. The color coding indicates stroke outcomes: blue for no stroke and orange for stroke. The depth of the tree is limited for clarity, focusing on key predictors such as age and glucose levels. Although these factors are strong predictors on their own, stroke risk is influenced by multiple interacting variables, pointing to its complex nature.
<img width="6000" height="3000" alt="decisiontree" src="https://github.com/user-attachments/assets/ea94e922-1bd8-4bed-806b-e32ce4284ecc" />


### Expected Outcomes
* Identification of top risk factors for stroke (e.g., age, glucose level, BMI, hypertension).
* Comparison of predictive model performance to determine the most effective algorithm.
* An interactive dashboard visualizing stroke likelihood by demographic and health attributes.
* Recommendations for using predictive insights in health monitoring or community wellness programs.

### Next steps
* Evaluate the ethical implications of using predictive analytics in healthcare decision-making.
* Ensure fairness by examining model performance across demographic subgroups (e.g., gender, residence).
* Present findings through a business intelligence (BI) dashboard that visualizes patterns and predictions for stakeholder use.


