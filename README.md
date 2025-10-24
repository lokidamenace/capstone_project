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

All models will be evaluated using consistent performance metrics, accuracy, ROC-AUC, precision, recall, and F1-score, with cross-validation to ensure reliability. The best-performing model will be selected based on a balance of accuracy and interpretability.

### Modeling & Analysis
Together, these four techniques create a systematic modeling framework that progresses from a simple, interpretable model to more complex and powerful ones. This multi-model approach ensures a well-rounded analysis—revealing not only how accurately stroke can be predicted, but also which risk factors have the most significant influence, supporting data-driven prevention and decision-making in healthcare.

As a baseline, a simple logistic regression model with this threshold is not performing well at predicting stroke. This highlights the challenge of working with imbalanced datasets where the positive class (stroke) is rare. Therefore, we will proceed to the next three models now: train Random Forest, SVM (RBF), and Gradient Boosting with the same preprocessing; then evaluate using ROC-AUC and PR-AUC.

While all three models show improvement over the initial logistic regression model (which had very low recall and F1 scores), the Gradient Boosting model achieved the best scores on both ROC-AUC and PR-AUC, suggesting it is the most effective of the three for predicting stroke risk in this dataset (see side-by-side comparison of the three models below).
<img width="1987" height="957" alt="sidebyside" src="https://github.com/user-attachments/assets/b2747843-15b0-474e-ba20-0b3a7c62377f" />

### Model Comparison Summary and Interpretation

The following four supervised learning models have been trained and tested: Logistic Regression, Random Forest, Support Vector Machine (SVM with RBF kernel), and Gradient Boosting. All models followed the same steps to prepare the data, which included filling in missing values, scaling, and one-hot encoding. Their performance was evaluated on the test dataset using two key metrics: ROC-AUC and PR-AUC (Average Precision). These metrics help us understand how well each model performs, especially with imbalanced data, where simply looking at accuracy can be misleading.

1. Logistic Regression (Baseline)
Logistic Regression provides a good starting point for understanding the factors that predict strokes. It shows a moderate ability to spot stroke cases. Still, it struggles to identify true positives due to a severe data imbalance, with very few strokes compared to the overall number of cases. While it can generally distinguish between cases, its low performance in finding positive cases limits its usefulness. Despite these challenges, Logistic Regression helps us understand how different factors contribute, but it cannot handle complex, nonlinear relationships.

2. Random Forest
The Random Forest model performed better than Logistic Regression because it can capture complex relationships between variables and reduce overfitting by averaging results from multiple trees. It achieved higher ROC-AUC and PR-AUC scores, indicating it is better at identifying stroke cases while minimizing false positives and false negatives. The Random Forest model also ranks feature importance, showing which factors most impact stroke prediction, such as age, glucose level, and hypertension.

3. Support Vector Machine (SVM – RBF Kernel)
The SVM model produced moderate results, performing slightly worse than Random Forest and Gradient Boosting in both metrics. SVMs work well with complex decision boundaries, but they are sensitive to data scaling and can require substantial computation, especially with one-hot-encoded categorical data. Additionally, SVMs are not very interpretable, making them less suitable for healthcare applications, where explainability and transparency are essential.

4. Gradient Boosting
Gradient Boosting performed the best among all the models. It had the highest scores for ROC-AUC and PR-AUC, which measure how well the model can classify cases. This model learns incrementally, allowing it to correct mistakes made by earlier trees. As a result, it is better at telling apart stroke cases from non-stroke cases. It was particularly good at identifying less common stroke cases. Additionally, it shows which features are most important using SHAP values. This combination of strong prediction and clear explanations makes Gradient Boosting an excellent choice for predicting health risks.

### Conclusion
Overall, Gradient Boosting proved to be the most effective model for predicting stroke occurrence, striking an optimal balance between accuracy and sensitivity to minority cases. Random Forest also performed well, providing valuable interpretive insights. In contrast, Support Vector Machine (SVM) and Logistic Regression offered useful benchmarks but exhibited lower predictive power. It is worth noting that accuracy and F1-score can be misleading for imbalanced datasets, such as the stroke dataset, so we should focus instead on ROC-AUC and PR-AUC, which provide a more realistic picture of performance. Because stroke cases are rare, ROC-AUC and PR-AUC tell us far more about how well the model identifies the people most at risk.

### Identification of Key Predictors for Stroke
The Gradient Boosting model identified several key factors that contribute most strongly to predicting stroke risk. Age, average glucose level, and body mass index (BMI) emerged as the top three predictors, suggesting that older individuals with higher glucose levels and elevated BMI are at greater risk of stroke. This aligns with established medical evidence linking aging, diabetes, and obesity to cardiovascular events. In addition, hypertension and heart disease were also significant, further reinforcing the importance of chronic conditions in stroke risk assessment. While variables such as work type and smoking status had minor effects, they still contributed to overall prediction, reflecting the role of lifestyle and environmental factors (see chart below). 
<img width="1737" height="1037" alt="top 10" src="https://github.com/user-attachments/assets/5abce352-c4e0-4ca2-a64b-818d6c076646" />

### Next steps
* Interactive Dashboard for Stroke Likelihood: An interactive dashboard can help visualize the likelihood of stroke based on factors like age, gender, glucose levels, BMI, and medical history. Using predictions from a Gradient Boosting analysis, this dashboard allows users to see how changes in key risk factors affect the likelihood of having a stroke. Users can apply filters and use visual tools such as bar charts, heatmaps, and risk-level indicators to make the data easier to understand. This type of visualization improves clarity and supports quick decision-making, enabling early identification of individuals or groups at high risk.
* Recommendations for Applying Predictive Insights: We can use the findings from this analysis to improve health monitoring and community wellness programs. For example, clinics and health agencies can use the model's risk indicators to focus on screening people with high glucose levels, high blood pressure, or obesity. Community programs can create outreach and educational campaigns that promote lifestyle changes to lower stroke risk, such as eating better, exercising regularly, and managing blood pressure. Predictive analytics can also help allocate resources by identifying communities with higher overall risk. This way, we can target public health efforts and plan wellness programs based on the data.



