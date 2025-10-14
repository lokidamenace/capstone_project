# An In-Depth Analysis of Stroke Risk Factors Using Predictive Analytics




<img width="2392" height="1192" alt="pic1" src="https://github.com/user-attachments/assets/e7c0d7f3-7733-4312-83c3-98a2441167d2" />

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
The first technique is Logistic Regression, which I will use as the baseline model. It helps predict if someone has had a stroke. This method clearly shows how different factors, like age, hypertension, or glucose level, affect the chance of having a stroke. We can convert the model’s coefficients into odds ratios to understand how strongly each factor influences the outcome. Logistic regression provides a straightforward base before we explore more complicated methods.

<img width="2390" height="1342" alt="pic2" src="https://github.com/user-attachments/assets/5232d82a-30e4-4a27-8ac1-95e6ab686a1b" />

### Methodology
The first technique is Logistic Regression, which I will use as the baseline model. It helps predict if someone has had a stroke. This method clearly shows how different factors, like age, hypertension, or glucose level, affect the chance of having a stroke. We can convert the model’s coefficients into odds ratios to understand how strongly each factor influences the outcome. Logistic regression provides a straightforward base before we explore more complicated methods.

The second technique, Random Forest, builds multiple decision trees and combines their predictions to improve prediction accuracy and reduce errors. This method can identify complex relationships and interactions among different factors that a simple model might overlook. It also ranks the importance of various factors, helping researchers find out which ones most clearly separate people who had a stroke from those who did not. Random Forest works well with data that includes different types of variables and some missing values, making it a good choice for understanding the main risk factors for stroke.

Lastly, I will use Gradient Boosting methods like XGBoost and LightGBM to improve our predictions. These algorithms create models one after another. Each new model learns from the mistakes of the previous ones, leading to high accuracy even with unbalanced data. Gradient Boosting models effectively find complex, nonlinear relationships between features. 

### Results
What did your research find?

### Next steps
What suggestions do you have for next steps?

### Outline of project

- [Link to notebook 1]()
- [Link to notebook 2]()
- [Link to notebook 3]()


### Contact and Further Information
