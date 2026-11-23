Predicting 30-Day Hospital Readmissions Among Patients With Diabetes
Machine Learning | Healthcare Analytics | Python | Scikit-learn

A reproducible machine-learning project that evaluates whether hospital encounter information can help identify patients with diabetes who are at risk of readmission within 30 days. The project compares Logistic Regression and Random Forest, uses patient-grouped validation to reduce leakage, and examines performance differences across demographic subgroups.
Research/educational use only. These models are not validated clinical decision-support tools and should not be used to make patient-care decisions.

Business problem
Unplanned hospital readmissions can disrupt recovery and increase healthcare utilization. Identifying patients who may need additional discharge planning or follow-up is a useful predictive analytics problem. This project investigates whether routinely recorded encounter-level variables can distinguish readmissions within 30 days from other outcomes, while recognizing that predictions alone do not establish which interventions would help.
Dataset
- Source: UCI Machine Learning Repository — Diabetes 130-US Hospitals for Years 1999–2008
- Observations: 101,766 hospital encounters
- Outcome: readmitted == '<30' is encoded as 1; all other outcomes as 0
- Positive-class prevalence: approximately 11.16% overall
- Patient grouping: patient_nbr is used for data splitting, not as a predictive feature
Selected predictors include length of stay, laboratory procedures, medications, prior inpatient/outpatient/emergency visits, number of diagnoses, age, gender, race, admission type/source, A1C and glucose result categories, and medication-change indicators.

The dataset is historical, and the findings may not generalize to present-day care settings.
Methods
1. Load the UCI data and inspect missing values and duplicates.
2. Define a binary 30-day readmission outcome and select predictor columns.
3. Create an approximately 80/20 patient-disjoint training/test split with GroupShuffleSplit; the saved run reports zero patient overlap.
4. Apply numeric imputation/scaling and categorical imputation/one-hot encoding inside each model pipeline, so preprocessing is learned only from the training partition or fold.
5. Compare class-weighted Logistic Regression and Random Forest using three-fold StratifiedGroupKFold cross-validation.
6. Select a model using mean cross-validated PR-AUC and evaluate predictions on the untouched holdout set.
7. Audit performance across age, gender, and race, marking subgroups with fewer than 30 positive outcomes as unstable.
The notebook also discusses XGBoost as an optional model, but the reported executed comparison contains only Logistic Regression and Random Forest.
Results
Three-fold patient-grouped cross-validation
Model	Mean PR-AUC	Mean ROC-AUC	Mean recall	Mean precision	Mean F1
Logistic Regression	0.1945	0.6325	0.4940	0.1668	0.2494
Random Forest	0.1575	0.6012	0.0481	0.2326	0.0797


Logistic Regression was selected based on its higher cross-validated PR-AUC. Random Forest's very low recall (0.0481) means that at the evaluated classification threshold it missed most positive readmissions. This may reflect its score distribution, class imbalance, or threshold choice; the notebook does not establish a single cause. Threshold tuning and calibration should be assessed on validation data before making operational claims.
Logistic Regression: held-out test set
Metric	Result
Test encounters	20,153
Positive prevalence	10.67%
Accuracy	0.6730
Precision	0.1587
Recall	0.4798
F1-score	0.2385
ROC-AUC	0.6269
PR-AUC / Average precision	0.1791


Confusion matrix (actual rows × predicted columns):
	Predicted negative	Predicted positive
Actual negative	12,531	5,471
Actual positive	1,119	1,032


The PR-AUC exceeds the holdout positive-prevalence baseline of 0.1067, but the 15.9% precision indicates many flagged encounters were not readmitted within 30 days. The model identified about 48% of positive cases at the 0.5 threshold. These results indicate modest discrimination, not clinical readiness.
Fairness, limitations, and responsible use
- Small subgroups: Seven age/gender/race subgroup categories had fewer than 30 positive outcomes. Their recall and PR-AUC estimates are particularly unstable; differences should not be treated as reliable evidence of disparities without uncertainty intervals and larger samples.
- Historical and observational data: Care patterns, coding, and patient populations may differ from modern practice. The dataset does not demonstrate that model-guided interventions improve outcomes.
- False negatives and false positives: Missed readmissions may leave patients without additional review; false alarms may create unnecessary workload and divert limited resources.
- Protected attributes: Age, gender, and race are included in the documented modeling features and used for auditing. Any real-world use would require careful review of necessity, fairness, and applicable law and policy.
- Privacy and governance: Patient identifiers must not be exposed in public repositories. Use only the public research dataset, restrict access to any sensitive derivatives, document model versions, and monitor performance drift.
- Validation: Before any deployment, conduct external and temporal validation, assess calibration, choose thresholds based on documented clinical costs, and require clinician oversight.
Repository structure
A suggested GitHub layout (adjust names to match files actually uploaded):
healthcare-readmission-prediction/
├── README.md
├── Healthcare_Readmission_Milestone2_Notebook.ipynb
├── requirements.txt                 # optional
├── reports/                         # optional: white paper and Q&A
└── outputs/                         # optional: generated figures and metrics
Note: The uploaded executed notebook contains some interactive troubleshooting cells and an unfinished Section 8 markdown heading. The model results above are from its saved outputs. For a clean reproducibility claim, remove debugging cells, replace Section 8 with findings, and run the finalized notebook top to bottom before publishing.
How to run
1. Install Python 3.10+ and Jupyter Notebook or JupyterLab.
2. Install dependencies:
   pip install jupyter pandas numpy scipy scikit-learn matplotlib seaborn xgboost ucimlrepo
3. Open the project notebook in Jupyter.
4. Download diabetic_data.csv from the UCI dataset page, and place it beside the notebook or in a data/ folder. The notebook also includes a download fallback.
5. Run the notebook from top to bottom. Review the saved tables and plots in outputs/ where available.
xgboost is optional for the reported two-model comparison. Exact reproducibility may require matching library versions and resolving local plotting/font configuration issues.
Future work
- Evaluate decision thresholds against precision–recall trade-offs and clinical workload constraints.
- Add calibration assessment and confidence intervals, especially for subgroup metrics.
- Perform temporal/external validation with more recent and representative data.
- Investigate alternative feature sets and model types without introducing target leakage.
- Collaborate with clinicians to define appropriate interventions, oversight, and outcome monitoring.
References
- UCI Machine Learning Repository. Diabetes 130-US Hospitals for Years 1999–2008. https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008
- Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records. BioMed Research International, 2014, 781670. https://doi.org/10.1155/2014/781670
Author
Adhilingam Adiswaran
Bellevue University — Data Science Capstone Project
