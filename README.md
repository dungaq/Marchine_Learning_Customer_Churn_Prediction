
#  Telecommunication Customer Churn Prediction

This project investigates **customer churn prediction in the UK telecommunications sector** using **machine learning** and **behavioural science** approaches.
It was developed as part of the MSc Business Analytics programme.

 [Full Report (PDF)](Final%20Report.pdf)

---

##  Project Overview

* **Business Problem:** Churn rates in UK telecom reach up to **16%** in mobile and **10%** in broadband segments. Retaining customers is more cost-effective than acquiring new ones.
* **Objective:** Identify key churn drivers and build predictive models to inform targeted retention strategies.
* **Dataset:** 1,409 telecom customer records (from Kaggle), with 10 selected variables: demographics, service usage, and subscription details.

---

##  Methodology

1. **Exploratory Data Analysis (EDA):**

   * Boxplots, histograms, correlation heatmaps.
   * Key patterns: churners often have shorter tenure, higher monthly charges, and fewer dependents.

2. **Data Pre-processing:**

   * Removal of irrelevant features.
   * Outlier capping using IQR method.
   * Encoding categorical variables (e.g., contract, payment method, gender).
   * Class imbalance handled with **SMOTE**.

3. **Models Implemented:**

   * Logistic Regression.
   * Logistic Regression with Interaction Terms.
   * Decision Tree Classification.

4. **Evaluation Metrics:**

   * AUC, Accuracy, Precision, Recall, and F1 Score.

---

##  Key Results

* **Best Model:** Logistic Regression with Interaction Terms

  * **AUC = 0.881**
  * **F1 Score = 0.825**
  * **Recall = 87.3%** (best at capturing churners).

* **Main Predictors:**

  * Contract type (monthly contracts most churn-prone).
  * Tenure (longer tenure reduces churn).
  * Monthly charges (higher charges increase churn risk).
  * Number of dependents (households more stable, less churn).

* **Decision Tree:** Highest precision (82%) → useful when minimizing false positives is critical.

---

##  Marketing Insights & Recommendations

* **Promote Long-Term Contracts:** Incentivize upgrades from month-to-month to annual plans.
* **First-Year Retention:** Target onboarding campaigns within first 12 months.
* **Smart Pricing & Transparency:** Offer tiered plans and bill shock prevention.
* **Household Bundling:** Design family packages to leverage dependents’ loyalty.
* **Personalised Offers:** Use payment method + usage data for micro-segmentation.

---

##  Repository Structure

```
telecom-churn-prediction/
│── scripts/               # R scripts for EDA, preprocessing, models
│── outputs/               # Figures, plots, confusion matrices
│── Final Report.pdf        # Full academic report
│── README.md              # Project documentation (this file)
│── .gitignore             # Ignore unnecessary files
│── LICENSE                # License (MIT)
```

---

##  Skills Demonstrated

* R programming for machine learning (logistic regression, decision trees).
* Data preprocessing (outlier capping, encoding, SMOTE).
* Model evaluation & selection using AUC, F1, precision-recall.
* Translating analytics into actionable marketing strategies.
* Reproducible workflow with Git/GitHub.

---

