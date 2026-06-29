<div align="center">
  
  <h1>🛡️ Hybrid-LSTM-Stacking-PPM</h1>
  <p><strong>Leakage-Free Predictive Process Monitoring: Achieving SOTA Performance on BPIC 2012</strong></p>

  [![Python](https://img.shields.io/badge/Python-3.13-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-ee4c2c.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
  [![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E.svg?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
  [![PM4Py](https://img.shields.io/badge/PM4Py-Process_Mining-8a2be2.svg?style=flat-square)](https://pm4py.fit.fraunhofer.de/)

</div>

⚠️ Copyright Notice
Copyright (c) 2026 Kang Gyu Min. All rights reserved.

<br/>

This project is a **Predictive Process Monitoring (PPM)** system that predicts bottleneck occurrences in business processes using real-world financial log data from the **BPI Challenge 2012 (BPIC 2012)**. 

We successfully resolved the **Data Leakage** problem that frequently occurs in existing process mining research. By combining deep learning (LSTM) for processing time-series sequences and a large-scale Stacking Ensemble utilizing 45 machine learning models, we achieved a **State-of-The-Art (SOTA) prediction accuracy of 93.39%**.

---

## 🚀 1. SOTA Performance Highlights

Existing research often suffered from overfitting due to reliance on single algorithms or flawed validation methods. This project recorded overwhelming metrics through **Hybrid Embeddings and Meta-Ensembles**, even under strictly controlled environments designed to prevent data leakage.

<div align="center">
  <img src="visualization/double_metrics_table.png" alt="Metrics Table" width="800"/>
</div>

**Key Insights:**
- The Stacking Ensemble broke through the ceiling of `HistGBM` (the strongest single model at 92.86%), reaching **93.39%**.
- It secured an exceptionally high **Recall** of 85.5% in pinpointing bottlenecks, designed to never miss critical risks when deployed in actual business environments.

<div align="center">
  <img src="visualization/top15_accuracy_table.png" alt="Top 15 Models" width="800"/>
</div>

<br/>

---

## 🏗️ 2. Architecture & Pipeline: Complete Prevention of Data Leakage

Most previous studies used random K-Fold cross-validation while ignoring time-series characteristics, causing fatal data leakage where 'future events' contaminate 'past training'. This architecture fundamentally blocks this issue.

<div align="center">
  <img src="visualization/architecture_diagram.png" alt="Architecture Diagram" width="1000"/>
</div>

1. **Strict 8:2 Split**: We strictly separated the train and test data based on chronological order and Trace structures without breaking the flow of event logs, ensuring the training data is never contaminated by the test set.
2. **Hybrid Feature Merge**: 
   - **Deep Learning**: Utilized `LSTM` to read the sequence context of events and extract `Hidden States` as deep learning embeddings.
   - **Machine Learning**: Merged classical statistical features based on macro Domain Logic (e.g., total duration, event frequency, number of resources involved).
3. **SMOTE Oversampling**: Resolved the severe class imbalance of bottleneck cases compared to normal cases.

<br/>

---

## 📊 3. Data Structure (BPIC 2012)

The raw event log (`.xes`) consists of a hierarchical structure: `Case (Trace)` - `Event` - `Resource (Org)`. This project flattened this complex tree-structured transaction data and performed precise Feature Engineering to convert it into a 2D table format suitable for machine learning.

<div align="center">
  <img src="visualization/data_structure_erd.png" alt="Data Structure ERD" width="800"/>
</div>
*We strictly controlled at the code level to ensure the `Duration` information—which is the Target variable—did not leak into the training feature set.*

<br/>

---

## 🔍 4. Algorithm Analysis & Evaluation

To prove which specific algorithm is most suitable for the process mining domain, we performed an Exhaustive Search across **45 different machine learning algorithms (Tree, Linear, SVM, KNN, Naive Bayes, MLP, etc.)**.

### 📈 Performance Distribution by Algorithm Family (Boxplot)
As a result, **Tree-based ensemble models** showed overwhelmingly superior performance on process mining event log data. Conversely, Linear or Naive Bayes models struggled to decipher the complex business logic.
<div align="center">
  <img src="visualization/accuracy_boxplot.png" alt="Accuracy Boxplot" width="800"/>
</div>

### 🌊 Performance Density Estimation of 45 Models (KDE)
Density analysis revealed that the vast majority of standard models were trapped under an accuracy ceiling of `0.80 ~ 0.85`. Only top-tier Gradient Boosting models (GBM, HistGBM, etc.) broke through this barrier to reach the top percentile of `0.92+`.
<div align="center">
  <img src="visualization/accuracy_kde.png" alt="Accuracy KDE" width="800"/>
</div>

<br/>

---

## 🏆 5. Breaking the Limits: Single vs Ensemble

To achieve the absolute State-of-The-Art (SOTA) record, we applied a **Stacking Meta-Ensemble** technique that mutually compensates for the weaknesses of single models. A Logistic Regression Meta-learner re-learns the predictions output by the Top-K base models to make the final decision.

<div align="center">
  <img src="visualization/accuracy_bar.png" alt="Accuracy Bar Chart" width="800"/>
</div>

> **💡 Business Insight (Accuracy vs Speed)**
> The final `Stacking Ensemble (93.39%)` model in this study is a SOTA model that proves the absolute limit of academic predictive power. However, it is relatively heavy as it combines dozens of models. If deployed in a business environment where **Real-Time inference speed is critical** (e.g., Enterprise ERP monitoring systems), we propose the strategic flexibility of adopting the top single model, **`HistGBM (92.86%)`**, which is hundreds of times faster with only a minor 0.5% performance drop.

<br/>

---

## 🎯 6. Error Analysis (Confusion Matrix)

We analyzed the prediction results of the SOTA ensemble model using a Confusion Matrix. In predictive models, the most fatal error is a "False Negative" (misjudging an impending bottleneck as a normal case). Our model extremely suppressed this error, demonstrating highly stable density along the diagonal (True Positives & True Negatives).

<div align="center">
  <img src="visualization/confusion_matrix.png" alt="Confusion Matrix" width="800"/>
</div>

<br/>

---

## 💡 7. Business Value & Process Innovation (IE Perspective)

Beyond merely improving machine learning performance, this project holds significant value from the perspective of Industrial Engineering (IE) and practical Business Process Innovation (PI).

1. **Paradigm Shift from Reactive to Proactive**
   Traditional process mining and legacy analytics were limited to post-mortem analysis (finding the cause after a problem occurred). This system proactively warns of bottleneck probabilities (>93% accuracy) in the middle of a workflow, enabling preemptive responses.
2. **Data-Driven Diagnosis & Resource Optimization based on TOC**
   Instead of just learning sequences, the model hybridized field domain features like 'number of personnel' and 'rework frequency'. Upon a bottleneck warning, managers can immediately reallocate idle personnel or resources to preemptively resolve the bottleneck.
3. **Universal Scalability from Finance (ERP) to Manufacturing (MES)**
   The mathematical structure of a financial ERP log (where a Loan Application Case passes through various Departments) is identical to the flow of a manufacturing MES (where Material passes through Machines). Thus, this pipeline architecture possesses powerful scalability and can be immediately applied to bottleneck detection in global manufacturing/production workflows.

<br/>

---

## ⚙️ Quick Start

If you have a Python environment set up, you can easily reproduce the Leakage-Free pipeline exhaustive search with the commands below.

```bash
# 1. Clone the repository
git clone https://github.com/Yani-Studio/Hybrid-LSTM-Stacking-PPM.git
cd Hybrid-LSTM-Stacking-PPM

# 2. Install dependencies
pip install -r requirements.txt
# (or manually: pip install pm4py pandas numpy scikit-learn torch imbalanced-learn)

# 3. Run the exhaustive V4 pipeline
python src/run_v4_exhaustive.py
```

<br/>

---

## 📚 Appendix A: Detailed Model Configurations

The 45 single Base Models and 17 Ensemble Techniques used in the exhaustive search of this study are as follows:

- **Base Models (45 Types)**:
  - **Tree-based**: Random Forest, Gradient Boosting (GBM), HistGBM, AdaBoost, ExtraTrees, Bagging, Decision Tree (including hyperparameter variations)
  - **Linear / Distance**: Logistic Regression, Ridge, SGD, Passive Aggressive, Perceptron, LinearSVC, KNN (K=3, 5, 10, 20), Nearest Centroid
  - **Probabilistic / Discriminant**: GaussianNB, BernoulliNB, MultinomialNB, ComplementNB, LDA, QDA
  - **Neural Network**: Multi-Layer Perceptron (Shallow, Deep, Wide architectures)
- **Meta-Ensemble Techniques (17 Types)**:
  - **Voting**: Hard Voting, Soft Voting
  - **Stacking (Meta-Learners)**: Logistic Regression (Achieved Final SOTA), Random Forest, HistGBM, ExtraTrees, GBM, Ridge, SGD, Decision Tree, GaussianNB, KNN, LDA, BernoulliNB, AdaBoost, Bagging, Passive Aggressive

<br/>

---

## 🗄️ Appendix B: Data Source & References

This project was conducted based on the real-world financial log open data **BPI Challenge 2012**, which is widely used as a standard benchmark in the process mining field.

- **Data Source**: [BPI Challenge 2012 (4TU.ResearchData)](https://data.4tu.nl/articles/dataset/BPI_Challenge_2012/12689204)
- **Author**: van Dongen, B.F. (Eindhoven University of Technology)
- **Context**: Real-world loan application process logs (including event tracking and timestamps) from a Dutch financial institution.
