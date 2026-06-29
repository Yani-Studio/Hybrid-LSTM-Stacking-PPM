<div align="center">
  
  <h1>🛡️ Hybrid-LSTM-Stacking-PPM</h1>
  <p><strong>Leakage-Free Predictive Process Monitoring: Achieving SOTA Performance on BPIC 2012</strong></p>

  [![Python](https://img.shields.io/badge/Python-3.13-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-ee4c2c.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
  [![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E.svg?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
  [![PM4Py](https://img.shields.io/badge/PM4Py-Process_Mining-8a2be2.svg?style=flat-square)](https://pm4py.fit.fraunhofer.de/)

</div>

<br/>

이 프로젝트는 **BPI Challenge 2012 (BPIC 2012)** 실제 금융 로그 데이터를 활용하여 비즈니스 프로세스의 병목(Bottleneck) 현상을 예측하는 **Predictive Process Monitoring (PPM)** 시스템입니다. 

기존 프로세스 마이닝 연구들에서 빈번하게 발생하는 **데이터 누수(Data Leakage)** 문제를 구조적으로 완벽히 차단하였으며, 시계열 시퀀스를 처리하는 딥러닝(LSTM)과 45개의 머신러닝을 총동원한 대규모 스태킹(Stacking) 앙상블 기법을 융합해 **93.39%라는 SOTA(State-of-The-Art, 최고 수준) 예측 정확도**를 달성했습니다.

---

## 🚀 1. SOTA Performance Highlights

기존 연구들은 특정 알고리즘 하나에 의존하거나 검증 방식의 오류로 인해 성능이 과대적합(Overfitting)되는 경향이 있었습니다. 본 프로젝트는 극도로 제한된 통제 환경 속에서도 **하이브리드 임베딩(Hybrid Embedding)과 메타 앙상블(Meta-Ensemble)**을 통해 압도적인 지표를 기록했습니다.

<div align="center">
  <img src="visualization/double_metrics_table.png" alt="Metrics Table" width="800"/>
</div>

**핵심 인사이트:**
- 단일 모델 중 가장 강력했던 `HistGBM`의 한계치(92.86%)를 스태킹 앙상블이 **93.39%**로 돌파했습니다.
- 병목(Bottleneck) 현상을 정확히 짚어내는 **Recall(재현율)** 측면에서도 85.5%라는 매우 높은 수치를 확보하여, 실제 비즈니스에 투입되었을 때 리스크를 놓치지 않도록 설계되었습니다.

<div align="center">
  <img src="visualization/top15_accuracy_table.png" alt="Top 15 Models" width="800"/>
</div>

<br/>

---

## 🏗️ 2. Architecture & Pipeline: 완벽한 데이터 누수(Leakage) 차단

대부분의 기존 연구는 시계열 특성을 무시한 무작위 K-Fold 교차 검증을 사용하여, '미래의 이벤트'가 '과거의 학습'에 개입하는 치명적인 데이터 누수(Data Leakage)를 발생시켰습니다. 본 아키텍처는 이를 원천 차단합니다.

<div align="center">
  <img src="visualization/architecture_diagram.png" alt="Architecture Diagram" width="1000"/>
</div>

1. **엄격한 시점 분리 (Strict 8:2 Split)**: 이벤트 로그의 흐름을 깨지 않고 시간순과 케이스(Trace) 구조를 철저히 분리하여 테스트 데이터가 훈련에 오염되지 않도록 막았습니다.
2. **하이브리드 피처 융합 (Hybrid Feature Merge)**: 
   - **딥러닝**: `LSTM` 모델을 활용하여 이벤트의 순서(Sequence) 맥락을 읽어내고, 그 은닉 상태(Hidden States)를 딥러닝 임베딩으로 추출합니다.
   - **머신러닝**: 전체 소요 시간, 이벤트 발생 빈도, 담당자(Resource) 수 등 거시적인 도메인 지식(Domain Logic) 기반의 통계 피처를 병합합니다.
3. **SMOTE 오버샘플링**: 정상 케이스에 비해 현저히 적은 병목 케이스의 불균형을 해소합니다.

<br/>

---

## 📊 3. Data Structure (BPIC 2012)

원본 이벤트 로그(`.xes`)는 `Case(Trace)` - `Event` - `Resource(Org)`의 계층형 구조로 이루어져 있습니다. 본 프로젝트는 이 복잡한 트리 구조의 트랜잭션 데이터를 평탄화(Flatten)하여 머신러닝이 학습할 수 있는 2차원 테이블 형태로 정밀하게 가공(Feature Engineering)했습니다.

<div align="center">
  <img src="visualization/data_structure_erd.png" alt="Data Structure ERD" width="800"/>
</div>
*데이터의 정답지(Target)가 되는 `Duration` 정보가 피처 셋에 섞여 들어가는 것(누수)을 코드 레벨에서 완벽히 통제했습니다.*

<br/>

---

## 🔍 4. Algorithm Analysis & Evaluation

어떤 특정 알고리즘이 프로세스 마이닝 도메인에 가장 적합한지 증명하기 위해, 총 **45개의 서로 다른 머신러닝 알고리즘 (Tree, Linear, SVM, KNN, Naive Bayes, MLP 등)**을 전수조사(Exhaustive Search)했습니다.

### 📈 알고리즘 계열별 성능 분포 (Boxplot)
결과적으로 프로세스 마이닝 이벤트 로그 데이터에서는 **트리(Tree) 기반의 앙상블 모델들**이 압도적으로 우수한 성과를 보였습니다. 반면, 선형(Linear)이나 나이브 베이즈(Naive Bayes) 계열은 복잡한 비즈니스 로직을 푸는 데 한계를 보였습니다.
<div align="center">
  <img src="visualization/accuracy_boxplot.png" alt="Accuracy Boxplot" width="800"/>
</div>

### 🌊 45개 모델 성능 밀도 추정 (KDE)
밀도 분석 결과, 대다수의 보편적인 모델들이 `0.80 ~ 0.85` 정확도 구간에 갇혀(Ceiling) 있는 것을 확인할 수 있습니다. 오직 최상위 그레디언트 부스팅(GBM, HistGBM 등) 계열만이 이 한계를 뚫고 `0.92` 이상의 극상위권에 도달했습니다.
<div align="center">
  <img src="visualization/accuracy_kde.png" alt="Accuracy KDE" width="800"/>
</div>

<br/>

---

## 🏆 5. Breaking the Limits: Single vs Ensemble

학술적인 최상위 기록(SOTA)을 달성하기 위해, 단일 모델의 약점을 상호 보완하는 **스태킹(Stacking) 메타-앙상블** 기법을 적용했습니다. Top-K 개의 우수한 단일 모델들이 내놓은 예측값을 다시 Logistic Regression 메타 러너(Meta-learner)가 학습하여 최종 결정을 내립니다.

<div align="center">
  <img src="visualization/accuracy_bar.png" alt="Accuracy Bar Chart" width="800"/>
</div>

> **💡 비즈니스 관점에서의 인사이트 (Accuracy vs Speed)**
> 본 연구의 최종 `Stacking Ensemble (93.39%)` 모델은 학술적 예측력의 극한을 증명한 SOTA 모델입니다. 다만 수십 개의 모델이 결합되어 있어 다소 무겁습니다. 만약 기업의 ERP 모니터링 시스템 등 **실시간(Real-Time) 추론 속도가 생명인 비즈니스 환경에 배포한다면**, 속도가 수백 배 빠르면서도 성능 차이가 불과 0.5%밖에 나지 않는 단일 1등 모델 **`HistGBM (92.86%)`**을 채택하는 전략적 유연성을 제안합니다.

<br/>

---

## 🎯 6. Error Analysis (Confusion Matrix)

SOTA 앙상블 모델의 예측 결과를 혼동 행렬(Confusion Matrix)로 분석했습니다. 예측 모델에서 가장 치명적인 에러는 "병목 현상이 발생할 것인데, 정상이라고 오판하는 경우(False Negative)" 입니다. 본 모델은 이 에러를 극한으로 억제하며 매우 안정적인 대각선(True Positives & True Negatives) 밀집도를 보였습니다.

<div align="center">
  <img src="visualization/confusion_matrix.png" alt="Confusion Matrix" width="800"/>
</div>

<br/>

---

## 💡 7. Business Value & Process Innovation (산업공학적 의의)

본 프로젝트는 단순한 머신러닝 예측 성능 향상을 넘어, 산업공학(IE) 및 현업 비즈니스 프로세스 혁신(PI) 관점에서 매우 중요한 가치를 지닙니다.

1. **사후 대응(Reactive)에서 사전 예측(Proactive)으로의 패러다임 전환**
   기존의 레거시 분석이나 전통적 프로세스 마이닝은 문제가 발생한 후 원인을 찾는 사후 분석에 머물렀습니다. 본 시스템은 흐름의 중간에 병목(Bottleneck) 확률을 93% 이상의 정확도로 미리 경고하여, 선제적인 대응을 가능케 합니다.
2. **데이터 주도적 문제 진단 및 제약이론(TOC) 기반 리소스 최적화**
   단순히 순서(Sequence)만 학습한 것이 아니라 '담당자 수', '재작업 빈도' 등 현장의 도메인 피처를 하이브리드로 학습했습니다. 병목 경고가 발생하면, 관리자는 즉각적으로 해당 공정에 유휴 인력이나 자원(Resource)을 재배치하여 병목을 선제적으로 해소할 수 있습니다.
3. **금융(ERP)에서 제조(MES) 환경까지 아우르는 범용성**
   대출 신청(Case)이 부서(Resource)를 거치는 금융 ERP 로그의 수학적 구조는, 자재(Material)가 공정(Machine)을 거치는 제조업 MES의 흐름과 완전히 동일합니다. 즉, 이 파이프라인 아키텍처는 글로벌 제조/생산 기업의 자재 흐름 및 공정 워크플로우 병목 탐지에도 즉각적으로 적용할 수 있는 강력한 확장성을 가집니다.

<br/>

---

## ⚙️ Quick Start

파이썬 환경이 구축되어 있다면, 누구나 아래 명령어를 통해 누수 방지(Leakage-Free) 파이프라인 전수조사를 재현해 볼 수 있습니다.

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

본 연구에서 전수조사에 사용된 45개의 단일 베이스 모델(Base Models)과 17개의 앙상블 조합(Ensemble Techniques)은 다음과 같습니다.

- **Base Models (45종)**:
  - **Tree-based**: Random Forest, Gradient Boosting (GBM), HistGBM, AdaBoost, ExtraTrees, Bagging, Decision Tree (하이퍼파라미터 변형 포함)
  - **Linear / Distance**: Logistic Regression, Ridge, SGD, Passive Aggressive, Perceptron, LinearSVC, KNN (K=3, 5, 10, 20), Nearest Centroid
  - **Probabilistic / Discriminant**: GaussianNB, BernoulliNB, MultinomialNB, ComplementNB, LDA, QDA
  - **Neural Network**: Multi-Layer Perceptron (Shallow, Deep, Wide 아키텍처)
- **Meta-Ensemble Techniques (17종)**:
  - **Voting**: Hard Voting, Soft Voting
  - **Stacking (Meta-Learners)**: Logistic Regression (최종 SOTA 달성), Random Forest, HistGBM, ExtraTrees, GBM, Ridge, SGD, Decision Tree, GaussianNB, KNN, LDA, BernoulliNB, AdaBoost, Bagging, Passive Aggressive

<br/>

---

## 🗄️ Appendix B: Data Source & References

본 프로젝트는 프로세스 마이닝 분야의 표준 벤치마크로 가장 널리 쓰이는 **BPI Challenge 2012** 실제 금융 로그 오픈 데이터를 기반으로 진행되었습니다.

- **Data Source**: [BPI Challenge 2012 (4TU.ResearchData)](https://data.4tu.nl/articles/dataset/BPI_Challenge_2012/12689204)
- **Author**: van Dongen, B.F. (Eindhoven University of Technology)
- **Context**: 네덜란드 금융 기관의 실제 대출 신청(Loan Application) 프로세스 로그 (이벤트 트래킹 및 타임스탬프 포함)
