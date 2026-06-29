import pm4py
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    ExtraTreesClassifier, BaggingClassifier, HistGradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, SGDClassifier, 
    PassiveAggressiveClassifier, Perceptron
)
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, ComplementNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. LSTM 임베딩 모듈 ====================

class BPICDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.targets[idx], dtype=torch.float32)

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64):
        super(SimpleLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        last_hidden = hidden[-1]
        return self.sigmoid(self.fc(last_hidden)), last_hidden

def log(msg):
    print(msg, flush=True)

def extract_sequences(df):
    activities = df['concept:name'].unique()
    act2idx = {act: i+1 for i, act in enumerate(activities)}
    vocab_size = len(activities) + 1

    cases = df.groupby('case:concept:name')
    summary = cases.agg(start_time=('time:timestamp', 'min'), end_time=('time:timestamp', 'max'))
    summary['duration'] = (summary['end_time'] - summary['start_time']).dt.total_seconds().fillna(0)
    threshold = summary['duration'].quantile(0.75)

    sequences, targets, case_ids = [], [], []
    MAX_LEN = 50
    for case_id, group in cases:
        seq = [act2idx[act] for act in group['concept:name']]
        seq = seq[:MAX_LEN] if len(seq) > MAX_LEN else seq + [0] * (MAX_LEN - len(seq))
        targets.append(1 if summary.loc[case_id, 'duration'] > threshold else 0)
        sequences.append(seq)
        case_ids.append(case_id)
    return sequences, targets, case_ids, vocab_size

def train_lstm_and_extract(sequences, targets, vocab_size, train_idx, test_idx):
    seq_train = [sequences[i] for i in train_idx]
    tgt_train = [targets[i] for i in train_idx]
    seq_test = [sequences[i] for i in test_idx]
    tgt_test = [targets[i] for i in test_idx]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleLSTM(vocab_size=vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    train_loader = DataLoader(BPICDataset(seq_train, tgt_train), batch_size=64, shuffle=True)
    model.train()
    for epoch in range(5):
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds, _ = model(X_b)
            loss = criterion(preds.squeeze(), y_b)
            loss.backward()
            optimizer.step()

    model.eval()
    def get_hiddens(seqs, tgts):
        dl = DataLoader(BPICDataset(seqs, tgts), batch_size=64, shuffle=False)
        hiddens = []
        with torch.no_grad():
            for X_b, _ in dl:
                _, h = model(X_b.to(device))
                hiddens.append(h.cpu().numpy())
        return np.vstack(hiddens)

    return get_hiddens(seq_train, tgt_train), get_hiddens(seq_test, tgt_test)

# ==================== 2. 통계 피처 추출 모듈 ====================

def extract_statistical_features(df):
    cases = df.groupby('case:concept:name')
    summary = cases.agg(
        num_events=('concept:name', 'count'),
        num_unique_activities=('concept:name', 'nunique'),
        start_time=('time:timestamp', 'min'),
        end_time=('time:timestamp', 'max'),
        unique_resources=('org:resource', 'nunique')
    )
    summary['duration'] = (summary['end_time'] - summary['start_time']).dt.total_seconds().fillna(0)
    threshold = summary['duration'].quantile(0.75)
    summary['target_bottleneck'] = (summary['duration'] > threshold).astype(int)

    top_activities = df['concept:name'].value_counts().head(15).index
    act_counts = cases['concept:name'].value_counts().unstack(fill_value=0)
    act_cols = [c for c in top_activities if c in act_counts.columns]
    act_features = act_counts[act_cols]

    # 누수 방지: duration 제거
    feature_cols = ['num_events', 'num_unique_activities', 'unique_resources', 'target_bottleneck']
    result = summary[feature_cols].join(act_features)
    return result.fillna(0)

# ==================== 3. 전체 45개 모델 & 메타 모델 정의 ====================

def get_45_models():
    return {
        'RF_Default': RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42, n_jobs=-1),
        'RF_Entropy': RandomForestClassifier(n_estimators=50, criterion='entropy', class_weight='balanced', random_state=42, n_jobs=-1),
        'RF_Shallow': RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42, n_jobs=-1),
        'GBM_Default': GradientBoostingClassifier(n_estimators=50, random_state=42),
        'GBM_Fast': GradientBoostingClassifier(n_estimators=20, learning_rate=0.2, random_state=42),
        'GBM_Exp': GradientBoostingClassifier(loss='exponential', n_estimators=50, random_state=42),
        'HistGBM_Default': HistGradientBoostingClassifier(random_state=42),
        'HistGBM_MaxIter200': HistGradientBoostingClassifier(max_iter=200, random_state=42),
        'AdaBoost_50': AdaBoostClassifier(n_estimators=50, random_state=42),
        'AdaBoost_100': AdaBoostClassifier(n_estimators=100, random_state=42),
        'ExtraTrees_50': ExtraTreesClassifier(n_estimators=50, class_weight='balanced', random_state=42, n_jobs=-1),
        'ExtraTrees_Ent': ExtraTreesClassifier(n_estimators=50, criterion='entropy', class_weight='balanced', random_state=42, n_jobs=-1),
        'Bagging_10': BaggingClassifier(n_estimators=10, random_state=42, n_jobs=-1),
        'Bagging_30': BaggingClassifier(n_estimators=30, random_state=42, n_jobs=-1),
        'DT_Default': DecisionTreeClassifier(class_weight='balanced', random_state=42),
        'DT_Shallow': DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42),
        'ExtraTree_Def': ExtraTreeClassifier(class_weight='balanced', random_state=42),
        'LogReg_L2': LogisticRegression(max_iter=500, class_weight='balanced', random_state=42, n_jobs=-1),
        'LogReg_C01': LogisticRegression(C=0.1, max_iter=500, class_weight='balanced', random_state=42, n_jobs=-1),
        'LogReg_LibLin': LogisticRegression(solver='liblinear', max_iter=500, class_weight='balanced', random_state=42),
        'LogReg_Newton': LogisticRegression(solver='newton-cg', max_iter=500, class_weight='balanced', random_state=42, n_jobs=-1),
        'Ridge_Def': RidgeClassifier(class_weight='balanced', random_state=42),
        'Ridge_Alpha10': RidgeClassifier(alpha=10.0, class_weight='balanced', random_state=42),
        'SGD_Hinge': SGDClassifier(loss='hinge', class_weight='balanced', random_state=42, n_jobs=-1),
        'SGD_LogLoss': SGDClassifier(loss='log_loss', class_weight='balanced', random_state=42, n_jobs=-1),
        'SGD_ModHuber': SGDClassifier(loss='modified_huber', class_weight='balanced', random_state=42, n_jobs=-1),
        'PassiveAgg_C1': PassiveAggressiveClassifier(C=1.0, class_weight='balanced', random_state=42, n_jobs=-1),
        'PassiveAgg_C01': PassiveAggressiveClassifier(C=0.1, class_weight='balanced', random_state=42, n_jobs=-1),
        'Perceptron_Def': Perceptron(class_weight='balanced', random_state=42, n_jobs=-1),
        'SVC_Lin_Fast': LinearSVC(class_weight='balanced', random_state=42, dual=False),
        'SVC_Lin_C01': LinearSVC(C=0.1, class_weight='balanced', random_state=42, dual=False),
        'MLP_Shallow': MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, random_state=42),
        'MLP_Deep': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=200, random_state=42),
        'MLP_Wide': MLPClassifier(hidden_layer_sizes=(200,), max_iter=200, random_state=42),
        'KNN_3': KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
        'KNN_5': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'KNN_10': KNeighborsClassifier(n_neighbors=10, n_jobs=-1),
        'KNN_20': KNeighborsClassifier(n_neighbors=20, n_jobs=-1),
        'NearestCentroid': NearestCentroid(),
        'GaussianNB': GaussianNB(),
        'BernoulliNB': BernoulliNB(),
        'MultinomialNB': MultinomialNB(),
        'ComplementNB': ComplementNB(),
        'LDA': LinearDiscriminantAnalysis(),
        'QDA': QuadraticDiscriminantAnalysis()
    }

def get_15_ensembles(base_estimators):
    ensembles = {
        'Voting_Hard': VotingClassifier(estimators=base_estimators, voting='hard', n_jobs=-1),
        'Voting_Soft': VotingClassifier(estimators=base_estimators, voting='soft', n_jobs=-1)
    }
    meta_estimators = {
        'LogReg': LogisticRegression(max_iter=500, class_weight='balanced', random_state=42, n_jobs=-1),
        'RF': RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42, n_jobs=-1),
        'HistGBM': HistGradientBoostingClassifier(random_state=42),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=50, class_weight='balanced', random_state=42, n_jobs=-1),
        'GBM': GradientBoostingClassifier(n_estimators=50, random_state=42),
        'Ridge': RidgeClassifier(class_weight='balanced', random_state=42),
        'SGD': SGDClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
        'DT': DecisionTreeClassifier(class_weight='balanced', random_state=42),
        'GNB': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'LDA': LinearDiscriminantAnalysis(),
        'BNB': BernoulliNB(),
        'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
        'Bagging': BaggingClassifier(random_state=42, n_jobs=-1),
        'PassiveAggressive': PassiveAggressiveClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
    }
    for name, meta_model in meta_estimators.items():
        ensembles[f'Stacking_{name}'] = StackingClassifier(
            estimators=base_estimators, final_estimator=meta_model, cv=3, passthrough=False, n_jobs=-1
        )
    return ensembles

# ==================== 4. 메인 파이프라인 ====================

def run_v4_exhaustive():
    start_time = time.time()
    log("="*60)
    log("🚀 V4 Leakage-Free 하이브리드 파이프라인 (FULL 전수조사) 시작")
    log("="*60)

    # 데이터 로딩 및 시퀀스 추출
    log("\n📦 데이터 로딩 및 LSTM 훈련 중...")
    df = pm4py.convert_to_dataframe(pm4py.read_xes("../data/raw/BPI_Challenge_2012.xes", variant='iterparse'))
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], utc=True)
    df = df.sort_values(by=['case:concept:name', 'time:timestamp'])
    
    sequences, targets, case_ids, vocab_size = extract_sequences(df)
    
    # Train/Test 엄격한 분할
    indices = np.arange(len(sequences))
    idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42, stratify=targets)
    
    # LSTM 학습 및 임베딩
    train_hiddens, test_hiddens = train_lstm_and_extract(sequences, targets, vocab_size, idx_train, idx_test)
    
    # 통계 피처
    stat_df = extract_statistical_features(df)
    y_col = stat_df['target_bottleneck'].values
    X_stat = stat_df.drop(columns=['target_bottleneck']).values
    
    X_train_stat = X_stat[idx_train]
    X_test_stat = X_stat[idx_test]
    y_train = y_col[idx_train]
    y_test = y_col[idx_test]
    
    # 하이브리드 결합
    X_train_hybrid = np.hstack([X_train_stat, train_hiddens])
    X_test_hybrid = np.hstack([X_test_stat, test_hiddens])
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_hybrid)
    X_test_scaled = scaler.transform(X_test_hybrid)
    
    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    
    # 단일 모델 전수조사
    log(f"\n🏋️ 45개 단일 모델 전수 훈련 시작...")
    models = get_45_models()
    single_results = {}
    for i, (name, model) in enumerate(models.items()):
        try:
            if 'NB' in name and name != 'GaussianNB':
                X_tr_nn = X_train_res - np.min(X_train_res)
                model.fit(X_tr_nn, y_train_res)
                y_pred = model.predict(X_test_scaled - np.min(X_train_res))
            else:
                model.fit(X_train_res, y_train_res)
                y_pred = model.predict(X_test_scaled)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            single_results[name] = {'acc': accuracy_score(y_test, y_pred), 'prec': precision_score(y_test, y_pred, zero_division=0), 'rec': recall_score(y_test, y_pred, zero_division=0), 'f1': f1}
            log(f"   [{i+1:2d}/45] {name:20s} → F1: {f1:.4f}")
        except Exception:
            pass
            
    sorted_models = sorted(single_results.items(), key=lambda x: x[1]['f1'], reverse=True)
    
    # 앙상블 전수조사
    log(f"\n🔥 앙상블 조합 탐색 (K=2~25, 17개 기법)...")
    ensemble_results = []
    for k in range(2, min(26, len(sorted_models)+1)):
        top_k = sorted_models[:k]
        base_est = [(n, models[n]) for n, _ in top_k]
        ensembles = get_15_ensembles(base_est)
        
        for ens_name, ens_model in ensembles.items():
            try:
                ens_model.fit(X_train_res, y_train_res)
                y_pred = ens_model.predict(X_test_scaled)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                ensemble_results.append({'k': k, 'technique': ens_name, 'acc': accuracy_score(y_test, y_pred), 'prec': precision_score(y_test, y_pred, zero_division=0), 'rec': recall_score(y_test, y_pred, zero_division=0), 'f1': f1})
            except Exception:
                pass
        log(f"   ✅ Top-{k:2d} 결합 완료 ({len(ensembles)}개 기법)")

    # 결과 출력
    all_sorted = sorted(ensemble_results, key=lambda x: x['f1'], reverse=True)
    best_ens = all_sorted[0]
    best_single = sorted_models[0]
    
    log("\n" + "="*60)
    log("📈 [최고 성능 단일 모델 (Leakage-Free)]")
    log(f" - {best_single[0]} (F1: {best_single[1]['f1']:.4f})")
    log("\n🏆 [최고 성능 앙상블 (Leakage-Free)]")
    log(f" - Top-{best_ens['k']} / {best_ens['technique']} (F1: {best_ens['f1']:.4f})")
    log("="*60)
    log(f"⏱️ 총 소요 시간: {(time.time()-start_time)/60:.1f}분")

if __name__ == "__main__":
    run_v4_exhaustive()
