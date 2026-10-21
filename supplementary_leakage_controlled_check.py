"""
Supplementary File S3 - leakage-controlled validation of the stacked ensemble (Section 4.8).

Differences from the executed notebook:
  (i)   the dataset is split BEFORE scaling and K-means (both fitted on training data only);
  (ii)  Optuna objectives (25 trials per learner) are computed on an inner 75/25 validation split
        of the TRAINING data (SMOTEENN applied to the inner-training part only);
  (iii) the decision threshold is chosen on 5-fold cross-validated meta-learner predictions of the
        TRAINING data (maximize F1 subject to accuracy >= 0.85); the test split is used once.
Single seed (42). Reported run: threshold 0.255, test accuracy 0.8715, precision 0.3695, recall 0.3676,
F1 0.3686, ROC-AUC 0.8123; test-oracle threshold 0.245 gives F1 0.3810.
Set DATA_PATH to customer_churn_business_dataset.csv. Requires: scikit-learn, imbalanced-learn,
xgboost, lightgbm, catboost, optuna, pandas, numpy.
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.combine import SMOTEENN
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
RS = 42
df = pd.read_csv('customer_churn_business_dataset.csv')
d = df.copy()
d['engagement_score'] = (d['monthly_logins']*d['weekly_active_days'])/(d['tenure_months']+1)
d['support_burden'] = d['support_tickets']/(d['tenure_months']+1)
d['revenue_per_login'] = d['total_revenue']/(d['monthly_logins']+1)
d['low_satisfaction_flag'] = ((d['csat_score']<=2)&(d['nps_score']<=0)).astype(int)
cat = ['gender','country','city','customer_segment','signup_channel','contract_type','payment_method','discount_applied','price_increase_last_3m','complaint_type','survey_response']
num = ['age','tenure_months','monthly_logins','weekly_active_days','avg_session_time','features_used','usage_growth_rate','last_login_days_ago','monthly_fee','total_revenue','payment_failures','support_tickets','avg_resolution_time','csat_score','escalations','email_open_rate','marketing_click_rate','nps_score','referral_count','engagement_score','support_burden','revenue_per_login','low_satisfaction_flag']
for c in cat:
    d[c] = LabelEncoder().fit_transform(d[c].astype(str))
feat = num+cat
X = d[feat]; y = d['churn']
# split FIRST, then scale/cluster on training only
Xtr_raw, Xte_raw, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
sc = StandardScaler().fit(Xtr_raw)
Xtr = pd.DataFrame(sc.transform(Xtr_raw), columns=feat, index=Xtr_raw.index)
Xte = pd.DataFrame(sc.transform(Xte_raw), columns=feat, index=Xte_raw.index)
km = KMeans(4, random_state=RS, n_init=10).fit(Xtr)
Xtr['cluster'] = km.predict(Xtr); Xte['cluster'] = km.predict(Xte)
Xtr = Xtr.reset_index(drop=True); ytr = ytr.reset_index(drop=True)

# hyperparameter tuning on an inner validation split of TRAIN only
Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.25, stratify=ytr, random_state=RS)
Xa_se, ya_se = SMOTEENN(random_state=RS).fit_resample(Xa, ya)
def obj_x(t):
    p = dict(n_estimators=t.suggest_int('n_estimators',100,600), max_depth=t.suggest_int('max_depth',3,10),
             learning_rate=t.suggest_float('learning_rate',0.005,0.3,log=True), subsample=t.suggest_float('subsample',0.5,1.0),
             colsample_bytree=t.suggest_float('colsample_bytree',0.5,1.0), min_child_weight=t.suggest_int('min_child_weight',1,15),
             reg_alpha=t.suggest_float('reg_alpha',1e-3,10,log=True), reg_lambda=t.suggest_float('reg_lambda',1e-3,10,log=True),
             eval_metric='logloss', random_state=RS)
    m = xgb.XGBClassifier(**p).fit(Xa_se, ya_se)
    return roc_auc_score(yv, m.predict_proba(Xv)[:,1])
def obj_l(t):
    p = dict(n_estimators=t.suggest_int('n_estimators',100,600), max_depth=t.suggest_int('max_depth',3,10),
             learning_rate=t.suggest_float('learning_rate',0.005,0.3,log=True), subsample=t.suggest_float('subsample',0.5,1.0),
             colsample_bytree=t.suggest_float('colsample_bytree',0.5,1.0), num_leaves=t.suggest_int('num_leaves',15,127),
             random_state=RS, verbose=-1)
    m = lgb.LGBMClassifier(**p).fit(Xa_se, ya_se)
    return roc_auc_score(yv, m.predict_proba(Xv)[:,1])
sx = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RS)); sx.optimize(obj_x, n_trials=25)
sl = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RS)); sl.optimize(obj_l, n_trials=25)
print('inner-val AUC xgb/lgb:', sx.best_value, sl.best_value)

names = ['xgb','lgb','cb','rf','lr']
def mk():
    return {'xgb': xgb.XGBClassifier(**sx.best_params, eval_metric='logloss', random_state=RS),
            'lgb': lgb.LGBMClassifier(**sl.best_params, random_state=RS, verbose=-1),
            'cb': CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, random_state=RS, verbose=0),
            'rf': RandomForestClassifier(n_estimators=400, max_depth=10, random_state=RS, n_jobs=-1),
            'lr': LogisticRegression(max_iter=2000, random_state=RS)}
skf = StratifiedKFold(5, shuffle=True, random_state=RS)
oof = {n: np.zeros(len(Xtr)) for n in names}; tp = {n: [] for n in names}
for tr, va in skf.split(Xtr, ytr):
    Xs, ys = SMOTEENN(random_state=RS).fit_resample(Xtr.iloc[tr], ytr.iloc[tr])
    for n, m in mk().items():
        m.fit(Xs, ys); oof[n][va] = m.predict_proba(Xtr.iloc[va])[:,1]; tp[n].append(m.predict_proba(Xte)[:,1])
MX = np.vstack([oof[n] for n in names]).T
MT = np.vstack([np.mean(tp[n],axis=0) for n in names]).T
meta = lgb.LGBMClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=RS, verbose=-1)
# threshold chosen on second-level cross-validated predictions of TRAIN data only
meta_oof = cross_val_predict(meta, MX, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=RS), method='predict_proba')[:,1]
meta.fit(MX, ytr); pte = meta.predict_proba(MT)[:,1]
best = None
for t in np.arange(0.05, 0.98, 0.005):
    pr = (meta_oof >= t).astype(int)
    a, f = accuracy_score(ytr, pr), f1_score(ytr, pr, zero_division=0)
    if a >= 0.85 and (best is None or f > best[1]): best = (t, f, a)
print('threshold from train-OOF:', best)
t = best[0]; pr = (pte >= t).astype(int)
print('TEST  acc %.4f prec %.4f rec %.4f f1 %.4f auc %.4f' % (accuracy_score(yte,pr), precision_score(yte,pr,zero_division=0), recall_score(yte,pr), f1_score(yte,pr), roc_auc_score(yte,pte)))
# oracle (test-tuned) threshold for the same model, for comparison
bo = None
for tt in np.arange(0.05,0.98,0.005):
    p2=(pte>=tt).astype(int); a,f=accuracy_score(yte,p2),f1_score(yte,p2,zero_division=0)
    if a>=0.85 and (bo is None or f>bo[1]): bo=(tt,f,a)
print('oracle test-tuned thr:', bo)
