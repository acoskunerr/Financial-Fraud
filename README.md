



import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.special import expit
from scipy.stats import ks_2samp, wasserstein_distance

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score,
    matthews_corrcoef, brier_score_loss, roc_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from catboost import CatBoostClassifier, Pool


try:
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception as e:
    print("XGBoost yüklenemedi (atlanacak):", e)
    XGB_OK = False

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

VARS = ["DSRI","GMI","AQI","SGI","DEPI","SGAI","LVGI","TATA"]
THRESH = -1.78  


STRICT_BOUNDS = {
    "DSRI": (1.00, 1.30, 1.30, 1.70),
    "GMI" : (0.90, 1.10, 1.10, 1.40),
    "AQI" : (0.90, 1.10, 1.10, 1.40),
    "SGI" : (1.00, 1.20, 1.30, 1.80),
    "DEPI": (0.95, 1.05, 0.80, 0.95),
    "SGAI": (0.95, 1.05, 1.05, 1.20),
    "LVGI": (0.90, 1.10, 1.10, 1.30),
    "TATA": (-0.02, 0.02, 0.03, 0.06)
}
GLOBAL_MIN = np.array([min(STRICT_BOUNDS[k][0], STRICT_BOUNDS[k][2]) for k in VARS], dtype=np.float32)
GLOBAL_MAX = np.array([max(STRICT_BOUNDS[k][1], STRICT_BOUNDS[k][3]) for k in VARS], dtype=np.float32)


def mscore(df_like: pd.DataFrame) -> np.ndarray:
    x = df_like
    return (-4.84 + 0.920*x["DSRI"] + 0.528*x["GMI"] + 0.404*x["AQI"] +
            0.892*x["SGI"] + 0.115*x["DEPI"] - 0.172*x["SGAI"] +
            4.679*x["TATA"] - 0.327*x["LVGI"]).to_numpy()

def auc_pr(y_true, prob):
    p, r, _ = precision_recall_curve(y_true, prob)
    return auc(r, p)




def best_threshold(y, prob, metric="MCC"):
    ts = np.arange(0.05, 0.96, 0.01)
    best_t, best_s = 0.5, -1e9
    for t in ts:
        pred = (prob >= t).astype(int)
        if metric=="MCC":
            s = matthews_corrcoef(y, pred)
        elif metric=="F1":
            s = f1_score(y, pred)
        else:
            tp = ((y==1) & (pred==1)).sum()
            tn = ((y==0) & (pred==0)).sum()
            fp = ((y==0) & (pred==1)).sum()
            fn = ((y==1) & (pred==0)).sum()
            tpr = 0 if (tp+fn)==0 else tp/(tp+fn)
            tnr = 0 if (tn+fp)==0 else tn/(tn+fp)
            s = 0.5*(tpr+tnr)
        if s > best_s:
            best_s, best_t = s, t
    return best_t, best_s


def sample_prob_label_from_M(M_array, thresh=-1.78, tau=0.45, base=0.00):
    p = expit((M_array - thresh) / tau) + base
    p = np.clip(p, 1e-4, 1-1e-4)
    return (np.random.rand(len(M_array)) < p).astype(int)


def draw_trunc_normal(size, a, b, mean, sd):
    s = np.random.normal(mean, sd, size*2)
    s = s[(s>=a) & (s<=b)]
    if len(s) < size:
        s2 = np.random.uniform(a, b, size - len(s))
        s = np.concatenate([s, s2])
    return s[:size]



def sample_seed_with_overlap(
    n=35000, class_ratio=0.10, overlap=0.25, global_noise=0.06, label_flip=0.03
) -> pd.DataFrame:
    y_hint = (np.random.rand(n) < class_ratio).astype(int)  
    data = {}
    for k in VARS:
        a0,b0,a1,b1 = STRICT_BOUNDS[k]
        rng0 = b0 - a0; rng1 = b1 - a1
        b0_adj = b0 + overlap * rng0
        a1_adj = a1 - overlap * rng1
        mu0, sd0 = (a0+b0_adj)/2, max((b0_adj-a0)/6, 1e-6)
        mu1, sd1 = (a1_adj+b1)/2, max((b1-a1_adj)/6, 1e-6)

        out = np.empty(n, dtype=float)
        idx1, idx0 = np.where(y_hint==1)[0], np.where(y_hint==0)[0]
        out[idx1] = draw_trunc_normal(len(idx1), a1_adj, b1, mu1, sd1)
        out[idx0] = draw_trunc_normal(len(idx0), a0, b0_adj, mu0, sd0)
        out += np.random.normal(0.0, global_noise, size=n)
        gmin, gmax = min(a0,a1), max(b0,b1)
        out = np.clip(out, gmin, gmax)
        data[k] = out

    df = pd.DataFrame(data)
    df["M"] = mscore(df)
    df["y"] = sample_prob_label_from_M(df["M"].to_numpy(), thresh=THRESH, tau=0.45, base=0.00)
    if label_flip > 0:
        flip_idx = np.random.choice(len(df), size=int(len(df)*label_flip), replace=False)
        df.loc[flip_idx, "y"] = 1 - df.loc[flip_idx, "y"]
    return df

seed_df = sample_seed_with_overlap()
print("Seed n:", len(seed_df), "| y=1 oranı:", round(seed_df["y"].mean(),3))
print(seed_df["M"].describe(percentiles=[.1,.25,.5,.75,.9]))

class BeneishDS(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32).reshape(-1,1)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

LATENT_DIM, COND_DIM, K = 16, 1, len(VARS)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM + COND_DIM, 96),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.05),
            nn.Linear(96, 96),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(96, K)
        )
        self.register_buffer("gmin", torch.tensor(GLOBAL_MIN.reshape(1,-1)))
        self.register_buffer("gmax", torch.tensor(GLOBAL_MAX.reshape(1,-1)))
    def forward(self, z, y):
        h = torch.cat([z, y], dim=1)
        sig = torch.sigmoid(self.net(h))
        return self.gmin + sig*(self.gmax - self.gmin)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(K + COND_DIM, 96),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.15),
            nn.Linear(96, 48),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(48, 1)
        )
    def forward(self, x, y):
        h = torch.cat([x, y], dim=1)
        return self.net(h)

def mscore_penalty_torch(X_t: torch.Tensor, y_t: torch.Tensor, lam=2.0) -> torch.Tensor:
    df = pd.DataFrame(X_t.detach().cpu().numpy(), columns=VARS)
    M  = mscore(df)
    yv = y_t.detach().cpu().numpy().reshape(-1)
    pen = np.zeros_like(M, dtype=np.float32)
    pen[yv==1] = np.maximum(0.0, THRESH - M[yv==1])  # fraud → M yüksek olmalı
    pen[yv==0] = np.maximum(0.0, M[yv==0] - THRESH)  # clean → M düşük olmalı
    return torch.tensor(pen.mean()*lam, dtype=torch.float32, device=DEVICE)

def add_instance_noise(x, sigma=0.02):
    if sigma <= 0: return x
    return x + torch.randn_like(x) * sigma

def smooth_labels_like(tensor, target_one=True, eps=0.1):
    if target_one:  return torch.ones_like(tensor)*(1.0 - eps)
    else:           return torch.zeros_like(tensor)+eps

def train_fingan(seed_df, epochs=120, batch_size=256, lambda_m=2.0,
                 inst_noise=0.02, d_updates=1, g_updates=1, patience=15):
    Xs = seed_df[VARS].to_numpy().astype(np.float32)
    ys = seed_df["y"].to_numpy().astype(np.float32).reshape(-1,1)
    loader = DataLoader(BeneishDS(Xs, ys), batch_size=batch_size, shuffle=True, drop_last=True)

    G, D = Generator().to(DEVICE), Discriminator().to(DEVICE)
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5,0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5,0.999))
    bce_logits = nn.BCEWithLogitsLoss()

    best_gloss, wait = float("inf"), 0
    for ep in range(1, epochs+1):
        Dl = Gl = nb = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            for _ in range(d_updates):
                opt_D.zero_grad()
                xr = add_instance_noise(xb, inst_noise)
                loss_real = bce_logits( D(xr, yb), smooth_labels_like(torch.empty_like(yb), True, 0.1) )
                z = torch.randn(xb.size(0), LATENT_DIM, device=DEVICE)
                fake_x = G(z, yb).detach()
                xf = add_instance_noise(fake_x, inst_noise)
                loss_fake = bce_logits( D(xf, yb), smooth_labels_like(torch.empty_like(yb), False, 0.1) )

                loss_D = loss_real + loss_fake
                loss_D.backward(); opt_D.step()


            for _ in range(g_updates):
                opt_G.zero_grad()
                z = torch.randn(xb.size(0), LATENT_DIM, device=DEVICE)
                gen_x = G(z, yb)
                adv = bce_logits( D(gen_x, yb), smooth_labels_like(torch.empty_like(yb), True, 0.0) )
                pen = mscore_penalty_torch(gen_x, yb, lam=lambda_m)
                loss_G = adv + pen
                loss_G.backward(); opt_G.step()

            Dl += float(loss_D.item()); Gl += float(loss_G.item()); nb += 1

        gmean = Gl/nb
        if ep % 10 == 0: print(f"ep {ep:3d} | D={Dl/nb:.4f} | G={gmean:.4f}")
        if gmean < best_gloss - 1e-3:
            best_gloss = gmean; wait = 0; best_state = (G.state_dict(), D.state_dict())
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop at epoch {ep} (best G={best_gloss:.4f})")
                break

    G.load_state_dict(best_state[0]); D.load_state_dict(best_state[1])
    return G, D

Gnet, Dnet = train_fingan(seed_df, epochs=120, batch_size=256,
                          lambda_m=2.0, inst_noise=0.02, d_updates=1, g_updates=1, patience=15)

def generate_with_G(G, n=100_000, p_fraud=0.08):
    n1 = np.random.binomial(n, p_fraud); n0 = n - n1
    y_cond = np.concatenate([np.ones(n1), np.zeros(n0)]).astype(np.float32)
    y_t = torch.tensor(y_cond.reshape(-1,1), device=DEVICE)
    z = torch.randn(n, LATENT_DIM, device=DEVICE)
    with torch.no_grad():
        Xg = G(z, y_t).cpu().numpy()
    df = pd.DataFrame(Xg, columns=VARS)
    df["M"] = mscore(df)
    # Nihai etiketi PROBABILISTIC belirle (generator koşulluluğunu kırar → metrikler düşer)
    df["y"] = sample_prob_label_from_M(df["M"].to_numpy(), thresh=THRESH, tau=0.45, base=0.00)
    return df

syn_raw = generate_with_G(Gnet, n=100_000, p_fraud=0.08)

def in_range_mask(df):
    ok = np.ones(len(df), dtype=bool)
    for j,k in enumerate(VARS):
        ok &= (df[k].values >= GLOBAL_MIN[j]) & (df[k].values <= GLOBAL_MAX[j])
    return ok

def iqr_mask(x, k=3.0):
    q1, q3 = np.quantile(x, 0.25), np.quantile(x, 0.75)
    iqr = q3-q1
    lo, hi = q1 - k*iqr, q3 + k*iqr
    return (x >= lo) & (x <= hi)

ok = in_range_mask(syn_raw)
for v in VARS: ok &= iqr_mask(syn_raw[v].values, k=3.0)
syn = syn_raw[ok].reset_index(drop=True)
print("GAN üretim (filtreli) n:", len(syn), "| y=1 oranı:", round(syn["y"].mean(),3))

X = syn[VARS].to_numpy()
y = syn["y"].to_numpy()

X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
X_train, X_val,  y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.20, stratify=y_tmp, random_state=SEED)

classes = np.array([0,1])
cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = {0: cw[0], 1: cw[1]}
print("Class weights:", {k: float(v) for k,v in class_weight_dict.items()})



rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    min_samples_leaf=30,
    max_features="sqrt",
    bootstrap=True,
    oob_score=True,
    random_state=SEED,
    class_weight=class_weight_dict,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_val_prob = rf.predict_proba(X_val)[:,1]
thr_rf, _ = best_threshold(y_val, rf_val_prob, metric="MCC")

train_pool = Pool(X_train, y_train, feature_names=VARS)
val_pool   = Pool(X_val,   y_val,   feature_names=VARS)

cb = CatBoostClassifier(
    iterations=3000,
    learning_rate=0.015,
    depth=5,
    l2_leaf_reg=16.0,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=SEED,
    od_type="Iter",
    od_wait=200,
    class_weights=[class_weight_dict[0], class_weight_dict[1]],
    verbose=250
)
cb.fit(train_pool, eval_set=val_pool, use_best_model=True)
cb_val_prob = cb.predict_proba(val_pool)[:,1]
thr_cb, _ = best_threshold(y_val, cb_val_prob, metric="MCC")





svm_base = LinearSVC(C=1.0, class_weight="balanced", random_state=SEED)
svm = CalibratedClassifierCV(svm_base, method="isotonic", cv=5)
svm.fit(X_train, y_train)
svm_val_prob = svm.predict_proba(X_val)[:,1]
thr_svm, _ = best_threshold(y_val, svm_val_prob, metric="MCC")

xgb, thr_xgb = None, None
xgb_val_prob = None
if XGB_OK:
    pos_ratio = y_train.mean()
    scale_pos = (1 - pos_ratio) / max(pos_ratio, 1e-6)
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=5,
        reg_lambda=5.0,
        objective="binary:logistic",
        tree_method="hist",
        monotone_constraints="(1,1,1,1,1,-1,-1,1)",  # DSRI,GMI,AQI,SGI,DEPI,SGAI,LVGI,TATA
        scale_pos_weight=scale_pos,
        random_state=SEED,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_val_prob = xgb.predict_proba(X_val)[:,1]
    thr_xgb, _ = best_threshold(y_val, xgb_val_prob, metric="MCC")







def pack_report(name, y_true, prob, thr):
    pred = (prob >= thr).astype(int)
    return {
        "model": name,
        "ROC_AUC": roc_auc_score(y_true, prob),
        "AUC_PR" : auc_pr(y_true, prob),
        "MCC"    : matthews_corrcoef(y_true, pred),
        "F1"     : f1_score(y_true, pred),
        "Brier"  : brier_score_loss(y_true, prob),
        "Thr"    : float(thr)
    }

rf_test_prob  = rf.predict_proba(X_test)[:,1]
cb_test_prob  = cb.predict_proba(X_test)[:,1]
svm_test_prob = svm.predict_proba(X_test)[:,1]

rows = [
    pack_report("RandomForest", y_test, rf_test_prob, thr_rf),
    pack_report("CatBoost",     y_test, cb_test_prob, thr_cb),
    pack_report("LinearSVM+Calibr.", y_test, svm_test_prob, thr_svm),
]
if XGB_OK:
    xgb_test_prob = xgb.predict_proba(X_test)[:,1]
    rows.append(pack_report("XGBoost (monotone)", y_test, xgb_test_prob, thr_xgb))

metrics = pd.DataFrame(rows)
print("\n=== TEST METRİKLERİ (probabilistic labels, sızıntısız) ===")
print(metrics.to_string(index=False))

X_test_shift = X_test.copy()
sgicol = VARS.index("SGI")
X_test_shift[:, sgicol] = np.clip(X_test_shift[:, sgicol]*1.05, GLOBAL_MIN[sgicol], GLOBAL_MAX[sgicol])

rf_prob_shift  = rf.predict_proba(X_test_shift)[:,1]
cb_prob_shift  = cb.predict_proba(X_test_shift)[:,1]
svm_prob_shift = svm.predict_proba(X_test_shift)[:,1]



rows_shift = [
    pack_report("RF_shift+5%SGI",  y_test, rf_prob_shift,  thr_rf),
    pack_report("CB_shift+5%SGI",  y_test, cb_prob_shift,  thr_cb),
    pack_report("SVM_shift+5%SGI", y_test, svm_prob_shift, thr_svm),
]
if XGB_OK:
    xgb_prob_shift = xgb.predict_proba(X_test_shift)[:,1]
    rows_shift.append( pack_report("XGB_shift+5%SGI", y_test, xgb_prob_shift, thr_xgb) )

metrics_shift = pd.DataFrame(rows_shift)
print("\n--- Domain-shift (SGI +%5, test) ---")
print(metrics_shift.to_string(index=False))

rf_importance = pd.Series(rf.feature_importances_, index=VARS).sort_values(ascending=False)
plt.figure(figsize=(7,4))
rf_importance.plot(kind="bar", title="RF Feature Importance (Beneish)")
plt.ylabel("Importance"); plt.tight_layout(); plt.savefig("rf_feature_importance.png", dpi=300)

cb_importance = pd.Series(cb.get_feature_importance(Pool(X_val, y_val, feature_names=VARS)),
                          index=VARS).sort_values(ascending=False)
plt.figure(figsize=(7,4))
cb_importance.plot(kind="bar", color="orange", title="CatBoost Feature Importance (Beneish)")
plt.ylabel("Importance"); plt.tight_layout(); plt.savefig("cb_feature_importance.png", dpi=300)

    xgb_gain = pd.Series(xgb.feature_importances_, index=VARS).sort_values(ascending=False)
    plt.figure(figsize=(7,4))
    xgb_gain.plot(kind="bar", title="XGBoost Feature Importance (gain)")
    plt.ylabel("Importance"); plt.tight_layout(); plt.savefig("xgb_feature_importance.png", dpi=300)

def plot_roc_pr(y_true, prob, prefix):
    fpr, tpr, _ = roc_curve(y_true, prob)
    p, r, _ = precision_recall_curve(y_true, prob)
    plt.figure(); plt.plot(fpr, tpr); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"{prefix} ROC (test)")
    plt.tight_layout(); plt.savefig(f"{prefix.lower().replace(' ','_')}_roc_test.png", dpi=300)
    plt.figure(); plt.plot(r, p); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"{prefix} PR (test)")
    plt.tight_layout(); plt.savefig(f"{prefix.lower().replace(' ','_')}_pr_test.png", dpi=300)

plot_roc_pr(y_test, cb_test_prob, "CatBoost")
plot_roc_pr(y_test, rf_test_prob, "RandomForest")
plot_roc_pr(y_test, svm_test_prob, "LinearSVM+Calibr.")
if XGB_OK:
    plot_roc_pr(y_test, xgb_test_prob, "XGBoost (monotone)")

def plot_calibration(y_true, prob, prefix):
    prob_true, prob_pred = calibration_curve(y_true, prob, n_bins=10, strategy="quantile")
    plt.figure(); plt.plot(prob_pred, prob_true, marker="o"); plt.plot([0,1],[0,1],'--')
    plt.xlabel("Predicted probability"); plt.ylabel("True fraction positive")
    plt.title(f"Calibration Curve ({prefix}, test)"); plt.tight_layout(); plt.savefig(f"{prefix.lower().replace(' ','_')}_calibration.png", dpi=300)

plot_calibration(y_test, cb_test_prob, "CatBoost")
plot_calibration(y_test, svm_test_prob, "LinearSVM+Calibr.")
if XGB_OK:
    plot_calibration(y_test, xgb_test_prob, "XGBoost (monotone)")

def plot_mscore_hist(df, fname="fig_mscore_hist.png"):
    plt.figure(figsize=(6,4))
    plt.hist(df["M"], bins=60, alpha=0.8)
    plt.axvline(THRESH, linestyle="--")
    plt.title("Beneish M-Score Histogramı")
    plt.xlabel("M"); plt.ylabel("Frekans")
    plt.tight_layout(); plt.savefig(fname, dpi=300)

plot_mscore_hist(syn)

def plot_marginals_grid(df, fname="fig_marginals_grid.png"):
    n = len(VARS)
    cols = 4; rows = int(np.ceil(n/cols))
    plt.figure(figsize=(cols*3.3, rows*2.6))
    for i, v in enumerate(VARS, 1):
        plt.subplot(rows, cols, i)
        x0 = df.loc[df["y"]==0, v].values
        x1 = df.loc[df["y"]==1, v].values
        plt.hist(x0, bins=40, alpha=0.6, density=True, label="Non-fraud")
        plt.hist(x1, bins=40, alpha=0.6, density=True, label="Fraud")
        plt.title(v); plt.xlabel(v); plt.ylabel("Yoğunluk")
        if i==1: plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(fname, dpi=300)

plot_marginals_grid(syn)

def plot_corr_heatmap(df, fname):
    C = np.corrcoef(df[VARS].to_numpy().T)
    plt.figure(figsize=(6,5))
    im = plt.imshow(C, vmin=-1, vmax=1)
    plt.xticks(range(len(VARS)), VARS, rotation=45, ha="right")
    plt.yticks(range(len(VARS)), VARS)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Korelasyon Isı Haritası")
    plt.tight_layout(); plt.savefig(fname, dpi=300)

plot_corr_heatmap(seed_df, "fig_corr_seed.png")
plot_corr_heatmap(syn,     "fig_corr_gan.png")

def realism_marginal_table(seed, gan, fname_csv="realism_marginal_ks.csv", fname_fig="fig_ks_wasserstein.png"):
    rows = []
    for v in VARS:
        a = seed[v].values; b = gan[v].values
        ks = ks_2samp(a, b).statistic
        wd = wasserstein_distance(a, b)
        rows.append({"var": v, "KS": ks, "Wasserstein": wd})
    tab = pd.DataFrame(rows).sort_values("KS", ascending=False)
    tab.to_csv(fname_csv, index=False)

    x = np.arange(len(tab))
    plt.figure(figsize=(7,4))
    plt.bar(x-0.2, tab["KS"].values, width=0.4)
    plt.bar(x+0.2, tab["Wasserstein"].values, width=0.4)
    plt.xticks(x, tab["var"].values, rotation=45, ha="right")
    plt.title("Seed vs GAN Realism (KS & Wasserstein)")
    plt.ylabel("Değer")
    plt.tight_layout(); plt.savefig(fname_fig, dpi=300)
    return tab

realism_tab = realism_marginal_table(seed_df, syn)

def plot_pr_with_baseline(y_true, prob, fname="fig_pr_with_baseline.png"):
    p, r, _ = precision_recall_curve(y_true, prob)
    base = y_true.mean()
    plt.figure(figsize=(5.5,4))
    plt.plot(r, p)
    plt.axhline(base, linestyle="--")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("PR Eğrisi (+ sınıf payı baz çizgisi)")
    plt.tight_layout(); plt.savefig(fname, dpi=300)

plot_pr_with_baseline(y_test, cb_test_prob)

def plot_threshold_tuning(y_val, prob, fname="fig_threshold_tuning_cb.png"):
    ts = np.arange(0.02, 0.98, 0.02)
    mccs, f1s, bals = [], [], []
    for t in ts:
        pred = (prob >= t).astype(int)
        tp = ((y_val==1)&(pred==1)).sum()
        tn = ((y_val==0)&(pred==0)).sum()
        fp = ((y_val==0)&(pred==1)).sum()
        fn = ((y_val==1)&(pred==0)).sum()
        mccs.append(matthews_corrcoef(y_val, pred))
        f1s.append(f1_score(y_val, pred))
        tpr = tp/(tp+fn) if (tp+fn)>0 else 0.0
        tnr = tn/(tn+fp) if (tn+fp)>0 else 0.0
        bals.append(0.5*(tpr+tnr))
    plt.figure(figsize=(6,4))
    plt.plot(ts, mccs, label="MCC")
    plt.plot(ts, f1s,  label="F1")
    plt.plot(ts, bals, label="Balanced Acc")
    plt.xlabel("Eşik (t)"); plt.ylabel("Skor")
    plt.title("Eşik Duyarlılık (Validation, CatBoost)")
    plt.legend()
    plt.tight_layout(); plt.savefig(fname, dpi=300)

plot_threshold_tuning(y_val, cb_val_prob)


def plot_pca_seed_vs_gan(seed, gan, n_sample=5000, fname="fig_pca_seed_vs_gan.png"):
    s = seed.sample(min(n_sample, len(seed)), random_state=SEED)
    g = gan.sample(min(n_sample, len(gan)), random_state=SEED)
    Xp = np.vstack([s[VARS].to_numpy(), g[VARS].to_numpy()])
    lab = np.array([0]*len(s) + [1]*len(g))
    pca = PCA(n_components=2, random_state=SEED).fit(Xp)
    Z = pca.transform(Xp)
    plt.figure(figsize=(6,4.5))
    plt.scatter(Z[lab==0,0], Z[lab==0,1], s=8, alpha=0.5, label="Seed")
    plt.scatter(Z[lab==1,0], Z[lab==1,1], s=8, alpha=0.5, label="GAN")
    plt.title("PCA: Seed vs GAN")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend(markerscale=2)
    plt.tight_layout(); plt.savefig(fname, dpi=300)

plot_pca_seed_vs_gan(seed_df, syn)

def domain_shift_tornado(model, X_test, y_test, base_thr, fname="fig_domain_shift_tornado.png"):
    base_prob = model.predict_proba(X_test)[:,1]
    base_mcc = matthews_corrcoef(y_test, (base_prob>=base_thr).astype(int))
    diffs = []
    for j, v in enumerate(VARS):
        Xs = X_test.copy()
        Xs[:, j] = np.clip(Xs[:, j]*1.05, GLOBAL_MIN[j], GLOBAL_MAX[j])
        p = model.predict_proba(Xs)[:,1]
        mcc = matthews_corrcoef(y_test, (p>=base_thr).astype(int))
        diffs.append(mcc - base_mcc)
    order = np.argsort(diffs)
    plt.figure(figsize=(6,5))
    ytick = [VARS[i] for i in order]
    plt.barh(np.arange(len(VARS)), np.array(diffs)[order])
    plt.yticks(np.arange(len(VARS)), ytick)
    plt.axvline(0, linestyle="--")
    plt.title("Domain-shift Hassasiyeti (MCC farkı, +%5)")
    plt.xlabel("ΔMCC (shift - base)")
    plt.tight_layout(); plt.savefig(fname, dpi=300)

domain_shift_tornado(cb, X_test, y_test, thr_cb)


def plot_reliability_hist(y_true, prob, fname="fig_reliability_hist.png"):
    prob_true, prob_pred = calibration_curve(y_true, prob, n_bins=10, strategy="quantile")
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(prob_pred, prob_true, marker="o"); plt.plot([0,1],[0,1],'--')
    plt.xlabel("Tahmin olasılığı"); plt.ylabel("Gerçek pozitif oranı"); plt.title("Kalibrasyon Eğrisi")
    plt.subplot(1,2,2)
    plt.hist(prob, bins=20, alpha=0.9)
    plt.xlabel("Tahmin olasılığı"); plt.ylabel("Frekans"); plt.title("Olasılık Dağılımı (Test)")
    plt.tight_layout(); plt.savefig(fname, dpi=300)

plot_reliability_hist(y_test, cb_test_prob)

def plot_fingan_architecture(fname="fig_fingan_architecture.png"):
    plt.figure(figsize=(7.5,4.5))
    def box(x,y,w,h,text):
        plt.gca().add_patch(plt.Rectangle((x,y),w,h, fill=False, linewidth=1.5))
        plt.text(x+w/2, y+h/2, text, ha="center", va="center")
    def arrow(x1,y1,x2,y2):
        plt.annotate("", (x2,y2), (x1,y1), arrowprops=dict(arrowstyle="->"))
    box(0.2,0.55,0.18,0.2,"Noise z")
    box(0.2,0.15,0.18,0.2,"Label y")
    box(0.45,0.35,0.22,0.25,"Generator G\n(min–max proj.)")
    box(0.75,0.35,0.22,0.25,"Discriminator D")
    box(0.45,0.05,0.22,0.20,"M-Score\nPenalty")
    box(0.75,0.05,0.22,0.20,"BCE Loss")
    arrow(0.38,0.65,0.45,0.55)
    arrow(0.38,0.25,0.45,0.40)
    arrow(0.67,0.47,0.75,0.47)
    arrow(0.67,0.20,0.75,0.20)
    plt.axis("off"); plt.tight_layout(); plt.savefig(fname, dpi=300)

plot_fingan_architecture()

xgboost_(monotone)_calibration.png")




CWD = os.getcwd()
OUT = os.path.join(CWD, "figures")
os.makedirs(OUT, exist_ok=True)

expected_pngs = [
    # modeller
    "rf_feature_importance.png",
    "cb_feature_importance.png",
    "catboost_roc_test.png", "catboost_pr_test.png", "cb_calibration.png",
    "randomforest_roc_test.png", "randomforest_pr_test.png",
    "linearsvm+calibr._roc_test.png", "linearsvm+calibr._pr_test.png", "linearsvm+calibr._calibration.png",
    # xgb (varsa)
    "xgb_feature_importance.png",
    "xgboost_(monotone)_roc_test.png", "xgboost_(monotone)_pr_test.png", "xgboost_(monotone)_calibration.png",
    # realism & analiz
    "fig_mscore_hist.png", "fig_marginals_grid.png",
    "fig_corr_seed.png", "fig_corr_gan.png",
    "fig_ks_wasserstein.png", "fig_pr_with_baseline.png",
    "fig_threshold_tuning_cb.png", "fig_pca_seed_vs_gan.png",
    "fig_domain_shift_tornado.png", "fig_reliability_hist.png",
    "fig_fingan_architecture.png",
]

pngs = set(glob.glob("*.png")) | set(glob.glob("./*.png"))
for p in pngs:
    try:
        shutil.copy2(p, os.path.join(OUT, os.path.basename(p)))
    except Exception:
        pass

found = sorted(glob.glob(os.path.join(OUT, "*.png")))
print("\nGörseller şuraya kaydedildi:\n", OUT)
for fp in found:
    print(" -", os.path.basename(fp))


try:
    os.startfile(OUT)  # Windows
except Exception as e:
    print("Klasörü elle açın:", OUT, "| Hata:", e)


