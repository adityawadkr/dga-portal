"""
7-Class PINN Fault Classifier for DGA
=======================================
Physics-Informed Neural Network classifying transformer faults into
7 IEEE C57.104 / IEC 60599 classes with physics constraints.

Classes: Normal, PD, D1, D2, T1, T2, T3
Physics constraints per fault type:
  - D2 (Arcing):     C2H2/C2H4 > 1, high H2
  - D1 (Sparking):   C2H2/C2H4 in [0.1, 1]
  - T3 (>700°C):     C2H4/C2H6 > 4, C2H2/C2H4 < 0.1
  - T2 (300-700°C):  C2H4/C2H6 in [1, 4]
  - T1 (<300°C):     CH4/H2 > 1, C2H4/C2H6 < 1
  - PD:              H2 dominant, very low C2H2/C2H4
  - Normal:          All gases low

Author: Aditya Wadkar
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score)

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 1. FAULT PROFILES (IEEE C57.104 / IEC 60599)
# ============================================================

FAULT_PROFILES = {
    'Normal': {
        'H2': (5, 50), 'CH4': (5, 30), 'C2H6': (5, 30),
        'C2H4': (5, 20), 'C2H2': (0, 5), 'CO': (100, 400), 'CO2': (1000, 3000),
    },
    'PD': {
        'H2': (100, 500), 'CH4': (10, 50), 'C2H6': (5, 20),
        'C2H4': (5, 20), 'C2H2': (5, 30), 'CO': (50, 200), 'CO2': (500, 2000),
    },
    'D1': {
        'H2': (100, 400), 'CH4': (20, 100), 'C2H6': (10, 40),
        'C2H4': (20, 80), 'C2H2': (50, 200), 'CO': (50, 200), 'CO2': (500, 2000),
    },
    'D2': {
        'H2': (200, 800), 'CH4': (50, 200), 'C2H6': (20, 80),
        'C2H4': (50, 200), 'C2H2': (150, 600), 'CO': (100, 500), 'CO2': (1000, 4000),
    },
    'T1': {
        'H2': (50, 200), 'CH4': (100, 400), 'C2H6': (100, 400),
        'C2H4': (20, 80), 'C2H2': (0, 10), 'CO': (200, 800), 'CO2': (2000, 6000),
    },
    'T2': {
        'H2': (100, 400), 'CH4': (200, 600), 'C2H6': (50, 200),
        'C2H4': (100, 400), 'C2H2': (0, 20), 'CO': (300, 1000), 'CO2': (3000, 8000),
    },
    'T3': {
        'H2': (200, 600), 'CH4': (100, 400), 'C2H6': (20, 100),
        'C2H4': (300, 800), 'C2H2': (10, 80), 'CO': (500, 1500), 'CO2': (4000, 10000),
    },
}

CLASS_NAMES = ['Normal', 'PD', 'D1', 'D2', 'T1', 'T2', 'T3']
N_CLASSES = 7


# ============================================================
# 2. DATA GENERATION
# ============================================================

def generate_sample(fault_type, noise=0.15):
    """Generate one realistic DGA sample."""
    profile = FAULT_PROFILES[fault_type]
    sample = {}
    for gas in ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2', 'CO', 'CO2']:
        lo, hi = profile[gas]
        val = np.random.normal((lo + hi) / 2, (hi - lo) / 4)
        val *= (1 + np.random.uniform(-noise, noise))
        sample[gas] = max(0, val)

    # Correlated noise
    if fault_type in ['D1', 'D2']:
        cf = np.random.uniform(0.8, 1.2)
        sample['H2'] *= cf; sample['C2H2'] *= cf
    if fault_type in ['T1', 'T2', 'T3']:
        cf = np.random.uniform(0.8, 1.2)
        sample['CH4'] *= cf; sample['C2H4'] *= cf

    sample['fault_type'] = fault_type
    return sample


def generate_dataset(samples_per_class=500):
    """Generate balanced dataset."""
    np.random.seed(42)
    data = []
    for ft in CLASS_NAMES:
        for _ in range(samples_per_class):
            data.append(generate_sample(ft))
    df = pd.DataFrame(data).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def compute_features(df):
    """Engineer features including IEC/Rogers ratios and Duval coordinates."""
    eps = 0.001
    out = df.copy()
    out['R1_CH4_H2'] = df['CH4'] / (df['H2'] + eps)
    out['R2_C2H2_C2H4'] = df['C2H2'] / (df['C2H4'] + eps)
    out['R3_C2H4_C2H6'] = df['C2H4'] / (df['C2H6'] + eps)
    out['R4_C2H2_CH4'] = df['C2H2'] / (df['CH4'] + eps)
    out['R5_CO2_CO'] = df['CO2'] / (df['CO'] + eps)
    out['TCG'] = df['H2'] + df['CH4'] + df['C2H6'] + df['C2H4'] + df['C2H2'] + df['CO']

    total_tri = df['CH4'] + df['C2H4'] + df['C2H2'] + eps
    out['Duval_CH4'] = df['CH4'] / total_tri * 100
    out['Duval_C2H4'] = df['C2H4'] / total_tri * 100
    out['Duval_C2H2'] = df['C2H2'] / total_tri * 100

    # Log features
    for g in ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2', 'CO', 'CO2']:
        out[f'log_{g}'] = np.log1p(df[g])

    return out


FEATURE_COLS = ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2', 'CO', 'CO2',
                'R1_CH4_H2', 'R2_C2H2_C2H4', 'R3_C2H4_C2H6',
                'R4_C2H2_CH4', 'R5_CO2_CO', 'TCG',
                'Duval_CH4', 'Duval_C2H4', 'Duval_C2H2',
                'log_H2', 'log_CH4', 'log_C2H6', 'log_C2H4',
                'log_C2H2', 'log_CO', 'log_CO2']


# ============================================================
# 3. PINN MODEL
# ============================================================

class DGAPhysicsEncoder(nn.Module):
    """
    Encode DGA physics as soft indicators per fault class.
    
    Each fault class has known gas ratio signatures from IEEE/IEC:
      - D2: C2H2/C2H4 > 1 (arcing)
      - T3: C2H4/C2H6 > 4 (high temp thermal)
      - PD: H2 >> CH4 (partial discharge)
      etc.
    
    This module outputs physics-based logits for each class.
    """

    def __init__(self):
        super().__init__()
        # Learnable thresholds for gas ratio boundaries
        self.r_thresh = nn.ParameterDict({
            'c2h2_c2h4_d2': nn.Parameter(torch.tensor(1.0)),    # D2 threshold
            'c2h2_c2h4_d1': nn.Parameter(torch.tensor(0.1)),    # D1 lower
            'c2h4_c2h6_t3': nn.Parameter(torch.tensor(4.0)),    # T3 threshold
            'c2h4_c2h6_t2': nn.Parameter(torch.tensor(1.0)),    # T2 threshold
            'ch4_h2_t1': nn.Parameter(torch.tensor(1.0)),       # T1 threshold
        })
        # Learnable weights for physics indicators
        self.indicator_weight = nn.Parameter(torch.ones(N_CLASSES))

    def forward(self, gases):
        """
        Args: gases (B, 7) — [H2, CH4, C2H6, C2H4, C2H2, CO, CO2]
        Returns: physics_logits (B, 7) — one per class
        """
        eps = 0.001
        H2, CH4, C2H6, C2H4, C2H2, CO, CO2 = gases.unbind(dim=1)

        r_c2h2_c2h4 = C2H2 / (C2H4 + eps)
        r_c2h4_c2h6 = C2H4 / (C2H6 + eps)
        r_ch4_h2 = CH4 / (H2 + eps)
        tcg = H2 + CH4 + C2H6 + C2H4 + C2H2 + CO

        B = gases.size(0)
        logits = torch.zeros(B, N_CLASSES, device=gases.device)

        # Normal: all gases low → TCG < 500
        logits[:, 0] = torch.sigmoid(5.0 * (500 - tcg) / 500)

        # PD: H2 dominant, low C2H2
        logits[:, 1] = torch.sigmoid(3.0 * (H2 / (tcg + eps) - 0.3)) * \
                       torch.sigmoid(3.0 * (0.1 - r_c2h2_c2h4))

        # D1: moderate C2H2/C2H4
        logits[:, 2] = torch.sigmoid(3.0 * (r_c2h2_c2h4 - self.r_thresh['c2h2_c2h4_d1'])) * \
                       torch.sigmoid(3.0 * (self.r_thresh['c2h2_c2h4_d2'] - r_c2h2_c2h4))

        # D2: high C2H2/C2H4
        logits[:, 3] = torch.sigmoid(3.0 * (r_c2h2_c2h4 - self.r_thresh['c2h2_c2h4_d2']))

        # T1: CH4/H2 > 1, low C2H4/C2H6
        logits[:, 4] = torch.sigmoid(3.0 * (r_ch4_h2 - self.r_thresh['ch4_h2_t1'])) * \
                       torch.sigmoid(3.0 * (self.r_thresh['c2h4_c2h6_t2'] - r_c2h4_c2h6))

        # T2: moderate C2H4/C2H6
        logits[:, 5] = torch.sigmoid(3.0 * (r_c2h4_c2h6 - self.r_thresh['c2h4_c2h6_t2'])) * \
                       torch.sigmoid(3.0 * (self.r_thresh['c2h4_c2h6_t3'] - r_c2h4_c2h6))

        # T3: high C2H4/C2H6
        logits[:, 6] = torch.sigmoid(3.0 * (r_c2h4_c2h6 - self.r_thresh['c2h4_c2h6_t3']))

        return logits * torch.softmax(self.indicator_weight, dim=0).unsqueeze(0)


class PINN_7Class(nn.Module):
    """
    7-Class PINN for DGA fault classification.
    
    Architecture:
        gases → PhysicsEncoder → physics_logits (7)
        [features + physics_logits] → MLP → nn_logits (7)
        final = (1-gate) × physics_logits + gate × nn_logits
    """

    def __init__(self, n_features=23, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.physics = DGAPhysicsEncoder()

        input_dim = n_features + N_CLASSES  # features + physics logits
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, N_CLASSES),
        )
        self.residual_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, features, gases):
        """
        Returns:
            logits: (B, 7)
            phys_logits: (B, 7)
        """
        phys_logits = self.physics(gases)
        x = torch.cat([features, phys_logits], dim=1)
        nn_logits = self.net(x)
        gate = torch.sigmoid(self.residual_gate)
        logits = (1 - gate) * phys_logits + gate * nn_logits
        return logits, phys_logits


# ============================================================
# 4. PHYSICS-INFORMED LOSS
# ============================================================

class FocalPhysicsLoss(nn.Module):
    """
    Focal loss + physics consistency regularization.
    
    Focal loss handles class imbalance better than cross-entropy.
    Physics loss ensures predictions align with known gas signatures.
    """

    def __init__(self, gamma=2.0, lam_phys=0.2):
        super().__init__()
        self.gamma = gamma
        self.lam_phys = lam_phys

    def focal_loss(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

    def forward(self, logits, targets, phys_logits):
        loss_focal = self.focal_loss(logits, targets)

        # Physics consistency: physics logits should agree with true class
        loss_phys = self.focal_loss(phys_logits * 10, targets)  # Scale up physics logits

        total = loss_focal + self.lam_phys * loss_phys
        return total, {
            'focal': loss_focal.item(),
            'physics': loss_phys.item(),
        }


# ============================================================
# 5. TRAINING
# ============================================================

def prepare_data():
    """Generate data, compute features, split."""
    df = generate_dataset(samples_per_class=500)
    feat_df = compute_features(df)

    X = feat_df[FEATURE_COLS].values.astype(np.float32)
    gases = df[['H2','CH4','C2H6','C2H4','C2H2','CO','CO2']].values.astype(np.float32)

    le = LabelEncoder()
    le.fit(CLASS_NAMES)
    y = le.transform(df['fault_type'].values)

    X = np.nan_to_num(X, nan=0, posinf=100, neginf=-100)

    idx = np.arange(len(X))
    tr_idx, tmp_idx = train_test_split(idx, test_size=0.3, stratify=y, random_state=42)
    val_idx, te_idx = train_test_split(tmp_idx, test_size=0.5, stratify=y[tmp_idx], random_state=42)

    scaler = StandardScaler()
    X[tr_idx] = scaler.fit_transform(X[tr_idx])
    X[val_idx] = scaler.transform(X[val_idx])
    X[te_idx] = scaler.transform(X[te_idx])

    print(f"  Data: {len(df)} samples, {N_CLASSES} classes")
    print(f"  Splits: train={len(tr_idx)}, val={len(val_idx)}, test={len(te_idx)}")
    print(f"  Features: {len(FEATURE_COLS)}")

    return X, gases, y, tr_idx, val_idx, te_idx, scaler, le


def train_model(X, gases, y, tr_idx, val_idx,
                epochs=200, lr=1e-3, batch_size=128):
    """Train PINN 7-class model."""
    n_feat = X.shape[1]
    model = PINN_7Class(n_features=n_feat, hidden_dim=128, dropout=0.2)
    criterion = FocalPhysicsLoss(gamma=2.0, lam_phys=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    Xt = torch.FloatTensor(X[tr_idx])
    gt = torch.FloatTensor(gases[tr_idx])
    yt = torch.LongTensor(y[tr_idx])
    Xv = torch.FloatTensor(X[val_idx])
    gv = torch.FloatTensor(gases[val_idx])
    yv = torch.LongTensor(y[val_idx])

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    best_val = float('inf')
    best_state = None
    patience, pc = 30, 0

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"  Training PINN 7-Class  |  {n_params:,} params")
    print(f"{'='*60}")

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        ep_loss, nb = 0, 0

        for i in range(0, len(Xt), batch_size):
            idx = perm[i:i+batch_size]
            logits, phys = model(Xt[idx], gt[idx])
            loss, _ = criterion(logits, yt[idx], phys)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
            nb += 1

        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits_v, phys_v = model(Xv, gv)
            vl, _ = criterion(logits_v, yv, phys_v)
            pred_v = logits_v.argmax(dim=1).numpy()
            true_v = yv.numpy()
            val_acc = accuracy_score(true_v, pred_v)
            val_f1 = f1_score(true_v, pred_v, average='macro')

        history['train_loss'].append(ep_loss / nb)
        history['val_loss'].append(vl.item())
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        if vl.item() < best_val:
            best_val = vl.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pc = 0
        else:
            pc += 1

        if (epoch + 1) % 20 == 0:
            gate = torch.sigmoid(model.residual_gate).item()
            print(f"  Epoch {epoch+1:3d}  |  Train: {ep_loss/nb:.4f}  |  "
                  f"Val: {vl.item():.4f}  |  Acc: {val_acc:.4f}  |  "
                  f"F1: {val_f1:.4f}  |  Gate: {gate:.3f}")

        if pc >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model, history


# ============================================================
# 6. EVALUATION & VISUALIZATION
# ============================================================

def evaluate_and_plot(model, X, gases, y, te_idx, history, le):
    """Evaluate on test set with confusion matrix and training curves."""
    model.eval()
    Xte = torch.FloatTensor(X[te_idx])
    gte = torch.FloatTensor(gases[te_idx])
    yte = y[te_idx]

    with torch.no_grad():
        logits, phys = model(Xte, gte)
        pred = logits.argmax(dim=1).numpy()
        pred_phys = phys.argmax(dim=1).numpy()

    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average='macro')
    acc_phys = accuracy_score(yte, pred_phys)
    f1_phys = f1_score(yte, pred_phys, average='macro')
    gate = torch.sigmoid(model.residual_gate).item()

    print(f"\n{'='*60}")
    print(f"  TEST SET RESULTS (n={len(te_idx)})")
    print(f"{'='*60}")
    print(f"  PINN 7-Class:   Acc = {acc:.4f}  |  F1 = {f1:.4f}")
    print(f"  Physics-Only:   Acc = {acc_phys:.4f}  |  F1 = {f1_phys:.4f}")
    print(f"  Improvement:    {(f1 - f1_phys)*100:+.1f}% F1")
    print(f"  Gate: {gate:.3f}")
    print(f"{'='*60}")
    print(f"\n{classification_report(yte, pred, target_names=le.classes_)}")

    # Save metrics
    pd.DataFrame({
        'Model': ['PINN-7Class', 'Physics-Only', 'RF v2 (reference)'],
        'Accuracy': [acc, acc_phys, 0.986],
        'F1_Macro': [f1, f1_phys, 0.986],
    }).to_csv(os.path.join(RESULTS_DIR, 'pinn_7class_results.csv'), index=False)

    # --- PLOTS ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.patch.set_facecolor('#0D1117')
    c = {'p': '#58A6FF', 'o': '#F97316', 'g': '#56D364', 'r': '#F85149',
         't': '#C9D1D9', 'bg': '#161B22', 'ln': '#30363D'}

    # 1. Confusion Matrix
    ax = axes[0, 0]
    cm = confusion_matrix(yte, pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(le.classes_, color=c['t'], fontsize=10, rotation=45)
    ax.set_yticklabels(le.classes_, color=c['t'], fontsize=10)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            color = 'white' if cm_norm[i, j] > 0.5 else c['t']
            ax.text(j, i, f'{cm[i,j]}\n{cm_norm[i,j]:.0%}',
                    ha='center', va='center', color=color, fontsize=9)
    ax.set_title('Confusion Matrix (Test Set)', color=c['t'], fontweight='bold', fontsize=13)
    ax.set_xlabel('Predicted', color=c['t']); ax.set_ylabel('True', color=c['t'])

    # 2. Training curves
    ax = axes[0, 1]
    ax.set_facecolor(c['bg'])
    ax.plot(history['train_loss'], color=c['p'], label='Train', lw=1.5)
    ax.plot(history['val_loss'], color=c['o'], label='Val', lw=1.5)
    ax.set_xlabel('Epoch', color=c['t']); ax.set_ylabel('Loss', color=c['t'])
    ax.set_title('Training Curves', color=c['t'], fontweight='bold', fontsize=13)
    ax.legend(facecolor=c['bg'], edgecolor=c['ln'], labelcolor=c['t'])
    ax.tick_params(colors=c['t'])
    for sp in ax.spines.values(): sp.set_color(c['ln'])

    # 3. F1 per class (bar chart)
    ax = axes[1, 0]
    ax.set_facecolor(c['bg'])
    f1_per_class = f1_score(yte, pred, average=None)
    f1_phys_per = f1_score(yte, pred_phys, average=None)
    x_pos = np.arange(N_CLASSES)
    w = 0.35
    ax.bar(x_pos - w/2, f1_per_class, w, color=c['p'], label='PINN', alpha=0.9)
    ax.bar(x_pos + w/2, f1_phys_per, w, color=c['o'], label='Physics', alpha=0.9)
    ax.set_xticks(x_pos); ax.set_xticklabels(le.classes_, color=c['t'], rotation=45)
    ax.set_ylabel('F1 Score', color=c['t'])
    ax.set_title('Per-Class F1: PINN vs Physics', color=c['t'], fontweight='bold', fontsize=13)
    ax.legend(facecolor=c['bg'], edgecolor=c['ln'], labelcolor=c['t'])
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors=c['t'])
    for sp in ax.spines.values(): sp.set_color(c['ln'])

    # 4. Accuracy & F1 over training
    ax = axes[1, 1]
    ax.set_facecolor(c['bg'])
    ax.plot(history['val_acc'], color=c['g'], label='Accuracy', lw=1.5)
    ax.plot(history['val_f1'], color=c['p'], label='F1 Macro', lw=1.5)
    ax.axhline(acc_phys, color=c['o'], ls='--', lw=1, label=f'Physics Acc={acc_phys:.2f}')
    ax.set_xlabel('Epoch', color=c['t']); ax.set_ylabel('Score', color=c['t'])
    ax.set_title('Validation Metrics', color=c['t'], fontweight='bold', fontsize=13)
    ax.legend(facecolor=c['bg'], edgecolor=c['ln'], labelcolor=c['t'])
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors=c['t'])
    for sp in ax.spines.values(): sp.set_color(c['ln'])

    fig.suptitle('PINN — 7-Class DGA Fault Classifier (IEEE C57.104)',
                 color='white', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(RESULTS_DIR, 'pinn_7class_results.png'), dpi=200,
                bbox_inches='tight', facecolor='#0D1117')
    plt.close()
    print(f"  Saved: pinn_7class_results.png")

    return {'acc': acc, 'f1': f1, 'acc_phys': acc_phys, 'f1_phys': f1_phys, 'gate': gate}


# ============================================================
# 7. MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  PINN 7-Class DGA Fault Classifier")
    print("  IEEE C57.104 / IEC 60599 Compliant")
    print("=" * 60)

    X, gases, y, tr_idx, val_idx, te_idx, scaler, le = prepare_data()
    model, history = train_model(X, gases, y, tr_idx, val_idx, epochs=200, lr=1e-3)
    metrics = evaluate_and_plot(model, X, gases, y, te_idx, history, le)

    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_,
        'label_encoder_classes': le.classes_.tolist(),
        'feature_cols': FEATURE_COLS,
        'n_features': len(FEATURE_COLS),
        'metrics': metrics,
    }, os.path.join(RESULTS_DIR, 'pinn_7class_model.pt'))
    print(f"  Saved: pinn_7class_model.pt")

    return model, metrics


if __name__ == '__main__':
    model, metrics = main()
