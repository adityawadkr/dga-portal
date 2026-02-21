"""
DGA Analysis Portal — Backend v5
==================================
Robust upload (any format), 5 diagnostic methods,
leaderboard, confusion matrix images, model metrics.
"""

import os, sys, time, json, re
import numpy as np
import pandas as pd
import torch
import joblib
from flask import Flask, render_template, request, jsonify, send_from_directory

PORTAL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(PORTAL_DIR)
sys.path.insert(0, MODEL_DIR)

from flask_cors import CORS
app = Flask(__name__,
            static_folder=os.path.join(PORTAL_DIR, 'static'),
            template_folder=os.path.join(PORTAL_DIR, 'templates'))
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

GAS_COLUMNS = ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2', 'CO', 'CO2']
_models = {'rf_v2': None, 'pinn_7class': None, 'dp_pinn': None, 'loaded': False}


# ══════════════════════════════════════════════════
# CLASSIC DIAGNOSTIC METHODS
# IEEE C57.104: ratio methods only valid when TDCG exceeds Condition 1
# ══════════════════════════════════════════════════
TDCG_THRESHOLD = 720  # ppm — IEEE C57.104 Condition 1 TDCG limit
ML_TDCG_THRESHOLD = 50 # ppm — Below this, all AI models default to Normal (prevents PD bias on clean oil)

def _check_tdcg(h2=0, ch4=0, c2h6=0, c2h4=0, c2h2=0, co=0):
    """Total Dissolved Combustible Gas. If below threshold, fault diagnosis is unreliable."""
    return h2 + ch4 + c2h6 + c2h4 + c2h2 + co

def duval_triangle(ch4, c2h4, c2h2, h2=0, c2h6=0, co=0):
    tdcg = _check_tdcg(h2, ch4, c2h6, c2h4, c2h2, co)
    if tdcg < TDCG_THRESHOLD: return 'Normal'
    total = ch4 + c2h4 + c2h2
    if total == 0: return 'Normal'
    pch4 = ch4/total*100; pc2h4 = c2h4/total*100; pc2h2 = c2h2/total*100
    if pc2h2 > 29: return 'D2'
    if pc2h2 > 13: return 'D1' if pc2h4 < 23 else 'D2'
    if pc2h4 > 50: return 'T3'
    if pc2h4 > 20: return 'T2'
    if pch4 > 98 - pc2h4: return 'T1'
    if pc2h4 < 4 and pch4 < 10: return 'PD'
    return 'Normal'

def rogers_ratio(h2, ch4, c2h6, c2h4, c2h2, co=0):
    tdcg = _check_tdcg(h2, ch4, c2h6, c2h4, c2h2, co)
    if tdcg < TDCG_THRESHOLD: return 'Normal'
    eps = 0.001
    r1 = c2h2/(c2h4+eps); r2 = ch4/(h2+eps); r5 = c2h4/(c2h6+eps)
    if r1 < 0.1 and r2 > 1.0 and r5 < 1.0: return 'Normal'
    if r1 < 0.1 and r2 < 1.0: return 'PD'
    if 0.1 <= r1 < 3.0 and r2 >= 0.1 and r2 <= 1.0: return 'D1'
    if 0.1 <= r1 < 3.0 and r2 > 1.0: return 'D1'
    if r1 >= 3.0: return 'D2'
    if r1 < 0.1 and 1.0 <= r5 < 3.0: return 'T2'
    if r1 < 0.1 and r5 >= 3.0: return 'T3'
    if r1 < 0.1 and r2 > 1.0 and r5 >= 1.0: return 'T1'
    return 'T2'

def iec_ratio(h2, ch4, c2h6, c2h4, c2h2, co=0):
    tdcg = _check_tdcg(h2, ch4, c2h6, c2h4, c2h2, co)
    if tdcg < TDCG_THRESHOLD: return 'Normal'
    eps = 0.001
    r1 = c2h2/(c2h4+eps); r2 = ch4/(h2+eps); r5 = c2h4/(c2h6+eps)
    if r1 < 0.1 and r2 > 1 and r5 < 1: return 'Normal'
    if r1 < 0.1 and r2 < 1 and r5 < 1: return 'PD'
    if 1 <= r1 <= 3 and 0.1 <= r2 <= 1 and r5 > 1: return 'D1'
    if r1 > 3: return 'D2'
    if r1 < 0.1 and r2 > 1 and 1 <= r5 <= 3: return 'T2'
    if r1 < 0.1 and r2 > 1 and r5 > 3: return 'T3'
    return 'T1'


# ══════════════════════════════════════════════════
# IEEE C57.104 THRESHOLDS (for comparison chart)
# ══════════════════════════════════════════════════
IEEE_LIMITS = {
    'H2':   {'condition1': 100, 'condition2': 200, 'condition3': 500, 'condition4': 700},
    'CH4':  {'condition1': 75,  'condition2': 125, 'condition3': 400, 'condition4': 1000},
    'C2H6': {'condition1': 65,  'condition2': 80,  'condition3': 150, 'condition4': 250},
    'C2H4': {'condition1': 50,  'condition2': 100, 'condition3': 200, 'condition4': 500},
    'C2H2': {'condition1': 1,   'condition2': 2,   'condition3': 35,  'condition4': 80},
    'CO':   {'condition1': 350, 'condition2': 570, 'condition3': 1400,'condition4': 1800},
    'CO2':  {'condition1': 2500,'condition2': 4000,'condition3': 10000,'condition4': 15000},
}


# ══════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════
def get_models():
    if _models['loaded']: return _models
    import torch, joblib
    try:
        rf = joblib.load(os.path.join(MODEL_DIR, 'dga_model_v2.joblib'))
        _models['rf_v2'] = {'model': rf['model'], 'scaler': rf['scaler'],
            'label_encoder': rf['label_encoder'], 'feature_cols': rf['feature_cols']}
        print("✓ RF v2")
    except Exception as e: print(f"✗ RF v2: {e}")
    try:
        from pinn_7class import PINN_7Class
        ckpt = torch.load(os.path.join(MODEL_DIR, 'pinn_7class_model.pt'), map_location='cpu', weights_only=False)
        m = PINN_7Class(n_features=ckpt['n_features']); m.load_state_dict(ckpt['model_state_dict']); m.eval()
        _models['pinn_7class'] = {'model': m, 'classes': ckpt['label_encoder_classes'],
            'scaler_mean': ckpt['scaler_mean'], 'scaler_scale': ckpt['scaler_scale'], 'feature_cols': ckpt['feature_cols']}
        print("✓ PINN 7-Class")
    except Exception as e: print(f"✗ PINN 7-Class: {e}")
    try:
        from dp_pinn import PINN_DP_v2
        ckpt = torch.load(os.path.join(MODEL_DIR, 'dp_pinn_model.pt'), map_location='cpu', weights_only=False)
        m = PINN_DP_v2(n_features=ckpt['n_features']); m.load_state_dict(ckpt['model_state_dict']); m.eval()
        _models['dp_pinn'] = {'model': m, 'scaler_mean': ckpt['scaler_mean'],
            'scaler_scale': ckpt['scaler_scale'], 'feature_cols': ckpt['feature_cols']}
        print("✓ DP PINN")
    except Exception as e: print(f"✗ DP PINN: {e}")
    _models['loaded'] = True
    return _models


# ══════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════
def compute_features_pinn7(df):
    eps = 0.001; out = df.copy()
    out['R1_CH4_H2']=df['CH4']/(df['H2']+eps); out['R2_C2H2_C2H4']=df['C2H2']/(df['C2H4']+eps)
    out['R3_C2H4_C2H6']=df['C2H4']/(df['C2H6']+eps); out['R4_C2H2_CH4']=df['C2H2']/(df['CH4']+eps)
    out['R5_CO2_CO']=df['CO2']/(df['CO']+eps)
    out['TCG']=df[['H2','CH4','C2H6','C2H4','C2H2','CO']].sum(axis=1)
    t=df['CH4']+df['C2H4']+df['C2H2']+eps
    out['Duval_CH4']=df['CH4']/t*100; out['Duval_C2H4']=df['C2H4']/t*100; out['Duval_C2H2']=df['C2H2']/t*100
    for g in GAS_COLUMNS: out[f'log_{g}']=np.log1p(df[g])
    return out

def compute_features_dp(df):
    eps=0.001; out=df.copy()
    out['R1_CH4_H2']=df['CH4']/(df['H2']+eps); out['R2_C2H2_C2H4']=df['C2H2']/(df['C2H4']+eps)
    out['R3_C2H4_C2H6']=df['C2H4']/(df['C2H6']+eps); out['R4_CO2_CO']=df['CO2']/(df['CO']+eps)
    out['R5_C2H2_CH4']=df['C2H2']/(df['CH4']+eps); out['CO_total']=df['CO']+df['CO2']
    out['TCG']=df[['H2','CH4','C2H6','C2H4','C2H2','CO']].sum(axis=1)
    out['thermal_idx']=df['C2H4']*2+df['C2H6']*1.5+df['CH4']+df['CO']*2.5+df['H2']*0.3+df['C2H2']*0.3
    for g in GAS_COLUMNS: out[f'log_{g}']=np.log1p(df[g])
    return out

def compute_features_rf(df):
    eps=0.001; out=df.copy()
    out['R1_CH4_H2']=df['CH4']/(df['H2']+eps); out['R2_C2H2_C2H4']=df['C2H2']/(df['C2H4']+eps)
    out['R3_C2H4_C2H6']=df['C2H4']/(df['C2H6']+eps); out['R4_C2H2_CH4']=df['C2H2']/(df['CH4']+eps)
    out['R5_CO2_CO']=df['CO2']/(df['CO']+eps)
    out['TCG']=df[['H2','CH4','C2H6','C2H4','C2H2','CO']].sum(axis=1)
    t=df['CH4']+df['C2H4']+df['C2H2']+eps
    out['Duval_CH4']=df['CH4']/t*100; out['Duval_C2H4']=df['C2H4']/t*100; out['Duval_C2H2']=df['C2H2']/t*100
    return out

def dp_health(dp):
    if dp>=800: return {'status':'Excellent','level':'success','pct':95}
    if dp>=500: return {'status':'Good','level':'success','pct':75}
    if dp>=300: return {'status':'Moderate','level':'warning','pct':50}
    if dp>=200: return {'status':'Critical','level':'danger','pct':25}
    return {'status':'End of Life','level':'danger','pct':10}


# ══════════════════════════════════════════════════
#  ROBUST COLUMN DETECTION (handles ANY format)
# ══════════════════════════════════════════════════
GAS_ALIASES = {
    'H2':   [r'\bh2\b', r'\bhydrogen\b', r'\bh₂\b'],
    'CH4':  [r'\bch4\b', r'\bmethane\b', r'\bch₄\b'],
    'C2H6': [r'\bc2h6\b', r'\bethane\b', r'\bc₂h₆\b'],
    'C2H4': [r'\bc2h4\b', r'\bethylene\b', r'\bc₂h₄\b', r'\bethene\b'],
    'C2H2': [r'\bc2h2\b', r'\bacetylene\b', r'\bc₂h₂\b', r'\bacetyline\b'],
    'CO':   [r'(?<![a-z])co(?![a-z0-9₂2])', r'\bcarbon\s*mono', r'\bcarbomonoxide\b'],
    'CO2':  [r'\bco2\b', r'\bco₂\b', r'\bcarbon\s*di', r'\bcarbondioxide\b'],
}

def smart_detect_columns(df):
    """Detect gas columns from ANY naming convention."""
    col_map = {}
    df_cols = list(df.columns)

    # Pass 1: exact regex match per column (CO2 before CO to avoid conflicts)
    for target in ['CO2', 'CO', 'H2', 'CH4', 'C2H6', 'C2H4', 'C2H2']:
        if target in col_map: continue
        for cn in df_cols:
            cn_clean = str(cn).strip().lower()
            # Skip already matched
            if cn in col_map.values(): continue
            for pattern in GAS_ALIASES[target]:
                if re.search(pattern, cn_clean, re.IGNORECASE):
                    col_map[target] = cn
                    break
            if target in col_map: break

    # Pass 2: if column names are just gas formulas (H2, CH4, etc.) — case insensitive
    for target in GAS_COLUMNS:
        if target in col_map: continue
        for cn in df_cols:
            if str(cn).strip().upper() == target.upper():
                col_map[target] = cn; break

    # Pass 3: positional (header row might be Index, H2, CH4, C2H6, C2H4, C2H2, CO, CO2)
    if len(col_map) < 5 and len(df_cols) >= 7:
        # Try numeric columns only
        numeric_cols = [c for c in df_cols if pd.to_numeric(df[c], errors='coerce').notna().sum() > len(df)*0.5]
        if len(numeric_cols) >= 7:
            gas_order = ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2', 'CO', 'CO2']
            for i, g in enumerate(gas_order):
                if g not in col_map and i < len(numeric_cols):
                    col_map[g] = numeric_cols[i]

    return col_map


def smart_read_file(fpath):
    """Read any tabular file, auto-detecting header row."""
    fname = fpath.lower()

    if fname.endswith(('.xlsx', '.xls')):
        # Try first 15 rows to find header
        preview = pd.read_excel(fpath, engine='openpyxl', header=None, nrows=15)
        hdr = 0
        for i, row in preview.iterrows():
            row_strs = [str(v).lower().strip() for v in row.values if str(v).strip() not in ('nan','')]
            gas_hits = sum(1 for s in row_strs if any(re.search(p, s, re.I) for gas, pats in GAS_ALIASES.items() for p in pats))
            if gas_hits >= 3:
                hdr = i; break
        df = pd.read_excel(fpath, engine='openpyxl', header=hdr)
    elif fname.endswith('.csv'):
        # Try common separators
        for sep in [',', ';', '\t', '|']:
            try:
                df = pd.read_csv(fpath, sep=sep)
                if len(df.columns) >= 5: break
            except: pass
        else:
            df = pd.read_csv(fpath)
    elif fname.endswith('.tsv'):
        df = pd.read_csv(fpath, sep='\t')
    elif fname.endswith('.json'):
        df = pd.read_json(fpath)
    elif fname.endswith('.txt'):
        # Try whitespace, then comma, then tab
        for sep in [r'\s+', ',', '\t']:
            try:
                df = pd.read_csv(fpath, sep=sep, engine='python')
                if len(df.columns) >= 5: break
            except: pass
        else:
            df = pd.read_csv(fpath)
    else:
        # Try CSV as fallback
        df = pd.read_csv(fpath)

    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]
    return df


# ══════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        import torch
        data = request.json
        if 'gases' not in data: return jsonify({'error': 'No gas data'}), 400
        gases_df = pd.DataFrame([data['gases']])
        for c in GAS_COLUMNS:
            gases_df[c] = pd.to_numeric(gases_df.get(c, 0), errors='coerce').fillna(0)

        g = gases_df.iloc[0]
        models = get_models()
        result = {}

        # IEEE thresholds check
        ieee = {}
        for gas in GAS_COLUMNS:
            val = float(g[gas])
            limits = IEEE_LIMITS[gas]
            if val <= limits['condition1']: cond = 1
            elif val <= limits['condition2']: cond = 2
            elif val <= limits['condition3']: cond = 3
            else: cond = 4
            ieee[gas] = {'value': val, 'condition': cond, 'limits': limits}
        result['ieee_status'] = ieee

        # Classic methods (with TDCG threshold — returns Normal if total gas below 720 ppm)
        result['duval'] = {'prediction': duval_triangle(g['CH4'], g['C2H4'], g['C2H2'], h2=g['H2'], c2h6=g['C2H6'], co=g['CO']), 'method': 'Duval Triangle 1'}
        result['rogers'] = {'prediction': rogers_ratio(g['H2'], g['CH4'], g['C2H6'], g['C2H4'], g['C2H2'], co=g['CO']), 'method': 'Rogers Ratio'}
        result['iec'] = {'prediction': iec_ratio(g['H2'], g['CH4'], g['C2H6'], g['C2H4'], g['C2H2'], co=g['CO']), 'method': 'IEC 60599'}

        tdcg = _check_tdcg(g['H2'], g['CH4'], g['C2H6'], g['C2H4'], g['C2H2'], g['CO'])

        # RF v2
        if models['rf_v2']:
            t0 = time.time()
            if tdcg < ML_TDCG_THRESHOLD:
                result['rf_v2'] = {'prediction': 'Normal', 'confidence': 99.0, 'probabilities': {'Normal': 0.99}, 'latency_ms': round((time.time()-t0)*1000,1)}
            else:
                m = models['rf_v2']
                feats = compute_features_rf(gases_df)
                X = m['scaler'].transform(feats[m['feature_cols']].values)
                probs = m['model'].predict_proba(X)[0]
                pred = m['label_encoder'].inverse_transform(m['model'].predict(X))[0]
                result['rf_v2'] = {'prediction': pred, 'confidence': round(float(max(probs))*100,1),
                    'probabilities': {c:round(float(p),4) for c,p in zip(m['label_encoder'].classes_,probs)},
                    'latency_ms': round((time.time()-t0)*1000,1)}

        # PINN 7-Class
        if models['pinn_7class']:
            t0 = time.time()
            if tdcg < ML_TDCG_THRESHOLD:
                result['pinn_7class'] = {'prediction': 'Normal', 'confidence': 99.0, 'probabilities': {'Normal': 0.99}, 'latency_ms': round((time.time()-t0)*1000,1)}
            else:
                m = models['pinn_7class']
                feats = compute_features_pinn7(gases_df)
                X = np.nan_to_num(feats[m['feature_cols']].values.astype(np.float32))
                X = (X - m['scaler_mean'])/m['scaler_scale']
                gr = gases_df[GAS_COLUMNS].values.astype(np.float32)
                with torch.no_grad():
                    logits, _ = m['model'](torch.FloatTensor(X), torch.FloatTensor(gr))
                    probs = torch.softmax(logits, dim=1).numpy()[0]
                pi = int(np.argmax(probs))
                result['pinn_7class'] = {'prediction': m['classes'][pi], 'confidence': round(float(max(probs))*100,1),
                    'probabilities': {c:round(float(p),4) for c,p in zip(m['classes'],probs)},
                    'latency_ms': round((time.time()-t0)*1000,1)}

        # DP PINN
        if models['dp_pinn']:
            t0 = time.time()
            m = models['dp_pinn']
            feats = compute_features_dp(gases_df)
            X = np.nan_to_num(feats[m['feature_cols']].values.astype(np.float32))
            X = (X - m['scaler_mean'])/m['scaler_scale']
            gr = gases_df[GAS_COLUMNS].values.astype(np.float32)
            with torch.no_grad():
                dp_p, dp_ph, _ = m['model'](torch.FloatTensor(X), torch.FloatTensor(gr))
            dp_val = round(float(dp_p.item()),1)
            result['dp_pinn'] = {'dp_predicted': dp_val, 'dp_physics': round(float(dp_ph.item()),1),
                'health': dp_health(dp_val), 'latency_ms': round((time.time()-t0)*1000,1)}

        # Gas ratios
        eps = 0.001
        result['ratios'] = {
            'CH4_H2': round(g['CH4']/(g['H2']+eps), 3),
            'C2H2_C2H4': round(g['C2H2']/(g['C2H4']+eps), 3),
            'C2H4_C2H6': round(g['C2H4']/(g['C2H6']+eps), 3),
            'CO2_CO': round(g['CO2']/(g['CO']+eps), 3),
        }

        # Consensus
        from collections import Counter
        preds = [result.get(k,{}).get('prediction') for k in ['pinn_7class','rf_v2','duval','rogers','iec']]
        preds = [p for p in preds if p]
        if preds:
            c = Counter(preds).most_common(1)[0]
            result['consensus'] = {'prediction': c[0], 'agreement': f"{c[1]}/{len(preds)}", 'total_methods': len(preds)}

        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        import torch
        f = request.files.get('file')
        if not f: return jsonify({'error': 'No file provided'}), 400

        upload_dir = os.path.join(PORTAL_DIR, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        fpath = os.path.join(upload_dir, f.filename)
        f.save(fpath)

        # Smart read
        df = smart_read_file(fpath)
        col_map = smart_detect_columns(df)

        missing = [c for c in GAS_COLUMNS if c not in col_map]
        if missing:
            return jsonify({
                'error': f'Could not detect columns for: {", ".join(missing)}',
                'detected': {k: str(v) for k,v in col_map.items()},
                'available_columns': [str(c) for c in df.columns[:30]],
                'hint': 'Make sure your file has columns for H2, CH4, C2H6, C2H4, C2H2, CO, CO2'
            }), 400

        # Extract gas data
        gases_df = pd.DataFrame()
        for target, source in col_map.items():
            vals = df[source].astype(str).str.replace(r'[<>,%]', '', regex=True)
            gases_df[target] = pd.to_numeric(vals, errors='coerce').fillna(0).clip(lower=0)

        # Drop rows where all gases are 0
        mask = gases_df[GAS_COLUMNS].sum(axis=1) > 0
        gases_df = gases_df[mask].reset_index(drop=True)

        if len(gases_df) == 0:
            return jsonify({'error': 'No valid gas data rows found after parsing'}), 400

        models = get_models()
        results = []
        fault_counts = {}

        for idx, row in gases_df.iterrows():
            if idx >= 200: break  # Cap
            sample = {c: float(row[c]) for c in GAS_COLUMNS}
            r = {'gases': sample, 'index': int(idx)}

            r['duval'] = duval_triangle(row['CH4'], row['C2H4'], row['C2H2'], h2=row['H2'], c2h6=row['C2H6'], co=row['CO'])
            r['rogers'] = rogers_ratio(row['H2'], row['CH4'], row['C2H6'], row['C2H4'], row['C2H2'], co=row['CO'])
            tdcg = _check_tdcg(row['H2'], row['CH4'], row['C2H6'], row['C2H4'], row['C2H2'], row['CO'])

            rdf = pd.DataFrame([sample])
            if models['rf_v2']:
                if tdcg < ML_TDCG_THRESHOLD:
                    r['rf_v2'] = {'prediction': 'Normal', 'confidence': 99.0}
                else:
                    m = models['rf_v2']; feats = compute_features_rf(rdf)
                    X = m['scaler'].transform(feats[m['feature_cols']].values)
                    pred = m['label_encoder'].inverse_transform(m['model'].predict(X))[0]
                    conf = round(float(max(m['model'].predict_proba(X)[0]))*100,1)
                    r['rf_v2'] = {'prediction': pred, 'confidence': conf}

            if models['pinn_7class']:
                if tdcg < ML_TDCG_THRESHOLD:
                    r['pinn_7class'] = {'prediction': 'Normal', 'confidence': 99.0}
                    fault_counts['Normal'] = fault_counts.get('Normal', 0) + 1
                else:
                    m = models['pinn_7class']; feats = compute_features_pinn7(rdf)
                    X = np.nan_to_num(feats[m['feature_cols']].values.astype(np.float32))
                    X = (X-m['scaler_mean'])/m['scaler_scale']
                    gr = rdf[GAS_COLUMNS].values.astype(np.float32)
                    with torch.no_grad():
                        logits, _ = m['model'](torch.FloatTensor(X), torch.FloatTensor(gr))
                        probs = torch.softmax(logits,dim=1).numpy()[0]
                    pred = m['classes'][int(np.argmax(probs))]
                    r['pinn_7class'] = {'prediction': pred, 'confidence': round(float(max(probs))*100,1)}
                    fault_counts[pred] = fault_counts.get(pred, 0) + 1

            if models['dp_pinn']:
                m = models['dp_pinn']; feats = compute_features_dp(rdf)
                X = np.nan_to_num(feats[m['feature_cols']].values.astype(np.float32))
                X = (X-m['scaler_mean'])/m['scaler_scale']
                gr = rdf[GAS_COLUMNS].values.astype(np.float32)
                with torch.no_grad():
                    dp_p, _, _ = m['model'](torch.FloatTensor(X), torch.FloatTensor(gr))
                dp_val = round(float(dp_p.item()),1)
                r['dp_pinn'] = {'dp': dp_val, 'health': dp_health(dp_val)}

            results.append(r)

        return jsonify({
            'total': len(results),
            'columns_detected': {k: str(v) for k,v in col_map.items()},
            'fault_distribution': fault_counts,
            'samples': results,
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/models', methods=['GET'])
def model_status():
    m = get_models()
    return jsonify({'rf_v2': m['rf_v2'] is not None, 'pinn_7class': m['pinn_7class'] is not None, 'dp_pinn': m['dp_pinn'] is not None})


@app.route('/api/leaderboard', methods=['GET'])
def leaderboard():
    board = [
        {'rank':1,'model':'PINN 7-Class','type':'Neural (PINN)','accuracy':99.81,'f1_macro':99.81,'precision':99.81,'recall':99.81,
         'params':'29.7K','physics':'✓ Gas ratio constraints','training_data':'3,500 (synthetic)','notes':'Learnable physics encoder',
         'confusion_matrix':'/static/img/cm_pinn_7class.png'},
        {'rank':2,'model':'Random Forest v2','type':'Ensemble','accuracy':98.57,'f1_macro':98.57,'precision':98.57,'recall':98.57,
         'params':'~300 trees','physics':'✗','training_data':'2,100 (synthetic)','notes':'Calibrated, balanced',
         'confusion_matrix':'/static/img/cm_rf_v2.png'},
        {'rank':3,'model':'XGBoost (5-fold)','type':'Ensemble','accuracy':99.43,'f1_macro':99.41,'precision':99.40,'recall':99.43,
         'params':'~100 trees','physics':'✗','training_data':'510 (DGADATA.xlsx)','notes':'5-fold CV, real data',
         'confusion_matrix':'/static/img/cm_xgboost.png'},
        {'rank':4,'model':'Random Forest (5-fold)','type':'Ensemble','accuracy':99.43,'f1_macro':99.41,'precision':99.40,'recall':99.43,
         'params':'~100 trees','physics':'✗','training_data':'510 (DGADATA.xlsx)','notes':'5-fold CV, real data'},
        {'rank':5,'model':'SVM RBF (5-fold)','type':'Kernel','accuracy':85.0,'f1_macro':77.02,'precision':85.0,'recall':78.35,
         'params':'N/A','physics':'✗','training_data':'510 (DGADATA.xlsx)','notes':'Struggles with 7 classes',
         'confusion_matrix':'/static/img/cm_svm_rbf.png'},
        {'rank':6,'model':'Duval Triangle','type':'Rule-based','accuracy':50.5,'f1_macro':41.4,'precision':45.2,'recall':38.6,
         'params':'0','physics':'✓ IEEE standard','training_data':'N/A','notes':'Industry standard baseline'},
        {'rank':7,'model':'Rogers Ratio','type':'Rule-based','accuracy':38.2,'f1_macro':28.8,'precision':32.1,'recall':27.5,
         'params':'0','physics':'✓ Gas ratios','training_data':'N/A','notes':'Classic ratio method'},
    ]
    dp = [
        {'rank':1,'model':'PINN-DP v2','type':'Neural (PINN)','r2':86.26,'rmse':40.9,'mae':31.9,
         'physics':'✓ Chendong+Vaurchex+DePablo','notes':'Paper-enhanced, 33 real samples',
         'plot':'/static/img/dp_pinn_results.png'},
        {'rank':2,'model':'Physics-Only','type':'Equation','r2':68.0,'rmse':62.5,'mae':49.0,
         'physics':'✓ Multi-equation avg','notes':'No neural network'},
    ]
    return jsonify({
        'fault_classification': board,
        'dp_prediction': dp,
        'feature_importance': '/static/img/rf_feature_importances.png',
    })


# ══════════════════════════════════════════════════
# FEATURE 1: Auto-load real fleet data from DGADATA.xlsx
# ══════════════════════════════════════════════════
_fleet_cache = None

def _analyze_fleet():
    global _fleet_cache
    if _fleet_cache: return _fleet_cache
    xlsx = os.path.join(MODEL_DIR, 'DGADATA.xlsx')
    if not os.path.exists(xlsx): return None
    try:
        df = smart_read_file(xlsx)
        col_map = smart_detect_columns(df)
        missing = [c for c in GAS_COLUMNS if c not in col_map]
        if missing: return None
        gases_df = pd.DataFrame()
        for t, src in col_map.items():
            vals = df[src].astype(str).str.replace(r'[<>,%]', '', regex=True)
            gases_df[t] = pd.to_numeric(vals, errors='coerce').fillna(0).clip(lower=0)
        mask = gases_df[GAS_COLUMNS].sum(axis=1) > 0
        gases_df = gases_df[mask].reset_index(drop=True)
        mdls = get_models()
        results = []; fault_counts = {}
        for idx, row in gases_df.iterrows():
            if idx >= 200: break
            sample = {c: float(row[c]) for c in GAS_COLUMNS}
            r = {'gases': sample, 'index': int(idx)}
            tdcg = _check_tdcg(row['H2'], row['CH4'], row['C2H6'], row['C2H4'], row['C2H2'], row['CO'])
            r['duval'] = duval_triangle(row['CH4'], row['C2H4'], row['C2H2'], h2=row['H2'], c2h6=row['C2H6'], co=row['CO'])
            r['rogers'] = rogers_ratio(row['H2'], row['CH4'], row['C2H6'], row['C2H4'], row['C2H2'], co=row['CO'])
            rdf = pd.DataFrame([sample])
            if mdls['rf_v2']:
                if tdcg < ML_TDCG_THRESHOLD:
                    r['rf_v2'] = {'prediction': 'Normal', 'confidence': 99.0}
                else:
                    m = mdls['rf_v2']; feats = compute_features_rf(rdf)
                    X = m['scaler'].transform(feats[m['feature_cols']].values)
                    pred = m['label_encoder'].inverse_transform(m['model'].predict(X))[0]
                    conf = round(float(max(m['model'].predict_proba(X)[0]))*100,1)
                    r['rf_v2'] = {'prediction': pred, 'confidence': conf}
            if mdls['pinn_7class']:
                if tdcg < ML_TDCG_THRESHOLD:
                    r['pinn_7class'] = {'prediction': 'Normal', 'confidence': 99.0}
                    fault_counts['Normal'] = fault_counts.get('Normal', 0) + 1
                else:
                    m = mdls['pinn_7class']; feats = compute_features_pinn7(rdf)
                    X = np.nan_to_num(feats[m['feature_cols']].values.astype(np.float32))
                    X = (X-m['scaler_mean'])/m['scaler_scale']
                    gr = rdf[GAS_COLUMNS].values.astype(np.float32)
                    with torch.no_grad():
                        logits, _ = m['model'](torch.FloatTensor(X), torch.FloatTensor(gr))
                        probs = torch.softmax(logits,dim=1).numpy()[0]
                    pred = m['classes'][int(np.argmax(probs))]
                    r['pinn_7class'] = {'prediction': pred, 'confidence': round(float(max(probs))*100,1)}
                    fault_counts[pred] = fault_counts.get(pred, 0) + 1
            if mdls['dp_pinn']:
                m = mdls['dp_pinn']; feats = compute_features_dp(rdf)
                X = np.nan_to_num(feats[m['feature_cols']].values.astype(np.float32))
                X = (X-m['scaler_mean'])/m['scaler_scale']
                gr = rdf[GAS_COLUMNS].values.astype(np.float32)
                with torch.no_grad():
                    dp_p, _, _ = m['model'](torch.FloatTensor(X), torch.FloatTensor(gr))
                dp_val = round(float(dp_p.item()), 1)
                pct = min(max(int((dp_val/1100)*100), 0), 100)
                if dp_val > 700: lvl, st = 'success', 'Excellent'
                elif dp_val > 450: lvl, st = 'success', 'Good'
                elif dp_val > 250: lvl, st = 'warning', 'Moderate'
                elif dp_val > 150: lvl, st = 'danger', 'Critical'
                else: lvl, st = 'danger', 'End of Life'
                r['dp_pinn'] = {'dp': dp_val, 'health': {'level': lvl, 'pct': pct, 'status': st}}
            results.append(r)
        _fleet_cache = {
            'total': len(results), 'samples': results,
            'fault_distribution': fault_counts,
            'columns_detected': {k: str(v) for k, v in col_map.items()},
            'source': 'DGADATA.xlsx (auto-loaded)'
        }
        return _fleet_cache
    except Exception as e:
        import traceback
        print(f"Fleet auto-load error: {e}")
        traceback.print_exc()
        return None

@app.route('/api/fleet', methods=['GET'])
def fleet_data():
    data = _analyze_fleet()
    if data: return jsonify(data)
    return jsonify({'error': 'Could not load fleet data', 'total': 0, 'samples': []}), 404


# ══════════════════════════════════════════════════
# FEATURE 2: Maintenance Recommendations
# ══════════════════════════════════════════════════
RECOMMENDATIONS = {
    'Normal': {'severity': 'low', 'color': 'ok', 'actions': [
        'Continue routine monitoring schedule',
        'Next DGA test in 6-12 months',
        'No immediate action required'
    ]},
    'PD': {'severity': 'medium', 'color': 'wn', 'actions': [
        'Increase DGA testing frequency to quarterly',
        'Inspect bushings and tap changers for corona',
        'Check for loose connections or floating metal parts',
        'Monitor H₂ trend — key indicator for PD'
    ]},
    'D1': {'severity': 'high', 'color': 'dn', 'actions': [
        'Increase DGA testing to monthly',
        'Perform electrical testing (power factor, turns ratio)',
        'Inspect for tracking on insulation surfaces',
        'Check for low-energy sparking at poor contacts',
        'Consider oil filtration / degassing'
    ]},
    'D2': {'severity': 'critical', 'color': 'dn', 'actions': [
        '⚠️ IMMEDIATE: Schedule emergency inspection',
        'Reduce transformer loading if possible',
        'Perform on-line DGA monitoring (weekly minimum)',
        'Plan for potential transformer replacement',
        'Commission internal inspection at next outage',
        'Check for arcing damage to windings and core'
    ]},
    'T1': {'severity': 'medium', 'color': 'wn', 'actions': [
        'Check for hot spots in core/clamps (<300°C)',
        'Verify cooling system operation (fans, pumps)',
        'Increase DGA testing to quarterly',
        'Inspect for blocked oil ducts'
    ]},
    'T2': {'severity': 'high', 'color': 'dn', 'actions': [
        'Thermal fault 300-700°C detected',
        'Perform thermographic survey immediately',
        'Check for circulating currents in core',
        'Inspect tank and radiator connections',
        'Increase DGA testing to monthly'
    ]},
    'T3': {'severity': 'critical', 'color': 'dn', 'actions': [
        '⚠️ HIGH PRIORITY: Severe overheating (>700°C)',
        'Reduce loading immediately',
        'Schedule emergency internal inspection',
        'Check for shorted turns and core faults',
        'Prepare contingency plan for replacement'
    ]}
}

DP_RECOMMENDATIONS = {
    'Excellent': ['Paper insulation in excellent condition', 'Continue normal maintenance schedule'],
    'Good': ['Paper insulation healthy', 'Monitor DP trend annually'],
    'Moderate': ['Paper aging detected — increase furan testing frequency', 'Plan for insulation assessment within 2 years'],
    'Critical': ['⚠️ Significant paper degradation', 'Schedule furan analysis immediately', 'Begin planning for rewind or replacement'],
    'End of Life': ['🚨 CRITICAL: Paper insulation near end of life', 'Immediate load reduction required', 'Emergency replacement planning needed']
}

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.json
    fault = data.get('fault', 'Normal')
    dp_status = data.get('dp_status', 'Good')
    gases = data.get('gases', {})
    
    rec = RECOMMENDATIONS.get(fault, RECOMMENDATIONS['Normal']).copy()
    dp_rec = DP_RECOMMENDATIONS.get(dp_status, DP_RECOMMENDATIONS['Good'])
    
    # IEEE C57.104 condition
    tdcg = sum(gases.get(g, 0) for g in ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2', 'CO'])
    if tdcg < 720: ieee_cond = 1
    elif tdcg < 1920: ieee_cond = 2
    elif tdcg < 4630: ieee_cond = 3
    else: ieee_cond = 4
    
    return jsonify({
        'fault_type': fault,
        'severity': rec['severity'],
        'fault_actions': rec['actions'],
        'dp_status': dp_status,
        'dp_actions': dp_rec,
        'ieee_condition': ieee_cond,
        'ieee_desc': ['Normal','Caution','Warning','Critical'][ieee_cond-1],
        'tdcg': round(tdcg, 1),
        'overall_risk': 'critical' if rec['severity']=='critical' or dp_status in ['Critical','End of Life'] else
                        'high' if rec['severity']=='high' or dp_status=='Moderate' else
                        'medium' if rec['severity']=='medium' else 'low'
    })


# ══════════════════════════════════════════════════
# FEATURE 3: IEEE Overall Condition Score
# ══════════════════════════════════════════════════
@app.route('/api/condition', methods=['POST'])
def condition_score():
    gases = request.json.get('gases', {})
    tdcg = sum(gases.get(g, 0) for g in ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2', 'CO'])
    
    IEEE_LIMITS = {
        'H2':  [100, 200, 500, 700],
        'CH4': [75, 125, 400, 1000],
        'C2H6':[65, 80, 150, 200],
        'C2H4':[50, 100, 200, 500],
        'C2H2':[1, 2, 35, 80],
        'CO':  [350, 700, 1800, 2500],
        'CO2': [2500, 4000, 10000, 15000],
    }
    
    max_cond = 1
    gas_conditions = {}
    for gas, lims in IEEE_LIMITS.items():
        val = gases.get(gas, 0)
        if val <= lims[0]: c = 1
        elif val <= lims[1]: c = 2
        elif val <= lims[2]: c = 3
        else: c = 4
        gas_conditions[gas] = {'value': val, 'condition': c, 'limits': lims}
        max_cond = max(max_cond, c)
    
    if tdcg < 720: tdcg_cond = 1
    elif tdcg < 1920: tdcg_cond = 2
    elif tdcg < 4630: tdcg_cond = 3
    else: tdcg_cond = 4
    
    overall = max(max_cond, tdcg_cond)
    descs = {1: 'Normal — no action needed', 2: 'Caution — increase monitoring',
             3: 'Warning — plan corrective action', 4: 'Critical — immediate attention'}
    
    return jsonify({
        'overall_condition': overall,
        'description': descs[overall],
        'tdcg': round(tdcg, 1), 'tdcg_condition': tdcg_cond,
        'gas_conditions': gas_conditions
    })


# ══════════════════════════════════════════════════
# FEATURE 4: Transformer Comparison
# ══════════════════════════════════════════════════
@app.route('/api/compare', methods=['POST'])
def compare():
    data = request.json
    indices = data.get('indices', [])
    fleet = _analyze_fleet()
    if not fleet: return jsonify({'error': 'No fleet data'}), 400
    
    comparisons = []
    for idx in indices[:3]:  # max 3
        if 0 <= idx < len(fleet['samples']):
            s = fleet['samples'][idx]
            g = s['gases']
            tdcg = sum(g.get(k, 0) for k in ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2', 'CO'])
            comparisons.append({
                'index': idx, 'tag': f'TRF-{idx+1:04d}',
                'gases': g,
                'pinn': s.get('pinn_7class', {}).get('prediction', '-'),
                'rf': s.get('rf_v2', {}).get('prediction', '-'),
                'duval': s.get('duval', '-'),
                'dp': s.get('dp_pinn', {}).get('dp', '-'),
                'dp_health': s.get('dp_pinn', {}).get('health', {}).get('status', '-'),
                'tdcg': round(tdcg, 1)
            })
    return jsonify({'comparisons': comparisons})


# ══════════════════════════════════════════════════
# FEATURE 5: Session Audit Log
# ══════════════════════════════════════════════════
_audit_log = []

@app.route('/api/audit', methods=['GET'])
def get_audit():
    return jsonify({'log': _audit_log[-50:]})  # last 50 entries

@app.route('/api/audit', methods=['POST'])
def add_audit():
    entry = request.json
    entry['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    entry['id'] = len(_audit_log) + 1
    _audit_log.append(entry)
    return jsonify({'ok': True, 'entry': entry})


if __name__ == '__main__':
    import threading
    print("\n" + "="*50)
    print("🔬 DGA Analysis Portal v6")
    print("="*50)
    print("http://127.0.0.1:5050")
    print("="*50)
    # Pre-warm fleet cache in background (takes ~60s for 200 samples)
    def _warm():
        print("⏳ Loading fleet data from DGADATA.xlsx (background)...")
        _analyze_fleet()
        if _fleet_cache:
            print(f"✓ Fleet loaded: {_fleet_cache['total']} transformers")
        else:
            print("✗ Fleet data not available")
    threading.Thread(target=_warm, daemon=True).start()
    print("="*50 + "\n")
    app.run(host='127.0.0.1', port=5050, debug=False)
