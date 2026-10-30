"""Train the Task-1 system and estimate its official score with cross-validation.

usage: python train.py --data DIR_WITH_features.csv_and_reasoning_audit.csv --eval-dir CHIMERA-agent/evaluation --out model_dir
"""
import argparse, json, os, sys, warnings
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
from featurize import featurize, FEATURES
import common as C
import official_eval as OE

def row_inputs(r):
    s = {k: (None if pd.isna(r[k]) else (r[k].item() if hasattr(r[k], 'item') else r[k]))
         for k in ['psa', 'age', 'months', 'pirads', 'psad', 'psav', 'psap', 'vol', 'dre', 'bx', 'cspca', 'enc_type']}
    s['pmhx'] = None if pd.isna(r['pmhx']) else r['pmhx']
    return s, {'radiology_report': r['radiology_report']}

def main(a):
    raw = pd.read_csv(os.path.join(a.data, 'features.csv'))
    F_all = pd.DataFrame([featurize(*row_inputs(r)) for _, r in raw.iterrows()])[FEATURES]
    lab = raw['label'].notna().values
    F, y, ids = F_all[lab].reset_index(drop=True), raw.loc[lab, 'label'].astype(int).values, raw.loc[lab, 'case_id'].values
    ev = OE.load_official(a.eval_dir)
    gts = OE.build_gt(ids, y, os.path.join(a.data, 'reasoning_audit.csv'))
    wrows = [gts[c]['variable_weights'] for c in ids]

    # 1) out-of-fold risk scores (each patient scored by a model that never saw them)
    p = np.zeros(len(y)); R = 5
    for seed in range(R):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(F, y):
            p[te] += C.make_ensemble().fit(F.iloc[tr], y[tr]).predict_proba(F.iloc[te])[:, 1] / R
    print('OOF AUC (pooled): %.3f' % roc_auc_score(y, p))

    def preds_for(t, weights_list, conf_list):
        return {c: {'biopsy_decision': 'yes' if pp >= t else 'no', 'confidence': cf, 'variable_weights': w,
                    'reveal_sequence': list(C.REVEAL), 'free_text': 'x' * 60}
                for c, pp, w, cf in zip(ids, p, weights_list, conf_list)}

    # 2) decision threshold: maximise the official Task-1 score, smoothed over neighbouring thresholds
    priors_all = C.fit_priors(wrows)
    ts = np.round(np.arange(0.40, 0.72, 0.01), 2); vals = []
    for t in ts:
        pr = preds_for(t, [priors_all] * len(ids), ['clear'] * len(ids)); vals.append(OE.task1_score(ev, gts, pr)['task1_score'])
    vals = np.array(vals); sm = np.convolve(vals, np.ones(7) / 7, mode='same')
    threshold = float(ts[3:-3][sm[3:-3].argmax()])
    print('threshold (smoothed optimum): %.2f' % threshold)

    # 3) confidence cut-offs from the distribution of our own out-of-fold margins
    margins = np.abs(p - threshold); cuts = (float(np.quantile(margins, 0.10)), float(np.quantile(margins, 0.30)))

    # 4) honest nested estimate: priors/heads/threshold-independent parts are re-fitted inside each fold
    def cv_estimate(conf_mode, use_heads, reps=3):
        out = []
        for seed in range(reps):
            preds = {}
            for tr, te in KFold(5, shuffle=True, random_state=seed).split(F):
                pri = C.fit_priors([wrows[i] for i in tr])
                Xh = C.head_matrix(F, p, threshold)
                heads = C.fit_heads(Xh.iloc[tr].reset_index(drop=True), [wrows[i] for i in tr]) if use_heads else {}
                W = C.predict_weights(pri, heads, Xh.iloc[te].reset_index(drop=True))
                cf = C.confidence_label(p[te], threshold, cuts, conf_mode)
                for j, i in enumerate(te):
                    preds[ids[i]] = {'biopsy_decision': 'yes' if p[i] >= threshold else 'no', 'confidence': cf[j], 'variable_weights': W[j],
                                     'reveal_sequence': list(C.REVEAL), 'free_text': 'x' * 60}
            out.append(OE.task1_score(ev, gts, preds))
        return {k: float(np.mean([o[k] for o in out])) for k in out[0]}
    report = {}
    for name, (mode, heads) in {'priors + always-clear': ('always_clear', False), 'priors + heads + always-clear': ('always_clear', True),
                                'priors + heads + confidence rule': ('rule', True)}.items():
        report[name] = cv_estimate(mode, heads); r = report[name]
        print(f"{name:36s} Task1={r['task1_score']:.3f} mean_case={r['mean_case_score']:.3f} F1={r['biopsy_F1']:.3f} acc={r['accuracy']:.3f} "
              f"conf={r['confidence_score']:.3f} varW={r['variable_weight_score']:.3f} factorF1={r['important_decisive_factor_score']:.3f} tool={r['tool_score']:.3f} ground={r['section_grounding_score']:.3f}")
    # prefer the confidence rule (it reports real model uncertainty) unless it costs more than 0.005
    conf_mode = 'rule' if report['priors + heads + confidence rule']['task1_score'] >= report['priors + heads + always-clear']['task1_score'] - 0.005 else 'always_clear'
    print('confidence mode chosen:', conf_mode)

    # 5) final fit on all labelled cases
    ensemble = C.make_ensemble().fit(F, y)
    heads = C.fit_heads(C.head_matrix(F, p, threshold), wrows)
    os.makedirs(a.out, exist_ok=True)
    joblib.dump({'ensemble': ensemble, 'priors': priors_all, 'heads': heads, 'threshold': threshold,
                 'conf_cuts': cuts, 'conf_mode': conf_mode, 'features': FEATURES}, os.path.join(a.out, 'model.joblib'))
    json.dump({'oof_auc': float(roc_auc_score(y, p)), 'threshold': threshold, 'conf_cuts': cuts, 'conf_mode': conf_mode,
               'cv_estimates_official_deterministic': report, 'n_train': int(len(y)),
               'note': 'LLM rationale judge not available locally: scores exclude the rationale component.'},
              open(os.path.join(a.out, 'training_report.json'), 'w'), indent=2)
    np.save(os.path.join(a.out, 'oof_risk.npy'), p)
    print('saved model to', a.out)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--data', default='.'); ap.add_argument('--eval-dir', required=True); ap.add_argument('--out', default='model')
    main(ap.parse_args())
