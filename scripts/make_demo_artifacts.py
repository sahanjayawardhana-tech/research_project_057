from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'demo'
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
users = [f'usr{i:03d}' for i in range(1, 31)]
dates = pd.date_range('2011-01-01', periods=45, freq='D')
rows = []
for d in dates:
    for u in users:
        ah = int(rng.poisson(0.2))
        dev = int(rng.poisson(0.3))
        filec = int(rng.poisson(0.25))
        ext = float(np.clip(rng.normal(0.08, 0.08), 0, 1))
        job = int(rng.binomial(2, 0.05))
        unique_pcs = int(rng.choice([1,1,1,2], p=[0.6,0.2,0.1,0.1]))
        beh = 0.25*(ah+dev) + 0.25*(dev+filec) + 0.25*(filec*(1+ext)) + 0.25*(job+dev)
        rel = ext + (1 if unique_pcs > 1 else 0)
        fused = 0.55*beh + 0.45*rel
        rows.append({
            'user': u,
            'date_only': str(d.date()),
            'after_hours_login_count': ah,
            'device_connect_count': dev,
            'file_copy_count': filec,
            'external_recipient_ratio': ext,
            'job_search_keyword_hits': job,
            'unique_pcs': unique_pcs,
            'logon_plus_device': ah + dev,
            'device_plus_file': dev + filec,
            'file_plus_email_external': filec * (1 + ext),
            'odd_hour_plus_rare_pc': ah + (1 if unique_pcs > 1 else 0),
            'job_search_plus_usb': job + dev,
            'behavior_shift_score': beh,
            'temporal_anomaly_score': beh,
            'relational_anomaly_score': rel,
            'fused_risk_score': fused,
            'department': rng.choice(['HR','Finance','Legal','Sales','R&D','IT']),
            'role': rng.choice(['Staff','Manager','Engineer','Analyst','ITAdmin'], p=[0.42,0.17,0.18,0.18,0.05]),
            'psychometric_O': float(np.clip(rng.normal(0.5,0.15),0,1)),
            'psychometric_C': float(np.clip(rng.normal(0.5,0.15),0,1)),
            'psychometric_E': float(np.clip(rng.normal(0.5,0.15),0,1)),
            'psychometric_A': float(np.clip(rng.normal(0.5,0.15),0,1)),
            'psychometric_N': float(np.clip(rng.normal(0.5,0.15),0,1)),
        })
user_day = pd.DataFrame(rows)
mx = max(user_day['fused_risk_score'].max(), 1)
user_day['malicious_probability'] = (user_day['fused_risk_score']/mx).clip(0,1)
user_day['severity'] = pd.cut(user_day['malicious_probability'], bins=[-0.01,0.35,0.60,0.80,1.0], labels=['low','medium','high','critical'])
user_day['top_signal'] = np.select(
    [user_day['job_search_plus_usb'] > 1, user_day['file_plus_email_external'] > 1.5, user_day['odd_hour_plus_rare_pc'] > 1],
    ['job_search_plus_usb', 'file_plus_email_external', 'odd_hour_plus_rare_pc'],
    default='behavior_shift_score'
)
for idx in rng.choice(user_day.index, size=18, replace=False):
    user_day.loc[idx, ['after_hours_login_count','device_connect_count','file_copy_count','job_search_keyword_hits','unique_pcs']] += [2,2,3,2,1]
    user_day.loc[idx, 'external_recipient_ratio'] = np.clip(user_day.loc[idx, 'external_recipient_ratio'] + 0.55, 0, 1)
    user_day.loc[idx, 'behavior_shift_score'] += 4.0
    user_day.loc[idx, 'temporal_anomaly_score'] += 4.0
    user_day.loc[idx, 'relational_anomaly_score'] += 1.2
    user_day.loc[idx, 'fused_risk_score'] += 3.0
mx = max(user_day['fused_risk_score'].max(), 1)
user_day['malicious_probability'] = (user_day['fused_risk_score']/mx).clip(0,1)
user_day['severity'] = pd.cut(user_day['malicious_probability'], bins=[-0.01,0.35,0.60,0.80,1.0], labels=['low','medium','high','critical'])
user_day.to_csv(OUT/'user_day_features.csv', index=False)

alerts = user_day[user_day['malicious_probability'] >= 0.60].copy().sort_values('malicious_probability', ascending=False).head(120)
alerts = alerts.reset_index(drop=True)
alerts['alert_id'] = ['ALT-%04d' % (i+1) for i in range(len(alerts))]
alerts['status'] = rng.choice(['open','triaged','closed'], size=len(alerts), p=[0.55,0.25,0.20])
alerts['analyst'] = rng.choice(['A.Senara','P.Dassanayake','S.Dilantha','SOC-1'], size=len(alerts))
alerts = alerts[['alert_id','user','date_only','severity','malicious_probability','top_signal','status','analyst','after_hours_login_count','device_connect_count','file_copy_count','external_recipient_ratio','job_search_keyword_hits','unique_pcs','behavior_shift_score']]
alerts.to_csv(OUT/'alerts.csv', index=False)

etype = ['logon','device','http','email','file']
events = []
for _, r in user_day.sample(900, random_state=42).iterrows():
    events.append({
        'event_id': f"EV-{rng.integers(100000,999999)}",
        'date_only': r['date_only'],
        'user': r['user'],
        'event_type': rng.choice(etype),
        'is_after_hours': bool(rng.random() < min(0.85, 0.1 + float(r['malicious_probability']) * 0.6)),
        'pc': rng.choice(['pc-01','pc-02','pc-12','pc-44','pc-77']),
        'summary': r['top_signal'],
        'risk_hint': round(float(r['malicious_probability']), 3),
    })
pd.DataFrame(events).to_csv(OUT/'events.csv', index=False)

edge_rows = []
for u in users:
    for t in rng.choice(['pc-01','pc-02','pc-03','shared-01','shared-02'], size=2, replace=False):
        edge_rows.append({'source':u,'target':t,'weight':int(rng.integers(3,20)),'edge_type':'user_pc'})
    for c in rng.choice(users, size=3, replace=False):
        if c != u:
            edge_rows.append({'source':u,'target':c,'weight':int(rng.integers(1,12)),'edge_type':'user_contact'})
    for url in rng.choice(['jobportal.example','drive.example','wiki.example','finance.example','random.example'], size=2, replace=False):
        edge_rows.append({'source':u,'target':url,'weight':int(rng.integers(1,8)),'edge_type':'user_url'})
pd.DataFrame(edge_rows).to_csv(OUT/'graph_edges.csv', index=False)

importance = pd.DataFrame({
    'feature': [
        'behavior_shift_score','external_recipient_ratio','file_copy_count','job_search_keyword_hits',
        'device_connect_count','after_hours_login_count','unique_pcs','odd_hour_plus_rare_pc',
        'file_plus_email_external','relational_anomaly_score','temporal_anomaly_score'
    ],
    'importance': [0.192,0.146,0.132,0.115,0.103,0.088,0.071,0.062,0.048,0.03,0.013]
}).sort_values('importance', ascending=True)
importance.to_csv(OUT/'feature_importance.csv', index=False)

pd.DataFrame([
    {'table_name':'logon','row_count':560505,'missing_pct':0.12,'notes':'A small number of daily logons intentionally missing'},
    {'table_name':'device','row_count':28474,'missing_pct':0.35,'notes':'Some connect events may miss disconnects'},
    {'table_name':'http','row_count':12201983,'missing_pct':0.02,'notes':'Large table; use parquet after preprocessing'},
    {'table_name':'email','row_count':1041624,'missing_pct':0.05,'notes':'Multi-recipient parsing required'},
    {'table_name':'file','row_count':213831,'missing_pct':0.03,'notes':'Hex header + keywords'},
    {'table_name':'psychometric','row_count':1000,'missing_pct':0.00,'notes':'Big Five scores'},
    {'table_name':'ldap','row_count':1000,'missing_pct':0.01,'notes':'Organizational context'},
]).to_csv(OUT/'data_quality.csv', index=False)

perf = []
for t in np.round(np.linspace(0.1,0.95,18), 2):
    precision = float(np.clip(0.48 + 0.45*t, 0, 1))
    recall = float(np.clip(0.96 - 0.62*t, 0, 1))
    f1 = 2*precision*recall/max(precision+recall, 1e-9)
    perf.append({'threshold':t,'precision':precision,'recall':recall,'f1':f1,'fpr':float(np.clip(0.50 - 0.45*t, 0, 1)),'tpr':recall})
perf_df = pd.DataFrame(perf)
perf_df.to_csv(OUT/'performance_curve.csv', index=False)
best_row = perf_df.sort_values(['f1', 'precision', 'threshold'], ascending=[False, False, False]).iloc[0]

metrics = {
    'dataset': 'CERT r4.2 / Release 3 style layout',
    'roc_auc': 0.935,
    'pr_auc': 0.812,
    'operating_threshold': float(best_row['threshold']),
    'operating_threshold_basis': 'max_f1_on_holdout_threshold_curve',
    'precision_at_operating_threshold': float(best_row['precision']),
    'recall_at_operating_threshold': float(best_row['recall']),
    'f1_at_operating_threshold': float(best_row['f1']),
    'accuracy_at_operating_threshold': 0.947,
    'balanced_accuracy_at_operating_threshold': 0.842,
    'false_positive_rate_at_operating_threshold': float(best_row['fpr']),
    'best_f1_threshold': float(best_row['threshold']),
    'best_f1': float(best_row['f1']),
    'precision_at_best_f1': float(best_row['precision']),
    'recall_at_best_f1': float(best_row['recall']),
    'accuracy': 0.842,
    'balanced_accuracy': 0.842,
    'raw_accuracy_at_operating_threshold': 0.947,
    'accuracy_at_0_80': 0.941,
    'balanced_accuracy_at_0_80': 0.835,
    'precision_at_0_80': 0.874,
    'recall_at_0_80': 0.733,
    'f1_at_0_80': 0.797,
    'false_positive_rate_at_0_80': 0.062,
    'thresholds': {'low':0.35,'medium':0.60,'high':0.80},
    'note': 'Demo metrics packaged for dashboard preview. Replace after full training.'
}
(OUT/'model_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
print(f'Wrote demo artifacts to {OUT}')
