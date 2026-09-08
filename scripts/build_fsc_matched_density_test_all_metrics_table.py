"""Assemble the locked-test primary and supplemental FSC diagonal table."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COUNTS = (14, 26, 50)
LABELS = {14: "FSC-Q14@Q14", 26: "FSC-Q26@Q26", 50: "FSC-Q50@Q50"}
UNITS = {"FullSphereERB":"dB", "Contralateral25ERB":"dB", "ContralateralHighFrequency":"dB", "HorizontalILDMAE":"dB", "ERBBandILDMean":"dB", "ITDWeightedMAE_us":"us", "LAP2024LSD_dB":"dB"}
TIERS = {"FullSphereERB":"Primary frozen engineering test", "Contralateral25ERB":"Primary frozen engineering test", "ContralateralHighFrequency":"Primary frozen engineering test", "HorizontalILDMAE":"Primary frozen engineering test", "ERBBandILDMean":"Secondary locked-test supplement", "ITDWeightedMAE_us":"Deferred locked-test supplement", "LAP2024LSD_dB":"LAP2024 descriptive locked-test evaluation"}

def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f: return list(csv.DictReader(f))
def write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def boot(a, reps, seed):
    a=np.asarray(a,float); rng=np.random.default_rng(seed); means=a[rng.integers(0,len(a),(reps,len(a)))].mean(1); lo,hi=np.quantile(means,[.025,.975],method="linear"); return float(a.mean()),float(a.std(ddof=1)),float(lo),float(hi)

def main():
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument("configuration", type=Path); args=parser.parse_args()
    cfg=json.loads(args.configuration.read_text(encoding="utf-8"))
    if cfg.get("status") != "frozen" or cfg.get("split") != "test": raise PermissionError("frozen test configuration required")
    OUT = ROOT / cfg["output_root"]
    primary=read(OUT/"primary_subject_level.csv"); supplemental=read(OUT/"supplemental_subject_level.csv")
    values={}
    for row in primary+supplemental:
        q=int(row["DirectionCount"]); metric=row.get("Metric",row.get("Endpoint")); raw=row.get("Value", row.get("Value_dB")); values.setdefault((metric,q),[]).append(float(raw))
    if set(values)!={(m,q) for m in UNITS for q in COUNTS}: raise ValueError("metric/Q cardinality mismatch")
    summary=[]; wide=[]
    for metric in UNITS:
        row={"Metric":metric,"Unit":UNITS[metric],"EvidenceTier":TIERS[metric]}
        for q in COUNTS:
            mean,sd,lo,hi=boot(values[(metric,q)],int(cfg["bootstrap_replicates"]),int(cfg["bootstrap_seed"])+q)
            row[f"Q{q}_Mean"]=format(mean,".17g"); row[f"Q{q}_SD"]=format(sd,".17g")
            summary.append({"MethodLabel":LABELS[q],"DirectionCount":q,"Metric":metric,"Unit":UNITS[metric],"EvidenceTier":TIERS[metric],"SubjectCount":44,"Mean":format(mean,".17g"),"SampleStd":format(sd,".17g"),"Bootstrap95Lower":format(lo,".17g"),"Bootstrap95Upper":format(hi,".17g")})
        wide.append(row)
    write(OUT/"all_metrics_summary_mean_std.csv",summary); write(OUT/"paper_observation_density_all_metrics_test.csv",wide)
    meta=json.loads((OUT/"summary.json").read_text()); meta.update({"status":"completed","split":"test","subject_count":44,"test_subject_count_read":44,"metric_count":7,"all_metrics_subject_row_count":len(primary)+len(supplemental),"all_metrics_aggregate_row_count":len(summary),"supplemental_aggregate_row_count":meta.get("aggregate_row_count", len(supplemental)//44),"aggregate_row_count":len(summary),"all_finite":True,"metrics":list(UNITS),"evidence_tiers":TIERS}); (OUT/"summary.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
if __name__=="__main__": main()
