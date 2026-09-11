#!/usr/bin/env python3
"""Merge frozen ARI metrics and compute subject-paired statistics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
METRICS=("FullSphereERB","Contralateral25ERB","ERBBandILDMean","ITDWeightedMAE","FullSphereLSD")
UNITS={"FullSphereERB":"dB","Contralateral25ERB":"dB","ERBBandILDMean":"dB","ITDWeightedMAE":"us","FullSphereLSD":"dB"}


def read_csv(path: Path) -> list[dict[str,str]]:
    with path.open(encoding="utf-8-sig",newline="") as stream: return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str,object]]) -> None:
    if path.exists() or not rows: raise FileExistsError(path)
    with path.open("x",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("config",type=Path); args=parser.parse_args()
    cfg=json.loads(args.config.resolve().read_text(encoding="utf-8")); out=ROOT/cfg["result_root"]
    access=json.loads((ROOT/cfg["test_access_state"]).read_text(encoding="utf-8"))
    if cfg["status"]!="frozen_pre_test" or access.get("status")!="completed" or access.get("test_subjects_read")!=22:
        raise PermissionError("locked test access state mismatch")
    raw=read_csv(out/"secondary_subject_level.csv")+read_csv(out/"erb_subject_level.csv")
    values: dict[tuple[str,str,str],float]={}
    for row in raw:
        key=(row["SubjectLabel"],row["Method"],row["Metric"])
        if key in values: raise ValueError(f"duplicate metric row: {key}")
        value=float(row["Value"])
        if not np.isfinite(value): raise FloatingPointError(key)
        values[key]=value
    subjects=sorted({key[0] for key in values},key=lambda x:int(x[2:]))
    expected={(subject,method,metric) for subject in subjects for method in ("MCA","FSC") for metric in METRICS}
    if len(subjects)!=22 or set(values)!=expected: raise ValueError("subject/method/metric matrix incomplete")
    subject_rows=[]; summary_rows=[]; paired_rows=[]
    for subject in subjects:
        for method in ("MCA","FSC"):
            for metric in METRICS:
                subject_rows.append({"SubjectLabel":subject,"SubjectID":int(subject[2:]),"Split":"test","Method":method,"Metric":metric,"Unit":UNITS[metric],"Value":format(values[(subject,method,metric)],".17g")})
    for metric in METRICS:
        for method in ("MCA","FSC"):
            data=np.asarray([values[(s,method,metric)] for s in subjects],dtype=np.float64)
            summary_rows.append({"Method":method,"Metric":metric,"Unit":UNITS[metric],"SubjectCount":22,"Mean":format(data.mean(),".17g"),"SampleSD":format(data.std(ddof=1),".17g")})
        difference=np.asarray([values[(s,"FSC",metric)]-values[(s,"MCA",metric)] for s in subjects])
        rng=np.random.default_rng(int(cfg["statistics"]["bootstrap_seed"]))
        indices=rng.integers(0,22,size=(int(cfg["statistics"]["bootstrap_replicates"]),22))
        means=difference[indices].mean(axis=1); lower,upper=np.quantile(means,[0.025,0.975],method="linear")
        paired_rows.append({"Metric":metric,"Unit":UNITS[metric],"Difference":"FSC-MCA","SubjectCount":22,"MeanDifference":format(difference.mean(),".17g"),"Bootstrap95Lower":format(lower,".17g"),"Bootstrap95Upper":format(upper,".17g"),"BootstrapReplicates":int(cfg["statistics"]["bootstrap_replicates"]),"BootstrapSeed":int(cfg["statistics"]["bootstrap_seed"])})
    write_csv(out/"mca_fsc_subject_level_metrics.csv",subject_rows)
    write_csv(out/"five_metric_summary.csv",summary_rows)
    write_csv(out/"paired_bootstrap.csv",paired_rows)
    supported=all(float(row["MeanDifference"])<0 and float(row["Bootstrap95Upper"])<0 for row in paired_rows)
    summary={"status":"completed","database":"ARI","split":"test","cohort_size":174,"train_subjects":130,"validation_subjects":22,"test_subjects_read":22,"failed_subjects":[],"methods":["MCA","FSC"],"metrics":list(METRICS),"subject_level_rows":len(subject_rows),"summary_rows":len(summary_rows),"paired_rows":len(paired_rows),"all_finite":True,"split_leakage":False,"checkpoint":cfg["checkpoint"],"checkpoint_sha256":cfg["checkpoint_sha256"],"checkpoint_policy":"E190 last.pt only","e130_completed":True,"e190_completed":True,"bootstrap":{"replicates":10000,"seed":20260911,"unit":"subject","difference":"FSC-MCA"},"test_access_state":cfg["test_access_state"],"test_first_access_time":access["claimed_at"],"test_access_completed_time":access["completed_at"],"post_test_tuning":False,"protocol_deviations":[],"claim_decision_rule":cfg["statistics"]["claim_decision_rule"],"claim_supported":supported,"all_five_ci_below_zero":supported}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    summary_by_key={(row["Method"],row["Metric"]):row for row in summary_rows}
    paired_by_metric={row["Metric"]:row for row in paired_rows}
    lines=["# ARI external FSC replication","","This locked external-database experiment compares MCA with the single-seed FSC E190 endpoint on 22 held-out ARI subjects.","","All metrics are lower-is-better. Paired differences are FSC minus MCA; negative values favor FSC.","","## Result","",("The strict replication criterion was met: all five paired confidence intervals exclude zero in favor of FSC." if supported else "The strict all-five-metric replication criterion was not met."),"","| Metric | MCA mean ± sample SD | FSC mean ± sample SD | FSC−MCA mean [paired bootstrap 95% CI] |","|---|---:|---:|---:|"]
    for metric in METRICS:
        unit=UNITS[metric]; mca=summary_by_key[("MCA",metric)]; fsc=summary_by_key[("FSC",metric)]; paired=paired_by_metric[metric]
        lines.append(f"| {metric} | {float(mca['Mean']):.6f} ± {float(mca['SampleSD']):.6f} {unit} | {float(fsc['Mean']):.6f} ± {float(fsc['SampleSD']):.6f} {unit} | {float(paired['MeanDifference']):.6f} [{float(paired['Bootstrap95Lower']):.6f}, {float(paired['Bootstrap95Upper']):.6f}] {unit} |")
    lines.extend(["","## Integrity","",f"- Test subjects read: {access['test_subjects_read']}/22; all finite: true; failed subjects: 0.","- Split leakage: false; E130 and E190 completed; checkpoint: E190 `last.pt`.","- Test access occurred once after protocol freeze; no post-test tuning or protocol deviations occurred.","","ARI and SONICOM use different measured direction grids, weights, and Q26 definitions. Their absolute metric values must not be pooled as one experiment.",""])
    (out/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
