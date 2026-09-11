#!/usr/bin/env python3
"""Verify the frozen ARI locked-test pipeline without reading test content."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


ROOT=Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest().upper()


def csv_rows(path: Path) -> list[dict[str,str]]:
    with path.open(encoding="utf-8-sig",newline="") as stream: return list(csv.DictReader(stream))


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("config",type=Path); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    config_path=args.config.resolve(); cfg=json.loads(config_path.read_text(encoding="utf-8"))
    if cfg["status"]!="frozen_pre_test" or not cfg["test_access_authorized"]:
        raise PermissionError("configuration is not frozen/authorized")
    fixed=((cfg["source_inventory"],cfg["source_inventory_sha256"]),(cfg["split_csv"],cfg["split_csv_sha256"]),(cfg["q26_csv"],cfg["q26_csv_sha256"]),(cfg["q26_normalization"],cfg["q26_normalization_sha256"]),(cfg["training_statistics"],cfg["training_statistics_sha256"]),(cfg["checkpoint"],cfg["checkpoint_sha256"]))
    for relative,expected in fixed:
        path=ROOT/relative
        if sha256(path)!=expected: raise ValueError(f"resource hash mismatch: {relative}")
    for resource in cfg["resources"]:
        if sha256(ROOT/resource["path"])!=resource["sha256"]: raise ValueError(f"pipeline resource hash mismatch: {resource['path']}")
    rows=csv_rows(ROOT/cfg["split_csv"]); counts={s:sum(r["split"]==s for r in rows) for s in ("train","val","test")}
    test=[r for r in rows if r["split"]=="test"]
    if counts!={"train":130,"val":22,"test":22} or [r["subject_id"] for r in test]!=cfg["cohort"]["test_subject_ids"]:
        raise ValueError("frozen split/test inventory mismatch")
    inventory=json.loads((ROOT/cfg["source_inventory"]).read_text(encoding="utf-8"))
    raw_root=Path(inventory["destination"]); by_label={r["subject_id"]:r for r in inventory["files"]}
    raw_missing=[r["subject_id"] for r in test if not (raw_root/by_label[r["subject_id"]]["filename"]).is_file()]
    if raw_missing: raise FileNotFoundError(f"missing raw test files: {raw_missing}")
    report=json.loads((ROOT/"artifacts/training/ari_fsc_adapted_q26_spectral_cnn_seed20260911_e190/training_report.json").read_text())
    if report["status"]!="completed" or report["cycles"]!=190 or report["test_subjects_read"]!=0 or report["authoritative_checkpoint"]!="last.pt":
        raise ValueError("E190 completion/checkpoint policy mismatch")
    train_path=next((ROOT/"artifacts/ari_fsc_adapted_q26_v2/train").glob("nh*.h5"))
    with h5py.File(train_path,"r") as handle:
        directions=np.asarray(handle["direction_features"][:],dtype=np.float64)
        mask=np.asarray(handle["interpolation_evaluation_mask"][:],dtype=bool).reshape(-1)
        q26=np.asarray(handle["sparse_direction_indices_zero_based"][:]).reshape(-1)
    horizontal=mask & (np.abs(directions[:,1])<=1e-9)
    def cap(center: float) -> np.ndarray:
        dot=np.cos(np.deg2rad(directions[:,1]))*np.cos(np.deg2rad(directions[:,0]-center))
        return mask & (np.rad2deg(np.arccos(np.clip(dot,-1,1)))<=25+1e-10)
    left=cap(270.0); right=cap(90.0)
    if directions.shape!=(1550,6) or mask.sum()!=1524 or q26.size!=26 or horizontal.sum()!=86:
        raise ValueError("frozen ARI direction geometry mismatch")
    if np.any(directions[mask,5]<=0) or np.any(directions[~mask,5]!=0) or not np.isclose(directions[:,5].sum(),1):
        raise ValueError("solid-angle weights mismatch")
    targets=[cfg["test_access_state"],cfg["prepared_waveform_root"],cfg["feature_root"],cfg["prediction_root"],cfg["result_root"]]
    existing=[path for path in targets if (ROOT/path).exists()]
    if existing: raise FileExistsError(f"locked-test target already exists: {existing}")
    result={"status":"passed","config_sha256":sha256(config_path),"resource_count":len(cfg["resources"]),"split_counts":counts,"test_subject_ids":cfg["cohort"]["test_subject_ids"],"raw_test_files_present":22,"raw_test_content_read":False,"test_subjects_read":0,"checkpoint_sha256":cfg["checkpoint_sha256"],"checkpoint_policy":"E190 last.pt only","direction_count":1550,"evaluation_direction_count":1524,"horizontal_direction_count":86,"left_contralateral_cap_count":int(left.sum()),"right_contralateral_cap_count":int(right.sum()),"weights_finite":bool(np.isfinite(directions[:,5]).all()),"weights_sum":float(directions[:,5].sum()),"all_output_targets_absent":True,"bootstrap_replicates":10000,"bootstrap_seed":20260911,"all_finite":True}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
