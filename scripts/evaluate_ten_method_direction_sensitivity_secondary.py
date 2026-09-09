"""Validation-only secondary/deferred metrics for frozen Q14/Q26/Q50 outputs."""
from __future__ import annotations
import argparse, csv, hashlib, json
from collections import defaultdict
from pathlib import Path
import h5py, numpy as np, torch
from mcar.evaluation.secondary_metrics import (bootstrap_mean_interval, distance_binned_lsd, full_sphere_lsd, normalized_direction_weights, spectral_band_ild_profile, spectral_shape_metrics)
from mcar.evaluation.deferred_secondary_metrics import dominant_notch_location_metrics, enable_external_torchaudio
from mcar.fsp_ae_signal import estimate_itd_seconds
from mcar.paths import project_root

METHODS=("SHOnly","SUpDEqSH","SUpDEqNN","SUpDEqBary","MCA","MCARv351","FSPAE","RANF","HYBRID","BOUNDED")
UNITS={"FullSphereLSD":"dB","HFFirstDifferenceMAE":"dB/bin","HFSecondDifferenceMAE":"dB/bin^2","MultiScaleNotchDepthMAE":"dB","ERBBandILDMean":"dB","SpatialLSD_0_10deg":"dB","SpatialLSD_10_20deg":"dB","SpatialLSD_20_30deg":"dB","SpatialLSD_30_180deg":"dB","DominantNotchPenalizedMAE_Hz":"Hz","DominantNotchMatchedMAE_Hz":"Hz","DominantNotchMissRate":"fraction","DominantNotchSpuriousRate":"fraction","ReferenceNotchFraction":"fraction","ITDWeightedMAE_us":"us","ITDMaximumAbsoluteError_us":"us"}
CLASSICAL=set(METHODS[:5])

def write_csv(path, rows):
    if not rows: raise ValueError("empty output")
    with path.open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def seed(base,*parts): return (base+int.from_bytes(hashlib.sha256("\0".join(map(str,parts)).encode()).digest()[:4],"big"))%(2**32)
def boot(x,n,s):
    x=np.asarray(x,float); rng=np.random.default_rng(s); means=x[rng.integers(0,len(x),(n,len(x)))].mean(1)
    return float(x.mean()),*map(float,np.quantile(means,[.025,.975],method="linear"))
def read_cache(path):
    with h5py.File(path) as h:
        c=h["cache"]; grid=np.asarray(c["referenceGrid"]).T; mask=np.asarray(c["frequencyMask"]).reshape(-1).astype(bool)
        reference=np.stack([complex_h5(np.asarray(c["referenceLeft"])),complex_h5(np.asarray(c["referenceRight"]))])
        mca_complex=np.stack([complex_h5(np.asarray(c["mcaLeft"])),complex_h5(np.asarray(c["mcaRight"]))])
        ref=20*np.log10(np.maximum(np.abs(reference[:,mask,:].transpose(0,2,1)),1e-10))
        mca=20*np.log10(np.maximum(np.abs(mca_complex[:,mask,:].transpose(0,2,1)),1e-10))
        refhr=np.asarray(c["referenceHrir"]).transpose(1,0,2); freq=np.asarray(c["frequencyHz"]).reshape(-1)[mask]
    return grid,mask,ref,mca,refhr,freq
def complex_h5(values):
    return np.asarray(values["real"],np.float32)+1j*np.asarray(values["imag"],np.float32)
def hrir_from_db(db,mca,mask):
    base=np.asarray(mca); allv=np.empty((2,793,513),complex); allv[:,:,mask]=10**(db/20)*np.exp(1j*np.angle(base))
    # mca supplied selected only; reconstruct outside values from cache separately is required
    raise RuntimeError("internal")
def cache_full_mca(path):
    with h5py.File(path) as h:
        c=h["cache"]; mask=np.asarray(c["frequencyMask"]).reshape(-1).astype(bool)
        allv=np.stack([complex_h5(np.asarray(c["mcaLeft"])).T,complex_h5(np.asarray(c["mcaRight"])).T])
    return allv,mask
def strict_hrir(db,full,mask):
    v=full.copy(); v[:,:,mask]=10**(db/20)*np.exp(1j*np.angle(v[:,:,mask])); return np.fft.irfft(v,n=1024,axis=-1)[:,:,:256].transpose(1,0,2).astype("float32")
def h5_prediction(path):
    with h5py.File(path) as h: return np.asarray(h["predicted_magnitude_db"],"float32"),np.asarray(h["predicted_hrir"],"float32")
def residual(path,mca,full,mask):
    with h5py.File(path) as h: r=np.asarray(h["predicted_residual_db"],"float32")
    p=mca+r; return p,strict_hrir(p,full,mask)
def fsp(path,mask,freq):
    with h5py.File(path) as h: mag=np.asarray(h["predicted_magnitude_db"]); hr=np.asarray(h["predicted_hrir"]); ff=np.asarray(h["frequency_hz"])
    if not np.allclose(ff,freq.__class__ and np.r_[0,freq],atol=1e-3): raise ValueError("FSP frequency mismatch")
    return mag[:,:,np.flatnonzero(mask)[1:]].transpose(1,0,2).astype("float32"),hr.astype("float32")
def ranf(path,mask):
    with h5py.File(path) as h: hr=np.asarray(h["Data.IR"],"float32")
    spec=np.fft.rfft(hr,n=1024,axis=-1).transpose(1,0,2); return (20*np.log10(np.maximum(np.abs(spec[:,:,mask]),1e-10))).astype("float32"),hr
def dist(xyz,chosen):
    u=xyz/np.linalg.norm(xyz,axis=1)[:,None]; return np.rad2deg(np.arccos(np.clip((u@u[chosen].T).max(1),-1,1)))
def itd(hr):
    x=estimate_itd_seconds(torch.from_numpy(hr).to("cuda" if torch.cuda.is_available() else "cpu"),44100.,upsampled_rate_hz=384000.,lowpass_hz=1600.,maximum_itd_seconds=.001,direction_batch_size=64)
    return x.cpu().numpy()
def main():
 p=argparse.ArgumentParser();p.add_argument("config",type=Path);a=p.parse_args(); root=project_root(); cfg=json.loads(a.config.read_text())
 if cfg["status"]!="frozen" or cfg["split"]!="val" or cfg["test_access_allowed"]: raise PermissionError("validation frozen config required")
 out=root/cfg["output_root"]; partial=out.with_name(out.name+".partial")
 if out.exists() or partial.exists(): raise FileExistsError(out)
 enable_external_torchaudio(Path("D:/miniconda3/envs/asd/Lib/site-packages")); partial.mkdir(parents=True)
 with (root/"configs/data/sonicom_subject_split_v1.csv").open(encoding="utf-8-sig") as f: subjects=[r for r in csv.DictReader(f) if r["split"]=="val"]
 gridrows=list(csv.DictReader((root/"configs/data/sonicom_nested_sparse_grid_q14_q26_q50_v1.csv").open())); q={n:np.array([int(r["source_index_zero_based"]) for r in gridrows if int(r["direction_count"])==n]) for n in (14,26,50)}; fixed=np.ones(793,bool);fixed[q[50]]=False
 rows=[]; bands=[]; vals=defaultdict(lambda:defaultdict(lambda:defaultdict(dict))); base=cfg["bootstrap_base_seed"]
 for si,s in enumerate(subjects,1):
  label=s["subject_id"]; sid=int(label[1:])
  for count in (14,26,50):
   cache=root/"artifacts/sparsity/sonicom_bounded_e25_input_direction_sensitivity_v1/subjects"/label/f"q{count}"/"cache.mat"; grid,mask,ref,mca,refhr,freq=read_cache(cache); full,_=cache_full_mca(cache); xyz=np.column_stack((np.cos(np.deg2rad(90-grid[:,1]))*np.cos(np.deg2rad(grid[:,0])),np.cos(np.deg2rad(90-grid[:,1]))*np.sin(np.deg2rad(grid[:,0])),np.sin(np.deg2rad(90-grid[:,1])))); d=dist(xyz,q[count]); horizontal=fixed & (np.abs(90-grid[:,1])<=1e-9); ritd=itd(refhr); w=grid[:,2]
   preds={}
   for method in METHODS:
    if method in CLASSICAL: preds[method]=h5_prediction(root/cfg["classical_prediction_root"]/"subjects"/label/f"q{count}"/method/"prediction.h5")
    elif method=="MCARv351": preds[method]=residual(root/"artifacts/sparsity/sonicom_ten_method_direction_sensitivity_v1/subjects"/label/f"q{count}"/"mcar_v351_prediction.h5",mca,full,mask)
    elif method=="HYBRID": preds[method]=residual(root/"artifacts/sparsity/sonicom_ten_method_direction_sensitivity_v1/subjects"/label/f"q{count}"/"hybrid_e190_prediction.h5",mca,full,mask)
    elif method=="BOUNDED": preds[method]=residual(root/"artifacts/sparsity/sonicom_bounded_e25_input_direction_sensitivity_v1/subjects"/label/f"q{count}"/"bounded_prediction.h5",mca,full,mask)
    elif method=="FSPAE": preds[method]=fsp(root/"artifacts/sparsity/sonicom_ten_method_direction_sensitivity_v1/subjects"/label/f"q{count}"/"fspae_prediction.h5",mask,freq)
    else: preds[method]=ranf(root/("artifacts/reconstruction/sonicom_ranf_ten_method_direction_sensitivity_q%d_validation"%count if count!=26 else "artifacts/reconstruction/sonicom_ranf_q26_validation_frozen")/"subjects"/label/"prediction.sofa",mask)
   for method,(pred,hr) in preds.items():
    lsd,dl,_=full_sphere_lsd(pred,ref,fixed,w); x={"FullSphereLSD":lsd};x.update(spectral_shape_metrics(pred[:,fixed],ref[:,fixed],w[fixed],freq)); cen,prof,mean=spectral_band_ild_profile(pred[:,horizontal],ref[:,horizontal],w[horizontal],freq);x["ERBBandILDMean"]=mean
    x.update({"SpatialLSD_"+k+"deg":v for k,v in distance_binned_lsd(dl,d,fixed,w).items()});x.update(dominant_notch_location_metrics(pred[:,fixed],ref[:,fixed],w[fixed],freq)); err=np.abs(itd(hr)-ritd)*1e6;x["ITDWeightedMAE_us"]=float((err[fixed]*normalized_direction_weights(w[fixed])).sum());x["ITDMaximumAbsoluteError_us"]=float(err[fixed].max())
    for ep,v in x.items():
     if ep not in UNITS or not np.isfinite(v): raise FloatingPointError(f"{label} Q{count} {method} {ep}")
     vals[ep][method][count][label]=float(v);rows.append(dict(SubjectLabel=label,SubjectID=sid,Split="val",DirectionCount=count,Method=method,Endpoint=ep,Unit=UNITS[ep],Value=format(float(v),".17g")))
    for bi,(c,v) in enumerate(zip(cen,prof)): bands.append(dict(SubjectLabel=label,DirectionCount=count,Method=method,BandIndex=bi,BandCenter_Hz=format(float(c),".17g"),ILDAbsoluteError_dB=format(float(v),".17g")))
  print(f"secondary [{si}/44] {label}",flush=True)
 agg=[]; effects=[]; interactions=[]
 for ep in UNITS:
  for method in METHODS:
   for count in (14,26,50):
    x=[vals[ep][method][count][s["subject_id"]] for s in subjects];mean,lo,hi=boot(x,cfg["bootstrap_replicates"],seed(base,ep,method,count));agg.append(dict(Method=method,DirectionCount=count,Endpoint=ep,Unit=UNITS[ep],SubjectCount=44,Mean=format(mean,".17g"),SampleStd=format(float(np.std(x,ddof=1)),".17g"),Bootstrap95Lower=format(lo,".17g"),Bootstrap95Upper=format(hi,".17g")))
   for count in (14,50):
    x=np.array([vals[ep][method][count][s["subject_id"]]-vals[ep][method][26][s["subject_id"]] for s in subjects]);mean,lo,hi=boot(x,cfg["bootstrap_replicates"],seed(base,ep,method,count,"effect"));effects.append(dict(Method=method,DirectionCount=count,ReferenceDirectionCount=26,Endpoint=ep,Unit=UNITS[ep],QMinusQ26Mean=format(mean,".17g"),Bootstrap95Lower=format(lo,".17g"),Bootstrap95Upper=format(hi,".17g")))
  for method in METHODS[:-1]:
   for count in (14,50):
    x=np.array([(vals[ep][method][count][s["subject_id"]]-vals[ep][method][26][s["subject_id"]])-(vals[ep]["BOUNDED"][count][s["subject_id"]]-vals[ep]["BOUNDED"][26][s["subject_id"]]) for s in subjects]);mean,lo,hi=boot(x,cfg["bootstrap_replicates"],seed(base,ep,method,count,"interaction"));interactions.append(dict(Comparator=method,DirectionCount=count,Endpoint=ep,Unit=UNITS[ep],ComparatorMinusBoundedSensitivity=format(mean,".17g"),Bootstrap95Lower=format(lo,".17g"),Bootstrap95Upper=format(hi,".17g")))
 if (len(rows),len(agg),len(effects),len(interactions))!=(21120,480,320,288): raise ValueError("cardinality failure")
 write_csv(partial/"per_subject_metrics.csv",rows);write_csv(partial/"aggregate_metrics.csv",agg);write_csv(partial/"paired_direction_effects.csv",effects);write_csv(partial/"bounded_interactions.csv",interactions);write_csv(partial/"band_ild_profile.csv",bands)
 quality={"status":"passed","split":"val","subject_count":44,"test_subject_count_read":0,"method_count":10,"endpoint_count":16,"fixed_evaluation_direction_count":743,"metric_rows":len(rows),"aggregate_rows":len(agg),"effect_rows":len(effects),"interaction_rows":len(interactions),"all_finite":True};(partial/"quality_checks.json").write_text(json.dumps(quality,indent=2)+"\n");(partial/"summary.json").write_text(json.dumps(dict(quality,bootstrap_replicates=cfg["bootstrap_replicates"],bootstrap_base_seed=base),indent=2)+"\n");partial.replace(out)
if __name__=="__main__": main()
