#!/usr/bin/env python3
"""Locked ARI FSC inference plus the three frozen Python metrics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from mcar.data import Normalization
from mcar.evaluation.deferred_secondary_metrics import enable_external_torchaudio, strict_hrir_from_selected_db
from mcar.evaluation.secondary_metrics import full_sphere_lsd, normalized_direction_weights, spectral_band_ild_profile
from mcar.fsp_ae_signal import estimate_itd_seconds
from mcar.models.film_siren import ConditionEncoderConfig, FilmSiren, FilmSirenConfig
from mcar.models.film_siren_spectral_cnn import FilmSirenSpectralCNN, SpectralRefinerConfig
from mcar.q26_condition import Q26MagnitudeNormalization
from mcar.training.train_film_siren import conditioned_coordinate_block
from mcar.training.train_siren import frequency_coordinates


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"empty output: {path}")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def metadata(handle: h5py.File) -> dict[str, object]:
    group = handle["strict_ild"]
    return {
        "mca_selected_phase_rad": np.asarray(group["mca_selected_phase_rad"][:]),
        "mca_outside_real": np.asarray(group["mca_outside_real"][:]),
        "mca_outside_imag": np.asarray(group["mca_outside_imag"][:]),
        "selected_bin_indices_zero_based": np.asarray(group["selected_bin_indices_zero_based"][:]),
        "outside_bin_indices_zero_based": np.asarray(group["outside_bin_indices_zero_based"][:]),
        "single_sided_frequency_count": int(group.attrs["single_sided_frequency_count"]),
        "hrir_length": int(group.attrs["hrir_length"]),
    }


def load_model(checkpoint_path: Path, device: torch.device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("training_stage") != "film_siren_spectral_cnn_stage_d" or int(payload["cycle"]) != 190:
        raise ValueError("checkpoint is not frozen Stage-D E190")
    film = FilmSiren(
        FilmSirenConfig(**payload["film_siren_configuration"]),
        ConditionEncoderConfig(**payload["condition_encoder_configuration"]),
    )
    refiner = dict(payload["spectral_refiner_configuration"])
    refiner["dilation_schedule"] = tuple(refiner["dilation_schedule"])
    model = FilmSirenSpectralCNN(film, SpectralRefinerConfig(**refiner), freeze_film_siren=True).to(device)
    model.load_state_dict(payload["model_state"]); model.eval()
    return payload, model


@torch.no_grad()
def predict(model, payload, feature_path: Path, qnorm, normalization, device) -> np.ndarray:
    with h5py.File(feature_path, "r") as handle:
        if str(handle.attrs["split"]) != "test" or int(handle.attrs["test_subjects_read"]) != 1:
            raise PermissionError("derived test feature provenance mismatch")
        reference = np.asarray(handle["reference_logmag_db"][:], dtype=np.float32)
        mca = np.asarray(handle["mca_logmag_db"][:], dtype=np.float32)
        correction = np.asarray(handle["correction_logmag_db"][:], dtype=np.float32)
        directions = np.asarray(handle["direction_features"][:], dtype=np.float32)
        frequency = np.asarray(handle["frequency_hz"][:], dtype=np.float32).reshape(-1)
        q26 = np.asarray(handle["sparse_direction_indices_zero_based"][:], dtype=np.int64).reshape(-1)
    condition = torch.from_numpy(qnorm.normalize(reference[:, q26, :])[None]).to(device)
    condition_xyz = torch.from_numpy(directions[q26, 2:5][None]).to(device)
    condition_mask = torch.ones((1, 26), dtype=torch.bool, device=device)
    latent = model.encode_condition(condition, condition_xyz, condition_mask)
    mapping = payload["experiment_configuration"]["frequency_mapping"]
    fcoord = torch.from_numpy(frequency_coordinates(
        frequency, mapping["mode"], mapping["frequency_minimum_hz"], mapping["frequency_maximum_hz"]
    )).to(device)
    log_frequency = torch.from_numpy(((np.log10(frequency)-normalization.log_frequency_mean)/normalization.log_frequency_std).astype(np.float32)).to(device)
    output = np.empty_like(mca)
    for start in range(0, 1550, 64):
        stop = min(start+64, 1550); xyz = torch.from_numpy(directions[start:stop,2:5]).to(device)
        local = mca[:,start:stop,:]
        query = conditioned_coordinate_block(xyz, fcoord, "global_plus_local_mca", local, normalization)
        final, _, _ = model.forward_grid(
            query, latent,
            torch.from_numpy((local-normalization.mca_mean)/normalization.mca_std).to(device),
            torch.from_numpy((correction[:,start:stop,:]-normalization.correction_mean)/normalization.correction_std).to(device),
            log_frequency, xyz,
        )
        output[:,start:stop,:] = (final*normalization.target_std+normalization.target_mean).cpu().numpy()
    if output.shape != (2,1550,463) or not np.isfinite(output).all():
        raise FloatingPointError("invalid FSC test prediction")
    return output


def itd(hrir: np.ndarray, cfg: dict, device: torch.device) -> np.ndarray:
    return estimate_itd_seconds(
        torch.from_numpy(hrir.astype(np.float32)).to(device), cfg["sampling_rate_hz"],
        upsampled_rate_hz=cfg["upsampled_rate_hz"], lowpass_hz=cfg["lowpass_hz"],
        maximum_itd_seconds=cfg["maximum_itd_seconds"],
        direction_batch_size=cfg["direction_batch_size"],
    ).cpu().numpy().astype(np.float64)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("config",type=Path); args=parser.parse_args()
    cfg=json.loads(args.config.resolve().read_text(encoding="utf-8"))
    access=json.loads((ROOT/cfg["test_access_state"]).read_text(encoding="utf-8"))
    if cfg["status"]!="frozen_pre_test" or access.get("status")!="completed" or access.get("test_subjects_read")!=22:
        raise PermissionError("completed one-shot raw test access required")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    enable_external_torchaudio(Path(cfg["runtime"]["torchaudio_site_packages"]))
    checkpoint=ROOT/cfg["checkpoint"]; expected=cfg["checkpoint_sha256"]
    if sha256(checkpoint)!=expected: raise ValueError("checkpoint hash mismatch")
    prediction_root=ROOT/cfg["prediction_root"]; result_root=ROOT/cfg["result_root"]
    if prediction_root.exists() or result_root.exists(): raise FileExistsError("locked-test output exists")
    prediction_root.mkdir(parents=True); result_root.mkdir(parents=True)
    device=torch.device("cuda"); payload,model=load_model(checkpoint,device)
    qnorm=Q26MagnitudeNormalization.from_json(ROOT/cfg["q26_normalization"])
    normalization=Normalization.from_json(ROOT/cfg["training_statistics"])
    subjects=[r for r in read_rows(ROOT/cfg["split_csv"]) if r["split"]=="test"]
    secondary=[]; inference=[]
    for index,row in enumerate(subjects,1):
        label=row["subject_id"]; feature=ROOT/cfg["feature_root"]/f"{label}.h5"; waveform=ROOT/cfg["prepared_waveform_root"]/f"{label}.h5"
        residual=predict(model,payload,feature,qnorm,normalization,device)
        with h5py.File(feature,"r") as h:
            reference=np.asarray(h["reference_logmag_db"][:],dtype=np.float32); mca=np.asarray(h["mca_logmag_db"][:],dtype=np.float32)
            directions=np.asarray(h["direction_features"][:],dtype=np.float64); frequency=np.asarray(h["frequency_hz"][:]).reshape(-1); mask=np.asarray(h["interpolation_evaluation_mask"][:],dtype=bool).reshape(-1); meta=metadata(h)
        with h5py.File(waveform,"r") as h: reference_hrir=np.asarray(h["hrir"][:],dtype=np.float32)
        if reference_hrir.shape!=(1550,2,256) or mask.sum()!=1524: raise ValueError("locked-test derived grid mismatch")
        pred_path=prediction_root/"subjects"/label/"prediction.h5"; pred_path.parent.mkdir(parents=True)
        with h5py.File(pred_path,"x") as h:
            h.create_dataset("predicted_residual_db",data=residual,compression="gzip",compression_opts=4); h.attrs["split"]="test"; h.attrs["subject_id"]=int(label[2:]); h.attrs["subject_label"]=label; h.attrs["checkpoint_sha256"]=expected; h.attrs["complete"]=1; h.attrs["test_subjects_read"]=1
        reference_itd=itd(reference_hrir,cfg["itd"],device); weights=normalized_direction_weights(directions[mask,5]); horizontal=mask & (np.abs(directions[:,1])<=1e-9)
        if np.count_nonzero(horizontal)!=86: raise ValueError("expected 86 non-observed horizontal directions")
        for method,predicted in (("MCA",mca),("FSC",mca+residual)):
            lsd,_,_=full_sphere_lsd(predicted,reference,mask,directions[:,5])
            _,_,band=spectral_band_ild_profile(predicted[:,horizontal,:],reference[:,horizontal,:],directions[horizontal,5],frequency)
            predicted_hrir=strict_hrir_from_selected_db(predicted,meta).permute(1,0,2).cpu().numpy()
            itd_error=float(np.sum(np.abs(itd(predicted_hrir,cfg["itd"],device)[mask]-reference_itd[mask])*1e6*weights))
            for metric,value,unit in (("ERBBandILDMean",band,"dB"),("ITDWeightedMAE",itd_error,"us"),("FullSphereLSD",lsd,"dB")):
                if not np.isfinite(value): raise FloatingPointError(f"nonfinite {label}/{method}/{metric}")
                secondary.append({"SubjectLabel":label,"SubjectID":int(label[2:]),"Method":method,"Metric":metric,"Unit":unit,"Value":format(value,".17g")})
        inference.append({"subject_label":label,"prediction":str(pred_path),"shape":[2,1550,463],"finite":True})
        print(f"ARI FSC locked-test inference/secondary [{index}/22] {label}",flush=True)
    write_csv(result_root/"secondary_subject_level.csv",secondary)
    (prediction_root/"inference_report.json").write_text(json.dumps({"status":"completed","split":"test","subject_count":22,"test_subjects_read":22,"checkpoint_sha256":expected,"subjects":inference},indent=2)+"\n",encoding="utf-8")


if __name__=="__main__": main()
