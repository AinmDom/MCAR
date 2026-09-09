"""Generate frozen Q14/Q50 Stage-C and Stage-D member configs."""
import copy, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
base_c=json.loads((ROOT/'configs/experiments/sonicom_film_siren_gl_final_d1d2_notch_seed20260821_e130.json').read_text())
base_d=json.loads((ROOT/'configs/experiments/sonicom_film_siren_spectral_cnn_final_seed20260821_e190.json').read_text())
seeds=(20260821,20260822,20260823)
film_ckpt={s:f'artifacts/training/sonicom_fsc_q{{q}}_film_seed{s}_e130/last.pt' for s in seeds}
film_hash={20260821:'E37676D843D608B4DDD7B311EBA8B28A933183A03E7BAA49E1CF20E35F8B4129',20260822:'5463CFE9FA618F5CEA0ADF7415E7CEAC978D071C24FB2038772A968519808A1D',20260823:'DD3C55F6E7FC32A5AC089272978BEF771009BF140768E19A33EDD74E1C704070'}
for q in (14,50):
 for s in seeds:
  c=copy.deepcopy(base_c); c.update(experiment_id=f'FSC-Q{q}-FILM-E130-SEED{s}',dataset_root=f'data/processed/sonicom_fsc_q{q}_residual_v1',q_csv='configs/data/sonicom_nested_sparse_grid_q14_q26_q50_v1.csv',q_normalization=f'configs/data/sonicom_fsc_q{q}_magnitude_normalization_v1.json',observation_count=q,run_name=f'sonicom_fsc_q{q}_film_seed{s}_e130',model_version=f'fsc-q{q}-film-e130-v1',require_clean_git=False); c['q26_csv']=c['q_csv']; c['q26_normalization']=c['q_normalization']; (ROOT/f'configs/experiments/sonicom_fsc_q{q}_film_seed{s}_e130.json').write_text(json.dumps(c,indent=2)+'\n')
  d=copy.deepcopy(base_d); d.update(experiment_id=f'FSC-Q{q}-CNN-E190-SEED{s}',dataset_root=f'data/processed/sonicom_fsc_q{q}_residual_v1',q_csv='configs/data/sonicom_nested_sparse_grid_q14_q26_q50_v1.csv',q_normalization=f'configs/data/sonicom_fsc_q{q}_magnitude_normalization_v1.json',observation_count=q,initial_film_checkpoint=film_ckpt[s].format(q=q),initial_film_checkpoint_sha256=film_hash[s],run_name=f'sonicom_fsc_q{q}_cnn_seed{s}_e190',model_version=f'fsc-q{q}-cnn-e190-v1',require_clean_git=False); d['q26_csv']=d['q_csv']; d['q26_normalization']=d['q_normalization']; (ROOT/f'configs/experiments/sonicom_fsc_q{q}_cnn_seed{s}_e190.json').write_text(json.dumps(d,indent=2)+'\n')
