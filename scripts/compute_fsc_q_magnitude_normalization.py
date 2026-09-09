"""Compute train-only per-ear/frequency sparse-condition magnitude statistics."""
import csv,json,sys
from pathlib import Path
import h5py,numpy as np
root=Path(__file__).resolve().parents[1]
for q in (14,50):
    ds=root/f'data/processed/sonicom_fsc_q{q}_residual_v1'
    split={r['subject_id']:r['split'] for r in csv.DictReader((root/'configs/data/sonicom_subject_split_v1.csv').open(encoding='utf-8-sig'))}
    values=[]; ids=[]; freq=None
    for label in sorted(split):
        if split[label]!='train': continue
        with h5py.File(ds/'subjects'/label/f'q{q}.h5') as h:
            idx=np.asarray(h['sparse_direction_indices_zero_based'][:],dtype=int).reshape(-1)
            ref=np.asarray(h['reference_logmag_db'][:],dtype=np.float32)
            ref=np.take(ref, idx, axis=1)
            values.append(ref); ids.append(int(label[1:])); freq=np.asarray(h['frequency_hz'][:],dtype=float).reshape(-1)
    x=np.concatenate(values,axis=1); mean=x.mean(axis=1); std=x.std(axis=1,ddof=0)
    out={'schema_version':'1.0','source':f'Q{q} reference_logmag_db from complete frozen train split','training_subject_count':len(ids),'training_subject_ids':ids,'q_direction_count':q,'per_ear_frequency_sample_count':int(x.shape[1]),'frequency_hz':freq.tolist(),'mean_db':mean.tolist(),'std_population_db':std.tolist(),'test_subjects_read':0}
    (root/f'configs/data/sonicom_fsc_q{q}_magnitude_normalization_v1.json').write_text(json.dumps(out,indent=2)+'\n')
    print(q,len(ids),x.shape,float(std.min()),float(std.max()))
