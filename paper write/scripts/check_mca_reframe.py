"""Read-only numerical/role regression check for the author-requested MCA reframe."""
import csv, hashlib, json, re, subprocess
from pathlib import Path
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper write" / "CSMT_2026"
BASE = "e44e93e8b45c3295e4dacef4d6bc82c2e3f8ce9f"
def git_bytes(path):
    return subprocess.check_output(["git", "show", BASE + ":" + path], cwd=ROOT)
def text(path):
    return path.read_text(encoding="utf-8-sig")
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def rows(source):
    return {l.split("&")[0].strip(): re.findall(r"[+-]?\d+\.\d+", l.split("&",1)[1])
            for l in source.splitlines() if "&" in l and re.search(r"\d+\.\d+", l.split("&",1)[1])}
source_rel = "results/sonicom_complete_ten_method_test_v1/aggregate_metrics.csv"
source = list(csv.DictReader((ROOT/source_rel).open(encoding="utf-8-sig", newline="")))
lookup = {(r["Method"], r["Endpoint"]): r for r in source}
methods = {"SH only":"SHOnly","SUpDEq SH":"SUpDEqSH","SUpDEq NN":"SUpDEqNN",
           "SUpDEq Bary.":"SUpDEqBary","MCA":"MCA","MCAR v3.5.1":"MCARv351",
           "FSP-AE":"FSPAE","RANF":"RANF","Hybrid":"HYBRID","Bounded E25":"BOUNDED"}
primary = ["FullSphereERB","Contralateral25ERB","ContralateralHighFrequency","HorizontalILDMAE"]
secondary = ["HFFirstDifferenceMAE","HFSecondDifferenceMAE","MultiScaleNotchDepthMAE","ERBBandILDMean"]
checks, count = {}, 0
for stem, endpoints in [("primary_results",primary),("spectral_results",secondary)]:
    before = rows(git_bytes("paper write/CSMT_2026/tables/"+stem+".tex").decode("utf-8-sig"))
    main = rows(text(PAPER/"tables"/(stem+".tex")))
    inside = rows(text(PAPER/"tables"/("internal_"+stem+".tex")))
    assert len(main)==8 and len(inside)==3
    assert not ({"MCAR v3.5.1","Bounded E25"} & main.keys())
    assert set(inside)=={"MCAR v3.5.1","Bounded E25","Hybrid"}
    for name, values in (main | inside).items():
        assert values == before[name], (name, values, before[name])
    assert set(main)|set(inside)==set(before)
    for group in (main,inside):
        for name, values in group.items():
            expected=[]
            for endpoint in endpoints:
                r=lookup[(methods[name],endpoint)]
                expected.append(f'{float(r["Mean"]):.4f}')
                if endpoints==primary:
                    expected.append(f'{float(r["SampleStd"]):.4f}')
            assert values==expected,(stem,name,values,expected)
            count+=len(values)
    marked={}
    for line in text(PAPER/"tables"/(stem+".tex")).splitlines():
        if "&" not in line or line.split("&")[0].strip() not in main:
            continue
        name=line.split("&")[0].strip()
        for i,cell in enumerate(line.split("&")[1:]):
            if re.search(r"\\(?:mathbf|textbf)\{",cell):
                minimum=min(float(lookup[(methods[m],endpoints[i])]["Mean"]) for m in main)
                assert float(lookup[(methods[name],endpoints[i])]["Mean"])==minimum
                marked[endpoints[i]]=name
    assert len(marked)==4
    checks[stem]={"main_rows":8,"internal_rows":3,"bold_minima":marked,
                  "original_rows_preserved":True,"frozen_csv_match":True}
for stem in ("component_results","direction_sensitivity"):
    rel="paper write/CSMT_2026/tables/"+stem+".tex"
    assert (ROOT/rel).read_bytes().replace(b"\r\n",b"\n")==git_bytes(rel).replace(b"\r\n",b"\n")
    checks[stem]={"unchanged":True}
allowed={"ablation.tex","internal_primary_results.tex","internal_spectral_results.tex"}
for f in list((PAPER/"sections").glob("*.tex"))+list((PAPER/"tables").glob("*.tex"))+[PAPER/"main.tex"]:
    t=text(f)
    assert r"\draftnote" not in t
    if re.search(r"MCAR|Bounded",t):
        assert f.name in allowed, f
old_passport=json.loads(git_bytes("paper write/CSMT_2026/material_passport.json"))
passport=json.loads(text(PAPER/"material_passport.json"))
assert passport["claim_intent_manifests"][1:]==old_passport["claim_intent_manifests"]
checks["old_claim_intents_preserved"]=True
old_results=git_bytes("paper write/CSMT_2026/sections/results.tex").decode("utf-8-sig")
new_results=text(PAPER/"sections/results.tex")+text(PAPER/"sections/ablation.tex")
decimal_values=lambda s: set(re.findall(r"\d+\.\d+",s))
assert decimal_values(old_results)<=decimal_values(new_results)
checks["original_result_decimal_values_retained"]=True
for rel in (source_rel,"results/sonicom_film_siren_spectral_cnn_final_e190_frozen_test_ten_method/paired_bootstrap.csv"):
    assert (ROOT/rel).read_bytes().replace(b"\r\n",b"\n")==git_bytes(rel).replace(b"\r\n",b"\n")
checks["frozen_sources_unchanged"]=True
build=PAPER/"build_mca_reframe"
log=text(build/"main.log")
assert not re.search(r"undefined citations|undefined references|Citation .+ undefined|Reference .+ undefined|Missing character|Overfull", log)
pdf=build/"main.pdf"
reader=PdfReader(pdf)
pages=[p.extract_text() or "" for p in reader.pages]
checks["pdf"]={"pages":len(pages),"bytes":pdf.stat().st_size,"sha256":sha(pdf),
               "text_nonempty_all_pages":all(p.strip() for p in pages),
               "internal_model_pages":[i+1 for i,p in enumerate(pages) if re.search(r"MCAR|Bounded",p)],
               "visual_review":"separate manual inspection required",
               "underfull_count":log.count("Underfull"),
               "font_shape_warning":bool(re.search(r"Font shape .* undefined",log)),
               "caption_default_warning":"Unknown document class" in log}
checks["main_and_internal_table_numeric_entries_checked"]=count
checks["test_subjects_read"]=0
checks["new_statistics"]=False
checks["full_ARS_or_submission_certification"]=False
checks["source_sha256"]={source_rel:sha(ROOT/source_rel)}
print(json.dumps(checks,ensure_ascii=False,indent=2))
