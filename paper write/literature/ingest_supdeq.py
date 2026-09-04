"""Bounded source intake; preserve the earlier 17-source index and all PDFs."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
from pypdf import PdfReader

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
source = ROOT / "ref/method/TASLP.2019.2908057.pdf"
preflight = Path("C:/Users/ZhuanZ/.codex/skills/academic-research-suite/ars/scripts/pdf_read_preflight.py")
run = subprocess.run([sys.executable, str(preflight), str(source)],
                     check=True, capture_output=True, encoding="utf-8")
(OUT / "R2.preflight.json").write_text(run.stdout, encoding="utf-8")
reader = PdfReader(source)
payload = {"ref": "R2", "source": source.relative_to(ROOT).as_posix(),
           "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
           "pages": [{"pdf_page": i + 1, "text": page.extract_text() or ""}
                     for i, page in enumerate(reader.pages)]}
(OUT / "R2.text.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
index = {"ref": "R2", "source": payload["source"], "sha256": payload["sha256"],
         "bytes": source.stat().st_size, "pages": len(reader.pages),
         "preflight": json.loads(run.stdout),
         "read_scope": "PDF pages 1-4: title, abstract, Introduction, Method/Fig.1 and beginning of Evaluation; not full paper",
         "prior_index_preserved": "PDF_INDEX.json (17-source historical snapshot)",
         "human_read_attestation": "not supplied"}
(OUT / "R2.index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(index, ensure_ascii=False))
