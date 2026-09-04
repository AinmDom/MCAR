"""Read supplied PDFs; generate local, page-indexed text and ARS structural sidecars."""
from pathlib import Path
import json
import subprocess
import sys
import hashlib
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
PREFLIGHT = Path('C:/Users/ZhuanZ/.codex/skills/academic-research-suite/ars/scripts/pdf_read_preflight.py')
IDS = {'HUTUBS': 'R9', 'Spatially Oriented': 'R8', 'sonicom': 'R6',
       'AKtools': 'R12', 'FiLM Visual': 'R7', 'SIREN': 'R11',
       'Magnitude-Corrected': 'R1', 'RANF': 'R5', 'Arend2019': 'R2S',
       'Arend2021': 'R3', 'Spatial_Upsampling': 'R4',
       'Measurement of': 'R13', 'Spatial audio signal': 'R14',
       'A Review on': 'R15', 'A_Survey': 'R16', 'Deep Learning for': 'R17',
       'Loss functions': 'R18'}

def main():
    index = []
    for source in sorted((ROOT / 'ref').rglob('*.pdf')):
        ref = next(value for key, value in IDS.items() if key in source.name)
        sidecar = OUT / f'{ref}.preflight.json'
        run = subprocess.run([sys.executable, str(PREFLIGHT), str(source)], check=True, capture_output=True, encoding='utf-8')
        sidecar.write_text(run.stdout, encoding='utf-8')
        reader = PdfReader(str(source))
        pages = [page.extract_text() or '' for page in reader.pages]
        payload = {'ref': ref, 'source': source.relative_to(ROOT).as_posix(),
                   'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                   'metadata': {str(k): str(v) for k,v in (reader.metadata or {}).items()},
                   'pages': [{'pdf_page': i+1, 'text': t} for i,t in enumerate(pages)]}
        (OUT / f'{ref}.text.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        preflight = json.loads(sidecar.read_text(encoding='utf-8'))
        index.append({'ref': ref, 'source': payload['source'], 'sha256': payload['sha256'],
                      'pages': len(pages), 'preflight': preflight})
        print(ref, len(pages), source.name, flush=True)
    (OUT / 'PDF_INDEX.json').write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
