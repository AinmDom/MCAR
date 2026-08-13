"""Build the offline Lebedev / SH / SONICOM-Q26 spherical-grid viewer.

The script reuses the already embedded Lebedev tables from the original viewer,
then adds analytically generated Gauss-Legendre spherical-harmonic grids and the
project's locked SONICOM-Q26-v1 directions.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(
    r"C:\Users\27334\Documents\Codex\2026-07-31\lebedev\outputs\lebedev_grid_viewer.backup.html"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "visualizations" / "spherical_grid_viewer.html"
Q26_CSV = ROOT / "configs" / "data" / "sonicom_sparse_grid_q26_v1.csv"
SONICOM_REFERENCE_CSV = ROOT / "configs" / "data" / "sonicom_reference_grid_v1.csv"
FLIEGE_JSON = ROOT / "configs" / "data" / "fliege_maier_nodes_1_29.json"


def extract_lebedev(source: Path) -> tuple[list[dict[str, int]], dict[str, list[list[float]]]]:
    text = source.read_text(encoding="utf-8")
    catalog_match = re.search(r"const (?:CATALOG|LEBEDEV_CATALOG)\s*=\s*(.*?);\s*\n", text)
    grids_match = re.search(r"const (?:RAW_GRIDS|LEBEDEV_GRIDS)\s*=\s*(.*?);\s*\n", text)
    if not catalog_match or not grids_match:
        raise RuntimeError(f"Cannot find embedded Lebedev data in {source}")
    catalog = json.loads(catalog_match.group(1))
    grids = json.loads(grids_match.group(1))
    if len(catalog) != 32 or len(grids) != 32:
        raise RuntimeError("Expected 32 embedded Lebedev grids")
    return catalog, grids


def load_q26(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "q26_index": int(row["q26_index_zero_based"]),
                    "source_index": int(row["source_index_zero_based"]),
                    "azimuth": float(row["azimuth_deg"]),
                    "elevation": float(row["elevation_deg"]),
                    "colatitude": float(row["colatitude_deg"]),
                    "mirror_source_index": int(row["mirror_source_index_zero_based"]),
                    "role": row["selection_role"],
                }
            )
    if len(rows) != 26:
        raise RuntimeError(f"Expected 26 Q26 directions, found {len(rows)}")
    return rows


def load_sonicom_reference(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "source_index": int(row["source_index_zero_based"]),
                    "azimuth": float(row["azimuth_deg"]),
                    "elevation": float(row["elevation_deg"]),
                    "colatitude": float(row["colatitude_deg"]),
                    "weight": float(row["solid_angle_weight"]),
                    "is_q26": bool(int(row["is_q26_sparse_input"])),
                    "is_evaluation": bool(int(row["is_interpolation_evaluation"])),
                }
            )
    if len(rows) != 793:
        raise RuntimeError(f"Expected 793 SONICOM reference directions, found {len(rows)}")
    if sum(bool(row["is_q26"]) for row in rows) != 26:
        raise RuntimeError("Expected 26 Q26 directions in SONICOM reference grid")
    if abs(sum(float(row["weight"]) for row in rows) - 1.0) > 1e-12:
        raise RuntimeError("SONICOM reference weights are not normalized")
    return rows


def load_fliege(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    grids = payload.get("grids", [])
    if len(grids) != 29:
        raise RuntimeError(f"Expected 29 Fliege-Maier grids, found {len(grids)}")
    for order, grid in enumerate(grids, start=1):
        if grid["order"] != order or grid["points"] != (order + 1) ** 2:
            raise RuntimeError(f"Invalid Fliege-Maier metadata for N={order}")
        if abs(sum(row[3] for row in grid["values"]) - 1.0) > 1e-12:
            raise RuntimeError(f"Fliege-Maier weights do not normalize for N={order}")
    return grids


HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>球面采样网格可视化：Lebedev · 球谐 · Q26</title>
<style>
:root{--ink:#17233c;--muted:#65718a;--line:#dfe5ef;--paper:#f6f8fc;--panel:rgba(255,255,255,.95);--accent:#2257d6;--teal:#0d9b8a;--rose:#d84855;--shadow:0 18px 60px rgba(30,51,92,.12)}
*{box-sizing:border-box}html,body{height:100%;margin:0}body{overflow:hidden;color:var(--ink);background:radial-gradient(circle at 15% 15%,rgba(48,104,227,.1),transparent 32%),radial-gradient(circle at 86% 82%,rgba(13,155,138,.09),transparent 30%),var(--paper);font-family:Inter,"PingFang SC","Microsoft YaHei",system-ui,sans-serif}button,select,input{font:inherit}.app{height:100%;display:grid;grid-template-columns:338px 1fr;gap:18px;padding:18px}.sidebar,.stage{background:var(--panel);border:1px solid rgba(216,224,237,.9);border-radius:22px;box-shadow:var(--shadow)}.sidebar{overflow:auto;padding:23px}.stage{position:relative;min-width:0;overflow:hidden}.eyebrow{color:var(--accent);font-size:11px;font-weight:800;letter-spacing:.16em;text-transform:uppercase}h1{margin:7px 0 7px;font-size:24px;letter-spacing:-.035em;line-height:1.16}.intro{margin:0 0 17px;color:var(--muted);font-size:12.5px;line-height:1.6}.tabs{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin:12px 0 17px}.tab{min-height:39px;border:1px solid var(--line);border-radius:10px;background:#fff;color:#4c5971;font-size:11px;font-weight:750;cursor:pointer}.tab.active{color:#fff;background:var(--accent);border-color:var(--accent);box-shadow:0 5px 15px rgba(34,87,214,.2)}.field{margin:13px 0}.field label,.field-title{display:block;margin-bottom:7px;font-size:11.5px;font-weight:750;color:#45516a}.row{display:grid;grid-template-columns:1fr 1fr;gap:8px}select,.button{width:100%;min-height:40px;border:1px solid var(--line);border-radius:10px;background:#fff;color:var(--ink);padding:0 11px;outline:none}select:focus,.button:focus-visible{border-color:var(--accent);box-shadow:0 0 0 3px rgba(34,87,214,.12)}input[type=range]{width:100%;accent-color:var(--accent)}.range-head{display:flex;align-items:baseline;justify-content:space-between}.range-value{color:var(--accent);font-variant-numeric:tabular-nums;font-weight:750;font-size:11px}.stats{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:15px 0}.stat{padding:10px 11px;border-radius:12px;background:#f4f7fc;border:1px solid #e8edf5}.stat strong{display:block;font-size:17px;letter-spacing:-.025em;font-variant-numeric:tabular-nums}.stat span{display:block;margin-top:2px;color:var(--muted);font-size:9.5px}.info{padding:10px 11px;border-radius:11px;background:#eef4ff;border:1px solid #dbe7ff;color:#52627f;font-size:10.5px;line-height:1.5}.info strong{color:var(--accent)}.checks{display:grid;gap:8px;margin:14px 0}.check{display:flex;align-items:center;gap:8px;font-size:12px;color:#45516a;cursor:pointer}.check input{width:15px;height:15px;accent-color:var(--accent)}.actions{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:15px}.button{cursor:pointer;font-size:11.5px;font-weight:750;transition:.18s ease}.button:hover{transform:translateY(-1px);border-color:#b9c5d9}.button.primary{color:#fff;background:var(--accent);border-color:var(--accent)}.button.wide{grid-column:1/-1}.note{margin-top:17px;padding-top:14px;border-top:1px solid var(--line);color:var(--muted);font-size:9.7px;line-height:1.55}.note a{color:var(--accent)}canvas{display:block;width:100%;height:100%;cursor:grab;touch-action:none}canvas.dragging{cursor:grabbing}.topbar{position:absolute;top:18px;left:20px;right:20px;display:flex;justify-content:space-between;pointer-events:none}.badge,.hint{padding:9px 12px;border:1px solid rgba(211,221,236,.88);border-radius:11px;background:rgba(255,255,255,.84);backdrop-filter:blur(10px);color:#526078;font-size:11px;box-shadow:0 6px 24px rgba(35,55,91,.08)}.badge strong{color:var(--ink)}.legend{position:absolute;left:22px;bottom:20px;display:flex;align-items:center;gap:9px;font-size:10px;color:var(--muted)}.gradient{width:122px;height:8px;border-radius:99px;background:linear-gradient(90deg,#3157c8,#22a6b3,#f5b942,#d84855)}.tooltip{position:absolute;display:none;pointer-events:none;min-width:190px;padding:10px 12px;border-radius:11px;background:rgba(20,31,53,.93);color:#f6f8fd;font-size:10.5px;line-height:1.55;box-shadow:0 12px 30px rgba(17,27,48,.22);font-variant-numeric:tabular-nums}.tooltip b{color:#8edfd5}.hidden{display:none!important}
#familySelect{border-color:#b8c9ed;background:#f7faff;font-weight:750;color:#234a9f}
@media(max-width:860px){body{overflow:auto}.app{min-height:100%;height:auto;grid-template-columns:1fr;padding:10px}.sidebar{border-radius:17px;padding:18px}.stage{height:68vh;min-height:460px;border-radius:17px}.intro,.note{display:none}.stats{margin-bottom:10px}}
</style>
</head>
<body>
<main class="app">
<aside class="sidebar">
  <div class="eyebrow">Spherical sampling explorer</div>
  <h1>球面采样网格</h1>
  <p class="intro">统一比较常用球面求积、带限采样、等面积与准均匀网格；支持实球谐函数着色和径向形变。</p>
  <div class="field"><label for="familySelect">网格类型</label><select id="familySelect">
    <optgroup label="求积与球谐采样"><option value="lebedev">Lebedev 球面求积</option><option value="fliege">Fliege–Maier 优化求积</option><option value="sh">Gauss–Legendre 球谐网格</option><option value="sh_equal">等角球谐网格</option></optgroup>
    <optgroup label="等面积与准均匀采样"><option value="healpix">HEALPix 等面积网格</option><option value="fibonacci">Fibonacci 黄金螺旋</option><option value="hammersley">Hammersley 低差异序列</option></optgroup>
    <optgroup label="结构化几何网格"><option value="latlon">等经纬单元中心</option><option value="icosphere">二十面体测地网格</option><option value="cubesphere">立方球面网格</option></optgroup>
    <optgroup label="本项目"><option value="sonicom">SONICOM 完整参考网格（793点）</option><option value="q26">SONICOM-Q26-v1（26点）</option></optgroup>
  </select></div>
  <div class="field" id="gridField"><label id="gridLabel" for="gridSelect">Lebedev 代数精度</label><select id="gridSelect"></select></div>
  <div class="stats">
    <div class="stat"><strong id="orderStat">—</strong><span id="orderCaption">代数精度</span></div>
    <div class="stat"><strong id="pointStat">—</strong><span>球面节点数</span></div>
    <div class="stat"><strong id="minWeight">—</strong><span>最小归一化权重</span></div>
    <div class="stat"><strong id="sumWeight">—</strong><span>权重总和</span></div>
  </div>
  <div class="info" id="gridInfo"></div>
  <div class="field">
    <label for="colorMode">节点着色</label>
    <select id="colorMode"><option value="height">按 z 坐标</option><option value="weight">按积分权重</option><option value="sh">按实球谐函数 Yₗᵐ</option><option value="sampling">按 SONICOM 采样用途</option><option value="role">按 Q26 选点角色</option><option value="depth">按当前景深</option><option value="uniform">统一颜色</option></select>
  </div>
  <div id="harmonicControls" class="hidden">
    <div class="row">
      <div class="field"><label for="degree">球谐阶 l</label><select id="degree"></select></div>
      <div class="field"><label for="mode">球谐模态 m</label><select id="mode"></select></div>
    </div>
    <label class="check"><input id="deform" type="checkbox">按 Yₗᵐ 径向形变</label>
  </div>
  <div class="field"><div class="range-head"><span class="field-title">节点大小</span><span class="range-value" id="sizeValue">2.5 px</span></div><input id="pointSize" type="range" min="1" max="7" value="2.5" step="0.5"></div>
  <div class="checks">
    <label class="check"><input id="wireframe" type="checkbox" checked>显示经纬网和坐标轴</label>
    <label class="check" id="connectWrap"><input id="connectGrid" type="checkbox" checked>连接 SH 采样网格</label>
    <label class="check"><input id="autoRotate" type="checkbox">自动旋转</label>
  </div>
  <div class="actions"><button class="button" id="reset">恢复视角</button><button class="button" id="exportPng">导出 PNG</button><button class="button primary wide" id="exportCsv">导出当前网格 CSV</button></div>
  <div class="note">权重均归一化为总和 1；“近似面积权重”适合分布比较，不代表严格求积公式。部分 Fliege–Maier 阶数含带符号权重。Q26 来自项目锁定文件 <code>sonicom_sparse_grid_q26_v1.csv</code>。Lebedev 数据来源 <a href="https://github.com/ifilot/pylebedev" target="_blank" rel="noreferrer">PyLebedev 1.1.0</a>（GPL-3.0-or-later）。</div>
</aside>
<section class="stage" id="stage">
  <canvas id="canvas" aria-label="球面采样网格三维示意图"></canvas>
  <div class="topbar"><div class="badge"><strong id="badgeName">—</strong> · <strong id="badgePoints">—</strong> 点</div><div class="hint">拖拽旋转 · 滚轮缩放 · 悬停读数</div></div>
  <div class="legend"><span id="legendMin">−1</span><span class="gradient" id="gradient"></span><span id="legendMax">+1</span></div>
  <div class="tooltip" id="tooltip"></div>
</section>
</main>
<script>
const LEBEDEV_CATALOG=__LEBEDEV_CATALOG__;
const LEBEDEV_GRIDS=__LEBEDEV_GRIDS__;
const FLIEGE_GRIDS=__FLIEGE_GRIDS__;
const Q26=__Q26__;
const SONICOM_REFERENCE=__SONICOM_REFERENCE__;
const $=id=>document.getElementById(id),canvas=$('canvas'),ctx=canvas.getContext('2d',{alpha:false}),stage=$('stage');
let family='lebedev',selection=17,points=[],projected=[],dpr=1,width=0,height=0,yaw=-.62,pitch=-.35,zoom=1,dragging=false,lastX=0,lastY=0,lastTime=performance.now();
const factorial=n=>{let v=1;for(let i=2;i<=n;i++)v*=i;return v};
function assocLegendre(l,m,x){m=Math.abs(m);let pmm=1;if(m>0){let fact=1,s=Math.sqrt(Math.max(0,1-x*x));for(let i=1;i<=m;i++){pmm*=-fact*s;fact+=2}}if(l===m)return pmm;let pm1=x*(2*m+1)*pmm;if(l===m+1)return pm1;for(let n=m+2;n<=l;n++){const p=((2*n-1)*x*pm1-(n+m-1)*pmm)/(n-m);pmm=pm1;pm1=p}return pm1}
function realSH(l,m,az,z){const a=Math.abs(m),norm=Math.sqrt((2*l+1)/(4*Math.PI)*factorial(l-a)/factorial(l+a)),p=assocLegendre(l,a,z);if(m>0)return Math.sqrt(2)*norm*p*Math.cos(a*az);if(m<0)return Math.sqrt(2)*norm*p*Math.sin(a*az);return norm*p}
function gaussLegendre(n){const out=[];for(let i=1;i<=Math.ceil(n/2);i++){let x=Math.cos(Math.PI*(i-.25)/(n+.5)),pp=0;for(let k=0;k<30;k++){let p0=1,p1=x;for(let j=2;j<=n;j++){const p=((2*j-1)*x*p1-(j-1)*p0)/j;p0=p1;p1=p}pp=n*(x*p1-p0)/(x*x-1);const dx=p1/pp;x-=dx;if(Math.abs(dx)<1e-15)break}const w=2/((1-x*x)*pp*pp);out.push([-x,w]);if(out.length<n)out.push([x,w])}return out.sort((a,b)=>a[0]-b[0])}
function makeSHGrid(L){const gl=gaussLegendre(L+1),nphi=2*L+1,out=[];gl.forEach((r,ti)=>{const z=r[0],s=Math.sqrt(Math.max(0,1-z*z));for(let j=0;j<nphi;j++){const az=2*Math.PI*j/nphi;out.push({x:s*Math.cos(az),y:s*Math.sin(az),z,w:r[1]/(2*nphi),az,theta:Math.acos(z),ti,pj:j,index:out.length})}});return out}
function makeEqualSHGrid(L){const B=L+1,nt=2*B,np=2*B,out=[];let sum=0;for(let i=0;i<nt;i++){const theta=Math.PI*(i+.5)/nt,z=Math.cos(theta),s=Math.sin(theta);for(let j=0;j<np;j++){const az=2*Math.PI*j/np,w=s;sum+=w;out.push({x:s*Math.cos(az),y:s*Math.sin(az),z,w,az,theta,ti:i,pj:j,index:out.length})}}out.forEach(p=>p.w/=sum);return out}
function makeFibonacci(N){const out=[],gold=Math.PI*(3-Math.sqrt(5));for(let i=0;i<N;i++){const z=1-2*(i+.5)/N,s=Math.sqrt(1-z*z),az=(i*gold)%(2*Math.PI);out.push({x:s*Math.cos(az),y:s*Math.sin(az),z,w:1/N,az,theta:Math.acos(z),index:i})}return out}
function radicalInverse2(i){let x=0,f=.5;while(i>0){x+=f*(i&1);i>>=1;f*=.5}return x}
function makeHammersley(N){const out=[];for(let i=0;i<N;i++){const z=1-2*(i+.5)/N,s=Math.sqrt(1-z*z),az=2*Math.PI*radicalInverse2(i);out.push({x:s*Math.cos(az),y:s*Math.sin(az),z,w:1/N,az,theta:Math.acos(z),index:i})}return out}
function makeLatLon(nlat){const out=[],step=Math.PI/(nlat-1),nlon=2*(nlat-1),add=(theta,az,w)=>out.push({x:Math.sin(theta)*Math.cos(az),y:Math.sin(theta)*Math.sin(az),z:Math.cos(theta),w,az,theta,index:out.length});add(0,0,(1-Math.cos(step/2))/2);for(let i=1;i<nlat-1;i++){const theta=i*step,band=(Math.cos(theta-step/2)-Math.cos(theta+step/2))/2;for(let j=0;j<nlon;j++)add(theta,2*Math.PI*j/nlon,band/nlon)}add(Math.PI,0,(1-Math.cos(step/2))/2);return out}
function makeHEALPix(n){const out=[],addRing=(z,nr,offset)=>{const s=Math.sqrt(Math.max(0,1-z*z));for(let j=1;j<=nr;j++){const az=2*Math.PI*(j-offset)/nr;out.push({x:s*Math.cos(az),y:s*Math.sin(az),z,w:1/(12*n*n),az,theta:Math.acos(z),index:out.length})}};for(let ir=1;ir<4*n;ir++){if(ir<n)addRing(1-ir*ir/(3*n*n),4*ir,.5);else if(ir<=3*n){const offset=.5*(1+((ir+n)&1));addRing((2*n-ir)*2/(3*n),4*n,offset)}else{const k=4*n-ir;addRing(-1+k*k/(3*n*n),4*k,.5)}}return out}
function makeIcosphere(freq){const t=(1+Math.sqrt(5))/2,verts=[[-1,t,0],[1,t,0],[-1,-t,0],[1,-t,0],[0,-1,t],[0,1,t],[0,-1,-t],[0,1,-t],[t,0,-1],[t,0,1],[-t,0,-1],[-t,0,1]].map(v=>{const n=Math.hypot(...v);return v.map(x=>x/n)}),faces=[[0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],[1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],[3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],[4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]],map=new Map;for(const face of faces){const a=verts[face[0]],b=verts[face[1]],c=verts[face[2]];for(let i=0;i<=freq;i++)for(let j=0;j<=freq-i;j++){const k=freq-i-j,v=[(i*a[0]+j*b[0]+k*c[0])/freq,(i*a[1]+j*b[1]+k*c[1])/freq,(i*a[2]+j*b[2]+k*c[2])/freq],n=Math.hypot(...v),p=v.map(x=>x/n),key=p.map(x=>x.toFixed(9)).join(',');map.set(key,p)}}const arr=[...map.values()],w=1/arr.length;return arr.map((p,index)=>{const az=(Math.atan2(p[1],p[0])+2*Math.PI)%(2*Math.PI);return{x:p[0],y:p[1],z:p[2],w,az,theta:Math.acos(p[2]),index}})}
function makeCubeSphere(n){const raw=[],faces=[(u,v)=>[1,u,v],(u,v)=>[-1,u,v],(u,v)=>[u,1,v],(u,v)=>[u,-1,v],(u,v)=>[u,v,1],(u,v)=>[u,v,-1]];for(const face of faces)for(let i=0;i<n;i++)for(let j=0;j<n;j++){const u=-1+2*(i+.5)/n,v=-1+2*(j+.5)/n,a=face(u,v),r=Math.hypot(...a),p=a.map(x=>x/r),az=(Math.atan2(p[1],p[0])+2*Math.PI)%(2*Math.PI);raw.push({x:p[0],y:p[1],z:p[2],w:1/Math.pow(1+u*u+v*v,1.5),az,theta:Math.acos(p[2]),index:raw.length})}const sum=raw.reduce((s,p)=>s+p.w,0);raw.forEach(p=>p.w/=sum);return raw}
function addOptions(values,labeler){const sel=$('gridSelect');values.forEach(v=>sel.add(new Option(labeler(v),v)))}
function setFamily(next){family=next;$('familySelect').value=family;$('colorMode').value=(family==='q26'?'role':family==='sonicom'?'sampling':family==='sh'||family==='sh_equal'?'sh':'height');$('deform').checked=false;const sel=$('gridSelect');sel.innerHTML='';$('gridField').classList.toggle('hidden',family==='q26'||family==='sonicom');$('connectWrap').classList.toggle('hidden',family!=='sh'&&family!=='sh_equal');if(family==='lebedev'){selection=17;$('gridLabel').textContent='Lebedev 代数精度';LEBEDEV_CATALOG.forEach(v=>sel.add(new Option(`${v.order} 阶 · ${v.points} 点`,v.order)))}else if(family==='fliege'){selection=4;$('gridLabel').textContent='Fliege–Maier 空间阶 N';addOptions([...Array(29).keys()].map(x=>x+1),N=>`N = ${N} · ${(N+1)*(N+1)} 点`)}else if(family==='sh'){selection=3;$('gridLabel').textContent='最大球谐阶 L';addOptions([...Array(21).keys()],L=>`L = ${L} · ${(L+1)*(2*L+1)} 点`)}else if(family==='sh_equal'){selection=3;$('gridLabel').textContent='最大球谐阶 L';addOptions([...Array(21).keys()],L=>`L = ${L} · ${4*(L+1)*(L+1)} 点`)}else if(family==='healpix'){selection=4;$('gridLabel').textContent='HEALPix Nside';addOptions([1,2,4,8,16,32],n=>`Nside = ${n} · ${12*n*n} 点`)}else if(family==='fibonacci'||family==='hammersley'){selection=256;$('gridLabel').textContent='节点数 N';addOptions([32,50,64,100,128,200,256,500,1000,2000,5000],n=>`N = ${n}`)}else if(family==='latlon'){selection=12;$('gridLabel').textContent='纬度层数（含两极）';addOptions([4,6,9,12,18,24,36,48],n=>`${n} 层 · ${2+(n-2)*2*(n-1)} 点`)}else if(family==='icosphere'){selection=4;$('gridLabel').textContent='三角面细分频率';addOptions([1,2,3,4,5,6,8,10,12,16],n=>`频率 ${n} · ${10*n*n+2} 点`)}else if(family==='cubesphere'){selection=6;$('gridLabel').textContent='每个立方体面的边长采样数';addOptions([2,3,4,6,8,12,16,24,32],n=>`${n} × ${n} / 面 · ${6*n*n} 点`)}else selection=family;sel.value=String(selection);loadGrid()}
function loadGrid(){if(family==='lebedev'){const rows=LEBEDEV_GRIDS[String(selection)];points=rows.map((r,index)=>{const az=r[0]*Math.PI/180,theta=r[1]*Math.PI/180;return{x:Math.cos(az)*Math.sin(theta),y:Math.sin(az)*Math.sin(theta),z:Math.cos(theta),w:r[2],az,theta,index}})}else if(family==='fliege'){const rows=FLIEGE_GRIDS[Number(selection)-1].values;points=rows.map((r,index)=>{const az=(Math.atan2(r[1],r[0])+2*Math.PI)%(2*Math.PI),theta=Math.acos(Math.max(-1,Math.min(1,r[2])));return{x:r[0],y:r[1],z:r[2],w:r[3],az,theta,index}})}else if(family==='sh')points=makeSHGrid(Number(selection));else if(family==='sh_equal')points=makeEqualSHGrid(Number(selection));else if(family==='healpix')points=makeHEALPix(Number(selection));else if(family==='fibonacci')points=makeFibonacci(Number(selection));else if(family==='hammersley')points=makeHammersley(Number(selection));else if(family==='latlon')points=makeLatLon(Number(selection));else if(family==='icosphere')points=makeIcosphere(Number(selection));else if(family==='cubesphere')points=makeCubeSphere(Number(selection));else if(family==='sonicom')points=SONICOM_REFERENCE.map((r,index)=>{const az=r.azimuth*Math.PI/180,el=r.elevation*Math.PI/180;return{x:Math.cos(el)*Math.cos(az),y:Math.cos(el)*Math.sin(az),z:Math.sin(el),w:r.weight,az,theta:Math.PI/2-el,index,meta:r}});else points=Q26.map((r,index)=>{const az=r.azimuth*Math.PI/180,el=r.elevation*Math.PI/180;return{x:Math.cos(el)*Math.cos(az),y:Math.cos(el)*Math.sin(az),z:Math.sin(el),w:null,az,theta:Math.PI/2-el,index,meta:r}});updateStats();updateDegreeMenu();updateColorAvailability();$('tooltip').style.display='none';draw()}
function updateStats(){const ws=points.filter(p=>p.w!=null).map(p=>p.w),weighted=ws.length===points.length,n=Number(selection),neg=ws.filter(w=>w<0).length,meta={lebedev:[selection,'Lebedev 代数精度',`Lebedev ${selection} 阶`,`代数精度 <strong>${selection}</strong>；严格球面求积权重。`],fliege:[`N=${selection}`,'空间阶',`Fliege–Maier N=${selection}`,`优化球面求积，节点数 <strong>(N+1)² = ${points.length}</strong>；${neg?`当前阶含 <strong>${neg}</strong> 个负权重，属于原始求积系数。`:'当前阶权重均为正。'}`],sh:[`L=${selection}`,'最大球谐阶',`Gauss–Legendre SH L=${selection}`,`<strong>${n+1}</strong> 个 Gauss–Legendre 极角 × <strong>${2*n+1}</strong> 个等间隔方位角；严格乘积求积。`],sh_equal:[`L=${selection}`,'最大球谐阶',`等角 SH L=${selection}`,`带宽 B=${n+1}；<strong>${2*(n+1)} × ${2*(n+1)}</strong> 等角单元中心，采用归一化 sinθ 面积权重。`],healpix:[`Nside=${selection}`,'HEALPix 分辨率',`HEALPix Nside=${selection}`,`<strong>12 Nside²</strong> 个严格等面积、等纬环像素中心。`],fibonacci:[selection,'节点数 N',`Fibonacci N=${selection}`,`黄金角螺旋准均匀采样；每点等面积权重 <strong>1/N</strong>。`],hammersley:[selection,'节点数 N',`Hammersley N=${selection}`,`基数 2 radical-inverse 低差异序列；每点等权。`],latlon:[`${selection} 层`,'纬度层数',`等经纬 ${selection} 层`,`包含唯一南北极；中间纬圈各 ${2*(n-1)} 点，按纬度带面积赋权。`],icosphere:[`f=${selection}`,'细分频率',`二十面体 f=${selection}`,`正二十面体三角面重心细分并投影到球面；显示近似等面积权重。`],cubesphere:[`${selection}×${selection}`,'每面分辨率',`立方球面 ${selection}×${selection}`,`六个立方体面径向投影；采用立方球面 Jacobian 近似面积权重。`],sonicom:['793','完整参考方向',`SONICOM 完整网格`,`项目锁定的 <strong>793</strong> 个实测方向与固体角权重；其中 <strong>26</strong> 点为 Q26 输入，<strong>767</strong> 点为插值评估方向。`],q26:['Q26','SONICOM-Q26-v1','项目 Q26',`左右镜像对称；最小点间角 <strong>31.916°</strong>，覆盖半径 <strong>33.109°</strong>；三阶实 SH 设计矩阵秩 16。`]}[family];$('orderStat').textContent=meta[0];$('orderCaption').textContent=meta[1];$('badgeName').textContent=meta[2];$('gridInfo').innerHTML=meta[3];$('pointStat').textContent=points.length.toLocaleString('zh-CN');$('badgePoints').textContent=points.length.toLocaleString('zh-CN');$('minWeight').textContent=weighted?Math.min(...ws).toExponential(2):'—';$('sumWeight').textContent=weighted?ws.reduce((a,b)=>a+b,0).toFixed(10):'—'}
function updateDegreeMenu(){const d=$('degree'),old=Math.min(Number(d.value)||Math.min(Number(selection)||3,6),12),max=(family==='sh'||family==='sh_equal')?Math.max(0,Number(selection)):12;d.innerHTML='';for(let l=0;l<=max;l++)d.add(new Option(`l = ${l}`,l));d.value=String(Math.min(old,max));updateModeMenu()}
function updateModeMenu(){const l=Number($('degree').value),m=$('mode'),old=Number(m.value)||0;m.innerHTML='';for(let k=-l;k<=l;k++)m.add(new Option(`m = ${k}`,k));m.value=String(Math.max(-l,Math.min(l,old)));draw()}
function updateColorAvailability(){const roleOpt=$('colorMode').querySelector('option[value=role]'),samplingOpt=$('colorMode').querySelector('option[value=sampling]'),weightOpt=$('colorMode').querySelector('option[value=weight]');roleOpt.disabled=family!=='q26';samplingOpt.disabled=family!=='sonicom';weightOpt.disabled=family==='q26';if($('colorMode').value==='role'&&family!=='q26')$('colorMode').value='height';if($('colorMode').value==='sampling'&&family!=='sonicom')$('colorMode').value='height';if($('colorMode').value==='weight'&&family==='q26')$('colorMode').value='role';updateLegend()}
function rotate(x,y,z){const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),x1=cy*x+sy*z,z1=-sy*x+cy*z;return{x:x1,y:cp*y-sp*z1,z:sp*y+cp*z1}}
function projectPoint(p,radius,cx,cy,maxSH=1){let scale=1,sh=0;if($('colorMode').value==='sh'){sh=realSH(Number($('degree').value),Number($('mode').value),p.az,p.z);if($('deform').checked)scale=1+.42*sh/maxSH}const r=rotate(p.x*scale,p.y*scale,p.z*scale);return{sx:cx+r.x*radius,sy:cy-r.y*radius,depth:r.z,sh,scale}}
function shMax(){const l=Number($('degree').value),m=Number($('mode').value);let v=1e-12;for(const p of points)v=Math.max(v,Math.abs(realSH(l,m,p.az,p.z)));return v}
function resize(){const r=stage.getBoundingClientRect();dpr=Math.min(devicePixelRatio||1,2);width=r.width;height=r.height;canvas.width=Math.max(1,Math.round(width*dpr));canvas.height=Math.max(1,Math.round(height*dpr));canvas.style.width=width+'px';canvas.style.height=height+'px';ctx.setTransform(dpr,0,0,dpr,0,0);draw()}
function palette(t){t=Math.max(0,Math.min(1,t));const s=[[49,87,200],[34,166,179],[245,185,66],[216,72,85]],i=Math.min(2,Math.floor(t*3)),q=t*3-i,a=s[i],b=s[i+1];return`rgb(${Math.round(a[0]+(b[0]-a[0])*q)},${Math.round(a[1]+(b[1]-a[1])*q)},${Math.round(a[2]+(b[2]-a[2])*q)})`}
function roleColor(role){return role==='north_seed'?'#d84855':role==='median_seed'?'#f5b942':'#2257d6'}
function samplingColor(meta){return meta.is_q26?'#d84855':'#2257d6'}
function line3d(coords,radius,cx,cy,color,lw=1){ctx.beginPath();coords.forEach((v,i)=>{const r=rotate(v[0],v[1],v[2]),x=cx+r.x*radius,y=cy-r.y*radius;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.strokeStyle=color;ctx.lineWidth=lw;ctx.stroke()}
function drawWireframe(radius,cx,cy){if(!$('wireframe').checked)return;const c='rgba(74,97,136,.14)';for(let lat=-60;lat<=60;lat+=30){const a=lat*Math.PI/180,r=[];for(let k=0;k<=96;k++){const t=k*Math.PI*2/96;r.push([Math.cos(a)*Math.cos(t),Math.cos(a)*Math.sin(t),Math.sin(a)])}line3d(r,radius,cx,cy,c,.7)}for(let lon=0;lon<180;lon+=30){const a=lon*Math.PI/180,r=[];for(let k=0;k<=96;k++){const t=k*Math.PI*2/96;r.push([Math.cos(t)*Math.cos(a),Math.cos(t)*Math.sin(a),Math.sin(t)])}line3d(r,radius,cx,cy,c,.7)}const axes=[[[0,0,0],[1.17,0,0]],[[0,0,0],[0,1.17,0]],[[0,0,0],[0,0,1.17]]],colors=['rgba(216,72,85,.62)','rgba(13,155,138,.62)','rgba(34,87,214,.68)'],labels=['x','y','z'];axes.forEach((a,i)=>{line3d(a,radius,cx,cy,colors[i],1.2);const r=rotate(...a[1]);ctx.fillStyle=colors[i];ctx.font='600 12px Inter';ctx.fillText(labels[i],cx+r.x*radius+5,cy-r.y*radius-4)})}
function drawSHConnections(radius,cx,cy){if((family!=='sh'&&family!=='sh_equal')||!$('connectGrid').checked)return;const L=Number(selection),nt=family==='sh'?L+1:2*(L+1),np=family==='sh'?2*L+1:2*(L+1),c='rgba(34,87,214,.13)';for(let ti=0;ti<nt;ti++){const ring=[];for(let j=0;j<=np;j++){const p=points[ti*np+(j%np)];ring.push([p.x,p.y,p.z])}line3d(ring,radius,cx,cy,c,.7)}for(let j=0;j<np;j++){const mer=[];for(let ti=0;ti<nt;ti++){const p=points[ti*np+j];mer.push([p.x,p.y,p.z])}line3d(mer,radius,cx,cy,c,.7)}}
function draw(){if(!width||!height||!points.length)return;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,width,height);ctx.fillStyle='#fbfcff';ctx.fillRect(0,0,width,height);const cx=width/2,cy=height/2+8,radius=Math.min(width,height)*.365*zoom,halo=ctx.createRadialGradient(cx-radius*.28,cy-radius*.32,radius*.05,cx,cy,radius*1.12);halo.addColorStop(0,'#fff');halo.addColorStop(.72,'#f0f5fb');halo.addColorStop(1,'#dce5f1');ctx.beginPath();ctx.arc(cx,cy,radius,0,Math.PI*2);ctx.fillStyle=halo;ctx.fill();ctx.strokeStyle='rgba(76,101,142,.18)';ctx.stroke();drawWireframe(radius,cx,cy);drawSHConnections(radius,cx,cy);const ws=points.filter(p=>p.w!=null).map(p=>p.w),minW=ws.length?Math.min(...ws):0,maxW=ws.length?Math.max(...ws):1,maxSH=$('colorMode').value==='sh'?shMax():1,base=Number($('pointSize').value),mode=$('colorMode').value;projected=points.map(p=>({...projectPoint(p,radius,cx,cy,maxSH),p})).sort((a,b)=>a.depth-b.depth);for(const q of projected){let t=.22;if(mode==='height')t=(q.p.z+1)/2;else if(mode==='depth')t=(q.depth+1)/2;else if(mode==='weight')t=maxW===minW?.5:(q.p.w-minW)/(maxW-minW);else if(mode==='sh')t=.5+.5*q.sh/maxSH;const alpha=.42+.56*Math.max(0,Math.min(1,(q.depth+1)/2)),size=base*(.78+.36*Math.max(0,Math.min(1,(q.depth+1)/2)));ctx.globalAlpha=alpha;ctx.beginPath();ctx.arc(q.sx,q.sy,size,0,Math.PI*2);ctx.fillStyle=mode==='uniform'?'#2257d6':mode==='role'?roleColor(q.p.meta.role):mode==='sampling'?samplingColor(q.p.meta):palette(t);ctx.fill()}ctx.globalAlpha=1}
function updateLegend(){const mode=$('colorMode').value,ws=points.filter(p=>p.w!=null).map(p=>p.w);$('harmonicControls').classList.toggle('hidden',mode!=='sh');if(mode==='weight'&&ws.length){$('legendMin').textContent=Math.min(...ws).toExponential(2);$('legendMax').textContent=Math.max(...ws).toExponential(2)}else if(mode==='uniform'){$('legendMin').textContent='';$('legendMax').textContent=''}else if(mode==='role'){$('legendMin').textContent='镜像对';$('legendMax').textContent='种子点'}else if(mode==='sampling'){$('legendMin').textContent='评估 767';$('legendMax').textContent='Q26 输入 26'}else{$('legendMin').textContent='−';$('legendMax').textContent='+'}$('gradient').style.background=mode==='uniform'?'#2257d6':mode==='role'?'linear-gradient(90deg,#2257d6,#f5b942,#d84855)':mode==='sampling'?'linear-gradient(90deg,#2257d6 0 72%,#d84855 72% 100%)':'linear-gradient(90deg,#3157c8,#22a6b3,#f5b942,#d84855)';draw()}
function pointerPos(e){const r=canvas.getBoundingClientRect();return{x:e.clientX-r.left,y:e.clientY-r.top}}
canvas.addEventListener('pointerdown',e=>{dragging=true;lastX=e.clientX;lastY=e.clientY;canvas.setPointerCapture(e.pointerId);canvas.classList.add('dragging')});canvas.addEventListener('pointermove',e=>{if(dragging){yaw+=(e.clientX-lastX)*.008;pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-lastY)*.008));lastX=e.clientX;lastY=e.clientY;draw();$('tooltip').style.display='none';return}const m=pointerPos(e);let near=null,best=100;for(const q of projected){const d=Math.hypot(q.sx-m.x,q.sy-m.y);if(d<best){best=d;near=q}}if(near&&best<11){const p=near.p,t=$('tooltip'),az=p.az*180/Math.PI,el=90-p.theta*180/Math.PI,extra=p.meta?(family==='q26'?`<br>源索引 = ${p.meta.source_index}<br>镜像源索引 = ${p.meta.mirror_source_index}<br>角色 = ${p.meta.role}`:family==='sonicom'?`<br>源索引 = ${p.meta.source_index}<br>用途 = ${p.meta.is_q26?'Q26 稀疏输入':'插值评估'}`:''):'';t.innerHTML=`<b>节点 ${p.index+1}</b><br>方位角 = ${az.toFixed(3)}°<br>仰角 = ${el.toFixed(3)}°<br>x = ${p.x.toFixed(8)}<br>y = ${p.y.toFixed(8)}<br>z = ${p.z.toFixed(8)}${p.w==null?'':`<br>w = ${p.w.toExponential(8)}`}${extra}`;t.style.display='block';t.style.left=Math.min(width-210,m.x+14)+'px';t.style.top=Math.min(height-170,m.y+14)+'px'}else $('tooltip').style.display='none'});canvas.addEventListener('pointerup',e=>{dragging=false;canvas.releasePointerCapture(e.pointerId);canvas.classList.remove('dragging')});canvas.addEventListener('pointercancel',()=>{dragging=false;canvas.classList.remove('dragging')});canvas.addEventListener('mouseleave',()=>{if(!dragging)$('tooltip').style.display='none'});canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.58,Math.min(1.75,zoom*Math.exp(-e.deltaY*.001)));draw()},{passive:false});
function saveBlob(blob,name){const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}function slug(){return({lebedev:'lebedev_order',fliege:'fliege_maier_order',sh:'sh_gauss_legendre_L',sh_equal:'sh_equiangular_L',healpix:'healpix_nside',fibonacci:'fibonacci_N',hammersley:'hammersley_N',latlon:'latlon_levels',icosphere:'icosphere_frequency',cubesphere:'cubesphere_faceN',sonicom:'sonicom_reference_grid_v1',q26:'sonicom_q26_v1'}[family]||family)+((family==='q26'||family==='sonicom')?'':`_${selection}`)}function exportCSV(){const lines=['index,x,y,z,azimuth_deg,elevation_deg,normalized_weight,source_index_zero_based,is_q26_sparse_input,is_interpolation_evaluation,mirror_source_index_zero_based,selection_role'];points.forEach(p=>lines.push([p.index,p.x.toPrecision(16),p.y.toPrecision(16),p.z.toPrecision(16),(p.az*180/Math.PI).toFixed(10),(90-p.theta*180/Math.PI).toFixed(10),p.w==null?'':p.w.toPrecision(16),p.meta?.source_index??'',p.meta?.is_q26==null?'':Number(p.meta.is_q26),p.meta?.is_evaluation==null?'':Number(p.meta.is_evaluation),p.meta?.mirror_source_index??'',p.meta?.role??''].join(',')));saveBlob(new Blob(['\ufeff'+lines.join('\r\n')],{type:'text/csv;charset=utf-8'}),`${slug()}_${points.length}pts.csv`)}
$('familySelect').addEventListener('change',e=>setFamily(e.target.value));$('gridSelect').addEventListener('change',e=>{selection=Number(e.target.value);loadGrid()});$('colorMode').addEventListener('change',updateLegend);$('degree').addEventListener('change',updateModeMenu);$('mode').addEventListener('change',draw);$('deform').addEventListener('change',draw);$('pointSize').addEventListener('input',e=>{$('sizeValue').textContent=Number(e.target.value).toFixed(1)+' px';draw()});$('wireframe').addEventListener('change',draw);$('connectGrid').addEventListener('change',draw);$('reset').addEventListener('click',()=>{yaw=-.62;pitch=-.35;zoom=1;draw()});$('exportCsv').addEventListener('click',exportCSV);$('exportPng').addEventListener('click',()=>canvas.toBlob(b=>saveBlob(b,`${slug()}_${points.length}pts.png`),'image/png'));new ResizeObserver(resize).observe(stage);
function animate(now){const dt=Math.min(.05,(now-lastTime)/1000);lastTime=now;if($('autoRotate').checked&&!dragging){yaw+=dt*.22;draw()}requestAnimationFrame(animate)}setFamily('lebedev');resize();requestAnimationFrame(animate);
</script>
</body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-viewer", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    catalog, grids = extract_lebedev(args.source_viewer)
    q26 = load_q26(Q26_CSV)
    sonicom_reference = load_sonicom_reference(SONICOM_REFERENCE_CSV)
    fliege = load_fliege(FLIEGE_JSON)
    html = (
        HTML.replace("__LEBEDEV_CATALOG__", json.dumps(catalog, separators=(",", ":")))
        .replace("__LEBEDEV_GRIDS__", json.dumps(grids, separators=(",", ":")))
        .replace("__FLIEGE_GRIDS__", json.dumps(fliege, separators=(",", ":")))
        .replace("__Q26__", json.dumps(q26, ensure_ascii=False, separators=(",", ":")))
        .replace("__SONICOM_REFERENCE__", json.dumps(sonicom_reference, separators=(",", ":")))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1024 / 1024:.2f} MiB)")


if __name__ == "__main__":
    main()
