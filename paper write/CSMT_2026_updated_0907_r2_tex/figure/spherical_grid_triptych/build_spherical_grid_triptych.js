/* Build the publication triptych from the frozen nested Q14/Q26/Q50 grid.
 *
 * The orthographic projection and fixed yaw/pitch are inherited from the
 * project's spherical-grid web viewer.  The output is an offline HTML page
 * containing a vector SVG, plus provenance and QA-ready geometry metadata.
 */
const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..', '..', '..', '..');
const gridPath = path.join(root, 'configs', 'data', 'sonicom_nested_sparse_grid_q14_q26_q50_v1.csv');
const geometryPath = path.join(root, 'configs', 'data', 'sonicom_nested_sparse_grid_q14_q26_q50_v1.json');
const out = __dirname;
const WIDTH = 4800;
const HEIGHT = 1600;
const yaw = -0.62;
const pitch = -0.35;
const levels = [14, 26, 50];

function parseCsv(text) {
  const [header, ...lines] = text.trim().split(/\r?\n/);
  const names = header.split(',');
  return lines.map(line => Object.fromEntries(line.split(',').map((value, i) => [names[i], value])));
}
function esc(value) {
  return String(value).replace(/[&<>\"]/g, ch => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[ch]));
}
function rotate(x, y, z) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = cy * x + sy * z;
  const z1 = -sy * x + cy * z;
  return { x: x1, y: cp * y - sp * z1, z: sp * y + cp * z1 };
}
function project(point, cx, cy, radius) {
  const az = Number(point.azimuth_deg) * Math.PI / 180;
  const el = Number(point.elevation_deg) * Math.PI / 180;
  const r = rotate(Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el));
  return { x: cx + r.x * radius, y: cy - r.y * radius, depth: r.z };
}
function gridPath3d(points, cx, cy, radius) {
  return points.map(([x, y, z], i) => {
    const r = rotate(x, y, z);
    return `${i ? 'L' : 'M'}${(cx + r.x * radius).toFixed(2)},${(cy - r.y * radius).toFixed(2)}`;
  }).join(' ');
}
function circlePoints(latitudeDeg) {
  const e = latitudeDeg * Math.PI / 180;
  return Array.from({ length: 121 }, (_, i) => {
    const a = 2 * Math.PI * i / 120;
    return [Math.cos(e) * Math.cos(a), Math.cos(e) * Math.sin(a), Math.sin(e)];
  });
}
function meridianPoints(longitudeDeg) {
  const a = longitudeDeg * Math.PI / 180;
  return Array.from({ length: 121 }, (_, i) => {
    const t = -Math.PI / 2 + Math.PI * i / 120;
    return [Math.cos(t) * Math.cos(a), Math.cos(t) * Math.sin(a), Math.sin(t)];
  });
}
function minSeparation(points) {
  let best = Infinity;
  for (let i = 0; i < points.length; i++) for (let j = i + 1; j < points.length; j++) {
    const ai = Number(points[i].azimuth_deg) * Math.PI / 180, ei = Number(points[i].elevation_deg) * Math.PI / 180;
    const aj = Number(points[j].azimuth_deg) * Math.PI / 180, ej = Number(points[j].elevation_deg) * Math.PI / 180;
    const dot = Math.sin(ei) * Math.sin(ej) + Math.cos(ei) * Math.cos(ej) * Math.cos(ai - aj);
    best = Math.min(best, Math.acos(Math.max(-1, Math.min(1, dot))) * 180 / Math.PI);
  }
  return best;
}

const rows = parseCsv(fs.readFileSync(gridPath, 'utf8'));
const frozen = JSON.parse(fs.readFileSync(geometryPath, 'utf8'));
const grids = Object.fromEntries(levels.map(n => [n, rows.filter(r => Number(r.direction_count) === n)]));
for (const n of levels) {
  if (grids[n].length !== n) throw new Error(`Expected ${n} directions, found ${grids[n].length}.`);
  const delta = Math.abs(minSeparation(grids[n]) - frozen.geometry[`Q${n}`].minimum_pairwise_separation_deg);
  if (delta > 1e-8) throw new Error(`Q${n} minimum separation does not match frozen geometry.`);
}
for (let i = 0; i < levels.length - 1; i++) {
  const low = new Set(grids[levels[i]].map(r => r.source_index_zero_based));
  const high = new Set(grids[levels[i + 1]].map(r => r.source_index_zero_based));
  if (![...low].every(index => high.has(index))) throw new Error(`Q${levels[i]} is not nested in Q${levels[i + 1]}.`);
}

const colors = { q14: '#2257D6', q26: '#0D9B8A', q50: '#E08A00' };
function pointClass(point, n) {
  const index = point.source_index_zero_based;
  if (new Set(grids[14].map(r => r.source_index_zero_based)).has(index)) return 'q14';
  if (new Set(grids[26].map(r => r.source_index_zero_based)).has(index)) return 'q26';
  return 'q50';
}
function panel(n, panelIndex) {
  const cell = WIDTH / 3;
  const cx = cell * (panelIndex + 0.5);
  const cy = 850;
  const radius = 530;
  const points = grids[n];
  const gridlines = [];
  for (const lat of [-60, -30, 0, 30, 60]) gridlines.push(`<path class="gridline" d="${gridPath3d(circlePoints(lat), cx, cy, radius)}"/>`);
  for (const lon of [0, 30, 60, 90, 120, 150]) gridlines.push(`<path class="gridline" d="${gridPath3d(meridianPoints(lon), cx, cy, radius)}"/>`);
  const dots = points.map(p => ({ p, q: project(p, cx, cy, radius) })).sort((a, b) => a.q.depth - b.q.depth).map(({ p, q }) => {
    const opacity = (0.42 + 0.58 * Math.max(0, Math.min(1, (q.depth + 1) / 2))).toFixed(3);
    return `<circle cx="${q.x.toFixed(2)}" cy="${q.y.toFixed(2)}" r="18" fill="${colors[pointClass(p, n)]}" fill-opacity="${opacity}" stroke="#FFFFFF" stroke-width="4"/>`;
  }).join('');
  const spacing = frozen.geometry[`Q${n}`].minimum_pairwise_separation_deg.toFixed(1);
  const letter = String.fromCharCode(97 + panelIndex);
  return `<g class="panel">
    <text x="${cell * panelIndex + 104}" y="150" class="letter">${letter}</text>
    <text x="${cell * panelIndex + 175}" y="150" class="title">Q${n}</text>
    <text x="${cell * panelIndex + 175}" y="205" class="subtitle">${n} observed directions · minimum separation ${spacing}°</text>
    <circle cx="${cx}" cy="${cy}" r="${radius}" fill="#F7FAFE" stroke="#AAB9CC" stroke-width="4"/>
    ${gridlines.join('')}
    <circle cx="${cx}" cy="${cy}" r="${radius}" class="rim"/>
    ${dots}
  </g>`;
}

const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="8in" height="2.667in" viewBox="0 0 ${WIDTH} ${HEIGHT}" role="img" aria-labelledby="title description">
  <title id="title">Nested Q14, Q26 and Q50 spherical observation grids</title>
  <desc id="description">Three matched orthographic views of the frozen SONICOM observation grids. Blue marks belong to Q14; teal marks are added at Q26; amber marks are added at Q50.</desc>
  <style>
    .letter{font:700 64px Arial,Helvetica,sans-serif;fill:#17233C}.title{font:700 72px Arial,Helvetica,sans-serif;fill:#17233C}.subtitle{font:400 44px Arial,Helvetica,sans-serif;fill:#5B687D}.gridline{fill:none;stroke:#C9D4E2;stroke-width:3;stroke-dasharray:10 12}.rim{fill:none;stroke:#8FA0B8;stroke-width:5}.legend{font:400 44px Arial,Helvetica,sans-serif;fill:#3E4C63}.note{font:400 42px Arial,Helvetica,sans-serif;fill:#65718A}
  </style>
  <rect width="${WIDTH}" height="${HEIGHT}" fill="#FFFFFF"/>
  <line x1="1600" y1="100" x2="1600" y2="1310" stroke="#E1E7F0" stroke-width="3"/>
  <line x1="3200" y1="100" x2="3200" y2="1310" stroke="#E1E7F0" stroke-width="3"/>
  ${panel(14, 0)}${panel(26, 1)}${panel(50, 2)}
  <g transform="translate(905 1455)">
    <circle cx="0" cy="-10" r="16" fill="${colors.q14}" stroke="#FFFFFF" stroke-width="3"/><text x="34" y="1" class="legend">Q14 core</text>
    <circle cx="445" cy="-10" r="16" fill="${colors.q26}" stroke="#FFFFFF" stroke-width="3"/><text x="479" y="1" class="legend">Added at Q26</text>
    <circle cx="965" cy="-10" r="16" fill="${colors.q50}" stroke="#FFFFFF" stroke-width="3"/><text x="999" y="1" class="legend">Added at Q50</text>
  </g>
  <text x="2400" y="1542" text-anchor="middle" class="note">Frozen nested SONICOM observation grids: Q14 ⊂ Q26 ⊂ Q50. All panels share the same orthographic view.</text>
</svg>`;

const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Q14, Q26 and Q50 spherical grids</title><style>@page{size:8in 2.667in;margin:0}html,body{margin:0;padding:0;background:#fff}svg{display:block;width:8in;height:2.667in}</style></head><body>${svg}</body></html>`;
const provenance = {
  figure: 'Nested Q14/Q26/Q50 spherical observation-grid triptych',
  renderer: 'offline web SVG; orthographic projection and fixed view inherited from the project spherical-grid viewer',
  source_grid: path.relative(root, gridPath).replace(/\\/g, '/'),
  geometry_report: path.relative(root, geometryPath).replace(/\\/g, '/'),
  source_indices_zero_based: Object.fromEntries(levels.map(n => [`Q${n}`, grids[n].map(r => Number(r.source_index_zero_based))])),
  minimum_pairwise_separation_deg: Object.fromEntries(levels.map(n => [`Q${n}`, frozen.geometry[`Q${n}`].minimum_pairwise_separation_deg])),
  nesting: 'Q14 subset Q26 subset Q50',
  canvas: { width_px: WIDTH, height_px: HEIGHT, output_dpi: 600, physical_size_in: [8, 2.667] },
  panel_geometry_px: levels.map((n, i) => ({ panel: String.fromCharCode(97 + i), grid: `Q${n}`, center: [WIDTH / 3 * (i + 0.5), 850], radius: 530 })),
  data_exclusions: 'None; all frozen directions in each requested Q level are rendered.',
  generated_by: 'node build_spherical_grid_triptych.js'
};
fs.writeFileSync(path.join(out, 'spherical_grid_triptych.svg'), svg);
fs.writeFileSync(path.join(out, 'spherical_grid_triptych.html'), html);
fs.writeFileSync(path.join(out, 'provenance.json'), JSON.stringify(provenance, null, 2) + '\n');
console.log(`Wrote SVG, HTML and provenance to ${out}`);

// Interactive offline page for author-controlled single-grid exports.
const explorerPayload = JSON.stringify(Object.fromEntries(levels.map(n => [`Q${n}`, grids[n].map(row => ({
  source_index_zero_based: Number(row.source_index_zero_based),
  azimuth_deg: Number(row.azimuth_deg), elevation_deg: Number(row.elevation_deg)
}))])));
const explorerHtml = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SONICOM spherical-grid explorer</title><style>
:root{--ink:#17233c;--muted:#60708a;--line:#dbe3ef;--blue:#2257d6;--teal:#0d9b8a;--amber:#e08a00}*{box-sizing:border-box}body{margin:0;background:#f4f7fb;color:var(--ink);font:16px Arial,Helvetica,sans-serif}.app{max-width:920px;margin:0 auto;padding:26px 20px 36px}.card{background:#fff;border:1px solid var(--line);border-radius:18px;padding:20px;box-shadow:0 12px 40px rgba(26,49,87,.10)}h1{font-size:26px;margin:0 0 6px}.intro{color:var(--muted);line-height:1.45;margin:0 0 18px}.controls{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 16px}.button{border:1px solid var(--line);border-radius:9px;background:#fff;color:var(--ink);padding:9px 14px;font-weight:700;cursor:pointer}.button.active{background:var(--blue);border-color:var(--blue);color:#fff}.button.export{background:var(--ink);color:#fff;border-color:var(--ink)}.slider{display:flex;align-items:center;gap:7px;border:1px solid var(--line);border-radius:9px;padding:6px 10px;background:#fbfcfe;font-size:13px;font-weight:700}.slider input{width:92px;accent-color:var(--blue)}.slider output{min-width:36px;color:var(--muted);font-weight:400;text-align:right}canvas{display:block;width:min(100%,720px);height:auto;margin:auto;border:1px solid var(--line);border-radius:12px;cursor:grab;touch-action:none;background:#fff}canvas.dragging{cursor:grabbing}.note{margin:14px 0 0;color:var(--muted);font-size:13px;line-height:1.5}.legend{display:flex;justify-content:center;flex-wrap:wrap;gap:14px;margin:13px 0 0;color:#47566f;font-size:13px}.swatch{width:12px;height:12px;border-radius:50%;display:inline-block;margin-right:5px;vertical-align:-1px;border:1px solid #fff;box-shadow:0 0 0 1px #cad5e5}</style></head><body><main class="app"><div class="card"><h1>Nested SONICOM spherical grids</h1><p class="intro">Select one Q level, then drag on the sphere to choose a view with stronger depth. The download preserves the chosen view as a 2400 × 2400 PNG.</p><div class="controls"><button class="button active" data-q="Q14">Q14</button><button class="button" data-q="Q26">Q26</button><button class="button" data-q="Q50">Q50</button><button class="button" id="reset">Reset view</button><label class="slider">Grid width <input id="gridWidth" type="range" min="0.5" max="4" step="0.1" value="1.4"><output id="gridWidthValue">1.4 px</output></label><label class="slider">Grid opacity <input id="gridOpacity" type="range" min="0.05" max="0.85" step="0.01" value="0.26"><output id="gridOpacityValue">26%</output></label><button class="button export" id="export">Download PNG</button></div><canvas id="sphere" width="2400" height="2400" aria-label="Interactive spherical sampling grid"></canvas><div class="legend"><span><i class="swatch" style="background:#2257d6"></i>Q14 core</span><span><i class="swatch" style="background:#0d9b8a"></i>added at Q26</span><span><i class="swatch" style="background:#e08a00"></i>added at Q50</span></div><p class="note">Same orthographic projection as the project grid viewer. Blue directions belong to Q14; teal and amber mark the nested additions. The sphere is fully interactive; rotate it before each export.</p></div></main><script>
const grids=${explorerPayload};const colors={q14:'#2257d6',q26:'#0d9b8a',q50:'#e08a00'};const core14=new Set(grids.Q14.map(p=>p.source_index_zero_based)),core26=new Set(grids.Q26.map(p=>p.source_index_zero_based));let selected='Q14',yaw=-.62,pitch=-.35,gridLineWidth=1.4,gridLineAlpha=.26,drag=false,last=null;const canvas=document.getElementById('sphere'),ctx=canvas.getContext('2d');
function rotate(x,y,z){const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),x1=cy*x+sy*z,z1=-sy*x+cy*z;return{x:x1,y:cp*y-sp*z1,z:sp*y+cp*z1}}
function point3(p){const a=p.azimuth_deg*Math.PI/180,e=p.elevation_deg*Math.PI/180;return[Math.cos(e)*Math.cos(a),Math.cos(e)*Math.sin(a),Math.sin(e)]}
function path3Hemisphere(points,cx,cy,r,front){let previous=null,open=false;ctx.beginPath();for(const v of points){const q=rotate(v[0],v[1],v[2]),current={x:cx+q.x*r,y:cy-q.y*r,z:q.z},visible=(q.z>=0)===front;if(previous){const previousVisible=(previous.z>=0)===front;if(previousVisible!==visible){const t=previous.z/(previous.z-current.z),ix=previous.x+t*(current.x-previous.x),iy=previous.y+t*(current.y-previous.y);if(previousVisible)ctx.lineTo(ix,iy);else{ctx.moveTo(ix,iy);ctx.lineTo(current.x,current.y)}open=visible}else if(visible){if(open)ctx.lineTo(current.x,current.y);else{ctx.moveTo(current.x,current.y);open=true}}}else if(visible){ctx.moveTo(current.x,current.y);open=true}previous=current}ctx.stroke()}
function ring(lat){const e=lat*Math.PI/180,a=[];for(let i=0;i<=120;i++){const t=2*Math.PI*i/120;a.push([Math.cos(e)*Math.cos(t),Math.cos(e)*Math.sin(t),Math.sin(e)])}return a}
function meridian(lon){const a=lon*Math.PI/180,p=[];for(let i=0;i<=240;i++){const t=-Math.PI/2+2*Math.PI*i/240;p.push([Math.cos(t)*Math.cos(a),Math.cos(t)*Math.sin(a),Math.sin(t)])}return p}
function cls(p){return core14.has(p.source_index_zero_based)?'q14':core26.has(p.source_index_zero_based)?'q26':'q50'}
function draw(){const W=1200,H=1200,S=2,cx=600,cy=650,r=440;ctx.setTransform(S,0,0,S,0,0);ctx.clearRect(0,0,W,H);ctx.fillStyle='#fff';ctx.fillRect(0,0,W,H);ctx.fillStyle='#17233c';ctx.font='700 42px Arial';ctx.fillText(selected,90,90);ctx.font='400 25px Arial';ctx.fillStyle='#60708a';ctx.fillText(grids[selected].length+' observed directions',90,128);const halo=ctx.createRadialGradient(cx-r*.3,cy-r*.36,r*.05,cx,cy,r*1.08);halo.addColorStop(0,'#fff');halo.addColorStop(.65,'#eef4fb');halo.addColorStop(1,'#cedbea');ctx.beginPath();ctx.arc(cx,cy,r,0,2*Math.PI);ctx.fillStyle=halo;ctx.fill();ctx.strokeStyle='#9cadc2';ctx.lineWidth=2.5;ctx.stroke();const gridCurves=[-60,-30,0,30,60].map(ring).concat([0,30,60,90,120,150].map(meridian));ctx.strokeStyle='rgba(112,135,166,'+(gridLineAlpha*.32).toFixed(3)+')';ctx.lineWidth=Math.max(.5,gridLineWidth*.85);ctx.setLineDash([7,10]);gridCurves.forEach(curve=>path3Hemisphere(curve,cx,cy,r,false));ctx.strokeStyle='rgba(112,135,166,'+gridLineAlpha+')';ctx.lineWidth=gridLineWidth;ctx.setLineDash([]);gridCurves.forEach(curve=>path3Hemisphere(curve,cx,cy,r,true));const pole=rotate(0,0,1),tip=rotate(0,0,1.30),px=cx+pole.x*r,py=cy-pole.y*r,tx=cx+tip.x*r,ty=cy-tip.y*r,dx=tx-px,dy=ty-py,len=Math.max(1,Math.hypot(dx,dy)),ux=dx/len,uy=dy/len,nx=-uy,ny=ux;ctx.save();ctx.globalAlpha=.48+.52*Math.max(0,Math.min(1,(pole.z+1)/2));ctx.strokeStyle='#d84855';ctx.lineWidth=5;ctx.beginPath();ctx.moveTo(px,py);ctx.lineTo(tx,ty);ctx.stroke();ctx.fillStyle='#d84855';ctx.beginPath();ctx.moveTo(tx,ty);ctx.lineTo(tx-22*ux+11*nx,ty-22*uy+11*ny);ctx.lineTo(tx-22*ux-11*nx,ty-22*uy-11*ny);ctx.closePath();ctx.fill();ctx.restore();ctx.beginPath();ctx.arc(cx,cy,r,0,2*Math.PI);ctx.strokeStyle='#7f94b0';ctx.lineWidth=3;ctx.stroke();const dots=grids[selected].map(p=>{const v=point3(p),q=rotate(v[0],v[1],v[2]);return{p,x:cx+q.x*r,y:cy-q.y*r,z:q.z}}).sort((a,b)=>a.z-b.z);for(const d of dots){const alpha=.38+.62*Math.max(0,Math.min(1,(d.z+1)/2));ctx.globalAlpha=alpha;ctx.beginPath();ctx.arc(d.x,d.y,14,0,2*Math.PI);ctx.fillStyle=colors[cls(d.p)];ctx.fill();ctx.globalAlpha=1;ctx.strokeStyle='#fff';ctx.lineWidth=3;ctx.stroke()}ctx.fillStyle='#65718a';ctx.font='400 20px Arial';ctx.textAlign='center';ctx.fillText('Drag to rotate · choose the view you want before downloading',600,1140);ctx.textAlign='left'}
function setQ(q){selected=q;document.querySelectorAll('[data-q]').forEach(b=>b.classList.toggle('active',b.dataset.q===q));draw()}document.querySelectorAll('[data-q]').forEach(b=>b.addEventListener('click',()=>setQ(b.dataset.q)));document.getElementById('reset').addEventListener('click',()=>{yaw=-.62;pitch=-.35;draw()});document.getElementById('export').addEventListener('click',()=>{const a=document.createElement('a');a.href=canvas.toDataURL('image/png');a.download='sonicom_'+selected.toLowerCase()+'_spherical_grid.png';a.click()});function bindSlider(inputId,outputId,setter,formatter){const input=document.getElementById(inputId),output=document.getElementById(outputId);input.addEventListener('input',()=>{setter(Number(input.value));output.textContent=formatter(Number(input.value));draw()})}bindSlider('gridWidth','gridWidthValue',v=>gridLineWidth=v,v=>v.toFixed(1)+' px');bindSlider('gridOpacity','gridOpacityValue',v=>gridLineAlpha=v,v=>Math.round(v*100)+'%');canvas.addEventListener('pointerdown',e=>{drag=true;last=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);canvas.classList.add('dragging')});canvas.addEventListener('pointermove',e=>{if(!drag)return;yaw+=(e.clientX-last[0])*.008;pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-last[1])*.008));last=[e.clientX,e.clientY];draw()});canvas.addEventListener('pointerup',e=>{drag=false;canvas.releasePointerCapture(e.pointerId);canvas.classList.remove('dragging')});canvas.addEventListener('pointercancel',()=>{drag=false;canvas.classList.remove('dragging')});draw();
</script></body></html>`;
fs.writeFileSync(path.join(out, 'spherical_grid_explorer.html'), explorerHtml);
console.log(`Wrote interactive page to ${path.join(out, 'spherical_grid_explorer.html')}`);
