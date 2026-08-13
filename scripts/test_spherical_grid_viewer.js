const path = require('path');
const { chromium } = require('C:/Users/27334/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async () => {
  const browser = await chromium.launch({ headless: true, executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe' });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('pageerror', e => errors.push(e.message));
  await page.goto('file:///' + path.resolve('artifacts/visualizations/spherical_grid_viewer.html').replace(/\\/g, '/'));
  await page.waitForSelector('#gridSelect option', { state: 'attached' });

  const families = await page.locator('#familySelect option').evaluateAll(opts => opts.map(o => o.value));
  const results = {};
  async function choose(family, resolution) {
    await page.selectOption('#familySelect', family);
    if (resolution !== undefined) await page.selectOption('#gridSelect', String(resolution));
    results[family] = await page.evaluate(() => ({
      count: points.length,
      sumW: points.every(p => p.w != null) ? points.reduce((s, p) => s + p.w, 0) : null,
      maxNormError: Math.max(...points.map(p => Math.abs(p.x*p.x + p.y*p.y + p.z*p.z - 1)))
    }));
  }

  await choose('lebedev', 17);
  await choose('fliege', 10);
  results.fliege.negativeWeights = await page.evaluate(() => points.filter(p => p.w < 0).length);
  await page.selectOption('#colorMode', 'weight');
  await page.screenshot({ path: 'artifacts/visualizations/spherical_grid_viewer_preview.png', fullPage: true });
  await choose('sh', 8);
  results.sh.orthogonality = await page.evaluate(() => ({
    norm: 4*Math.PI*points.reduce((s,p)=>s+p.w*realSH(8,3,p.az,p.z)**2,0),
    cross: 4*Math.PI*points.reduce((s,p)=>s+p.w*realSH(8,3,p.az,p.z)*realSH(7,-2,p.az,p.z),0)
  }));
  await choose('sh_equal', 8);
  await choose('healpix', 4);
  await choose('fibonacci', 256);
  await choose('hammersley', 256);
  await choose('latlon', 12);
  await choose('icosphere', 4);
  await choose('cubesphere', 6);
  await choose('sonicom');
  results.sonicom.q26Count = await page.evaluate(() => points.filter(p => p.meta.is_q26).length);
  results.sonicom.evalCount = await page.evaluate(() => points.filter(p => p.meta.is_evaluation).length);
  await choose('q26');
  results.q26.indices = await page.evaluate(() => points.map(p => p.meta.source_index));
  console.log(JSON.stringify({ families, results, errors }));
  await browser.close();

  const expectedFamilies = ['lebedev','fliege','sh','sh_equal','healpix','fibonacci','hammersley','latlon','icosphere','cubesphere','sonicom','q26'];
  const expectedCounts = { lebedev:110, fliege:121, sh:153, sh_equal:324, healpix:192, fibonacci:256, hammersley:256, latlon:222, icosphere:162, cubesphere:216, sonicom:793, q26:26 };
  const expectedQ26 = [1,11,94,107,141,154,157,169,271,284,287,347,356,388,427,459,468,528,531,544,646,658,661,674,708,721];
  if (JSON.stringify(families) !== JSON.stringify(expectedFamilies)) process.exit(1);
  for (const [name, count] of Object.entries(expectedCounts)) {
    if (results[name].count !== count || results[name].maxNormError > 1e-12) process.exit(1);
    if (name !== 'q26' && Math.abs(results[name].sumW - 1) > 1e-11) process.exit(1);
  }
  if (Math.abs(results.sh.orthogonality.norm - 1) > 1e-11 || Math.abs(results.sh.orthogonality.cross) > 1e-11) process.exit(1);
  if (results.fliege.negativeWeights < 1) process.exit(1);
  if (results.sonicom.q26Count !== 26 || results.sonicom.evalCount !== 767) process.exit(1);
  if (JSON.stringify(results.q26.indices) !== JSON.stringify(expectedQ26) || errors.length) process.exit(1);
})();
