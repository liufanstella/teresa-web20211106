// Render designed covers for episodes that have no artwork yet.
// Usage: node make_covers.js <episodes.json> <outdir>
// episodes.json: [{ep, lines, en, scene}, ...]
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

(async () => {
  const eps = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
  const out = process.argv[3];
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1080, height: 1920 } });
  for (const cfg of eps) {
    const url = 'file://' + path.join(__dirname, 'cover.html') + '?cfg=' + encodeURIComponent(JSON.stringify(cfg));
    await page.goto(url);
    await page.waitForSelector('body[data-ready="1"]');
    await page.waitForTimeout(300);
    const file = path.join(out, `ep${String(cfg.ep).padStart(2, '0')}.png`);
    await page.screenshot({ path: file });
    console.log(file);
  }
  await browser.close();
})();
