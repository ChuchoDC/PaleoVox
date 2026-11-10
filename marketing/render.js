#!/usr/bin/env node
/**
 * Instagram campaign renderer: screenshots every `.card` of posts.html / stories.html
 * at 1x, so PNGs come out at exact Instagram sizes (1080×1350 posts, 1080×1920 stories).
 * Same Playwright install as presentation/scripts/render.js. Run from the repo root.
 *
 * Usage: node marketing/render.js [posts|stories|all]   (default: all)
 */
const fs = require('fs');
const path = require('path');

const PLAYWRIGHT_DIR = '/home/alan-amaro/.npm/_npx/9833c18b2d85bc59/node_modules';
const { chromium } = require(path.join(PLAYWRIGHT_DIR, 'playwright'));

const DIR = __dirname;
const SETS = {
  posts: { file: 'posts.html', out: path.join(DIR, 'renders', 'posts') },
  stories: { file: 'stories.html', out: path.join(DIR, 'renders', 'stories') },
};

async function launch() {
  const candidates = [
    '/home/alan-amaro/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome',
    '/home/alan-amaro/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome',
  ];
  const exe = candidates.find((p) => fs.existsSync(p));
  return chromium.launch({ headless: true, chromiumSandbox: false, ...(exe ? { executablePath: exe } : {}) });
}

async function render(browser, set) {
  fs.mkdirSync(set.out, { recursive: true });
  const page = await browser.newPage({ viewport: { width: 1200, height: 2000 }, deviceScaleFactor: 1 });
  page.on('console', (m) => ['error', 'warning'].includes(m.type()) && console.log('[console]', m.text()));
  page.on('pageerror', (e) => console.log('[pageerror]', e.message));
  page.on('requestfailed', (r) => console.log('[requestfailed]', r.url()));
  await page.goto('file://' + path.join(DIR, set.file), { waitUntil: 'networkidle' });
  await page.evaluate(() => document.fonts.ready);
  const cards = page.locator('.card');
  const n = await cards.count();
  for (let i = 0; i < n; i++) {
    const id = await cards.nth(i).getAttribute('id');
    const file = path.join(set.out, id + '.png');
    await cards.nth(i).screenshot({ path: file });
    console.log('[render]', path.relative(DIR, file));
  }
  await page.close();
}

(async () => {
  const which = process.argv[2] || 'all';
  const browser = await launch();
  try {
    for (const [name, set] of Object.entries(SETS))
      if (which === 'all' || which === name) await render(browser, set);
  } finally {
    await browser.close();
  }
})().catch((e) => { console.error('RENDER FAILED:', e.message); process.exit(1); });
