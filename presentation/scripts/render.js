#!/usr/bin/env node
/**
 * Presentation renderer — PaleoVox GUI · XIX Congreso Nacional de Paleontología.
 *
 * Uses the Playwright installation cached by npx (same tree as the poster
 * renderer), configured for the bundled Chromium/Chrome build.
 *
 * Usage:
 *   node presentation/scripts/render.js preview             -> renders/previews/slide_XX.png (2x)
 *   node presentation/scripts/render.js export              -> renders/PaleoVox_Presentacion_XIX_CNP.pdf
 *   node presentation/scripts/render.js all                 -> both
 *   node presentation/scripts/render.js preview-historia    -> renders/previews_historia/slide_XX.png (2x)
 *   node presentation/scripts/render.js export-historia     -> renders/PaleoVox_Historia_XIX_CNP.pdf
 *   node presentation/scripts/render.js all-historia        -> both
 *   node presentation/scripts/render.js preview-historia43  -> renders/previews_historia_4x3/slide_XX.png (2x)
 *   node presentation/scripts/render.js export-historia43   -> renders/PaleoVox_Historia_XIX_CNP_4x3.pdf
 *   node presentation/scripts/render.js all-historia43      -> both
 */
const fs = require('fs');
const path = require('path');

const PLAYWRIGHT_DIR = '/home/alan-amaro/.npm/_npx/9833c18b2d85bc59/node_modules';
const { chromium } = require(path.join(PLAYWRIGHT_DIR, 'playwright'));

const REPO = path.resolve(__dirname, '..', '..');
const OUT_DIR = path.join(REPO, 'presentation', 'renders');

const DECKS = {
  tecnico: {
    slidesFile: 'slides.html',
    previewDir: path.join(OUT_DIR, 'previews'),
    pdfName: 'PaleoVox_Presentacion_XIX_CNP.pdf',
    w: 1280,
    h: 720,
  },
  historia: {
    slidesFile: 'slides_historia.html',
    previewDir: path.join(OUT_DIR, 'previews_historia'),
    pdfName: 'PaleoVox_Historia_XIX_CNP.pdf',
    w: 1280,
    h: 720,
  },
  historia43: {
    slidesFile: 'slides_historia_4x3.html',
    previewDir: path.join(OUT_DIR, 'previews_historia_4x3'),
    pdfName: 'PaleoVox_Historia_XIX_CNP_4x3.pdf',
    w: 1280,
    h: 960,
  },
};

const W = 1280;
const H = 720;

async function launch() {
  const candidates = [
    '/home/alan-amaro/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome',
    '/home/alan-amaro/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome',
  ];
  const exe = candidates.find((p) => fs.existsSync(p));
  return chromium.launch({
    headless: true,
    chromiumSandbox: false,
    ...(exe ? { executablePath: exe } : {}),
  });
}

async function openSlides(browser, deck, deviceScaleFactor) {
  const slidesUrl = 'file://' + path.join(REPO, 'presentation', deck.slidesFile);
  const page = await browser.newPage({
    viewport: { width: deck.w || W, height: deck.h || H },
    deviceScaleFactor,
  });
  page.on('console', (msg) => {
    if (msg.type() === 'error' || msg.type() === 'warning')
      console.log('[console:' + msg.type() + ']', msg.text());
  });
  page.on('pageerror', (err) => console.log('[pageerror]', err.message));
  page.on('requestfailed', (req) =>
    console.log('[requestfailed]', req.url(), req.failure() && req.failure().errorText));
  await page.goto(slidesUrl, { waitUntil: 'networkidle' });
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(500);
  return page;
}

async function preview(browser, deck) {
  fs.mkdirSync(deck.previewDir, { recursive: true });
  const page = await openSlides(browser, deck, 2);
  const slides = page.locator('.slide');
  const count = await slides.count();
  for (let i = 0; i < count; i++) {
    const name = 'slide_' + String(i + 1).padStart(2, '0') + '.png';
    const file = path.join(deck.previewDir, name);
    await slides.nth(i).screenshot({ path: file });
    console.log('[preview]', name, fs.statSync(file).size, 'bytes');
  }
  console.log('[preview] slides rendered:', count, '(' + deck.slidesFile + ')');
  await page.close();
}

async function exportFinal(browser, deck) {
  const page = await openSlides(browser, deck, 1);
  const pdfPath = path.join(OUT_DIR, deck.pdfName);
  await page.pdf({
    path: pdfPath,
    printBackground: true,
    margin: { top: 0, right: 0, bottom: 0, left: 0 },
    preferCSSPageSize: true,
  });
  console.log('[export] PDF', pdfPath, fs.statSync(pdfPath).size, 'bytes');
  await page.close();
}

(async () => {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const mode = process.argv[2] || 'preview';
  const deck =
    mode.endsWith('-historia43') ? DECKS.historia43
    : mode.endsWith('-historia') ? DECKS.historia
    : DECKS.tecnico;
  const action = mode.replace(/-(historia43|historia)$/, '');
  const browser = await launch();
  try {
    if (action === 'preview' || action === 'all') await preview(browser, deck);
    if (action === 'export' || action === 'all') await exportFinal(browser, deck);
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error('RENDER FAILED:', e.message);
  process.exit(1);
});
