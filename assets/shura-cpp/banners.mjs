// The README's pictures, light and dark, as SVG with the text set as outlines (an SVG shown through <img> on GitHub cannot
// load fonts, so outlines are the only way it looks the same everywhere).
//
//   npm i fontkit && node banners.mjs [out-dir]     Montserrat Medium and SemiBold must be installed (MONTSERRAT_DIR)
//
// banner-{light,dark}.svg    the mark and the name, at the top of the README
// governor-{light,dark}.svg  the card's memory to scale: the desktop grows into its free room at once, the expert cache
//                            gives back banks so the room is whole again, then grows back bank by bank when the desktop
//                            lets go (a CSS animation, still under prefers-reduced-motion)
import { createRequire } from "node:module";
import { writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const require = createRequire(import.meta.url);
const fontkit = require(process.env.FONTKIT || "fontkit");
const FONT_DIR = process.env.MONTSERRAT_DIR || "/usr/share/fonts/julietaula-montserrat-fonts";
const fonts = {
  500: fontkit.openSync(join(FONT_DIR, "Montserrat-Medium.otf")),
  600: fontkit.openSync(join(FONT_DIR, "Montserrat-SemiBold.otf")),
};
const OUT = process.argv[2] || ".";
mkdirSync(OUT, { recursive: true });

// room: the outline of the desktop's free room, the desktop's own amber (darker on the light ground, so it still reads)
const THEMES = {
  light: { ground: "#EEF4F3", ink: "#082A31", model: "#0F4A55", onModel: "#E6F1EF", cache: "#7DE3CB", desk: "#E4B54A",
           room: "#B4851A", tile: "#0F4A55", mint: "#7DE3CB" },
  dark:  { ground: "#082A31", ink: "#E6F1EF", model: "#1D5E6A", onModel: "#E6F1EF", cache: "#7DE3CB", desk: "#E4B54A",
           room: "#E4B54A", tile: "#0F4A55", mint: "#7DE3CB" },
};

const r1 = (s) => s.replace(/-?\d+\.\d+/g, (n) => String(Math.round(parseFloat(n) * 10) / 10));

// Text as one path. Returns its width so callers can align to the end.
function measure(str, size, weight = 500, tracking = 0) {
  const f = fonts[weight], s = size / f.unitsPerEm, run = f.layout(str);
  let w = 0;
  run.positions.forEach((p) => { w += p.xAdvance * s + tracking * size; });
  return w - tracking * size;
}
function text(str, { x, y, size, weight = 500, fill, anchor = "start", tracking = 0 }) {
  const f = fonts[weight], s = size / f.unitsPerEm, run = f.layout(str);
  const w = measure(str, size, weight, tracking);
  let pen = anchor === "end" ? x - w : anchor === "middle" ? x - w / 2 : x;
  let d = "";
  run.glyphs.forEach((g, i) => {
    const p = run.positions[i];
    d += g.path.scale(s, -s).translate(pen + p.xOffset * s, y - p.yOffset * s).toSVG();
    pen += p.xAdvance * s + tracking * size;
  });
  return `<path fill="${fill}" d="${r1(d)}"/>`;
}

// The mark (assets/shura-cpp/icon.svg), in its own 100-unit box.
function mark(t, id) {
  return `<rect x="2" y="2" width="96" height="96" rx="22" fill="${t.tile}"/>
  <mask id="${id}"><rect width="100" height="100" fill="#fff"/>
    <path d="M 100 34 H 66 A 16 16 0 1 0 50 50 A 16 16 0 1 1 34 66 H 0" fill="none" stroke="#000" stroke-width="10"/>
    <rect x="66" y="0" width="34" height="39" fill="#000"/><rect x="60" y="0" width="40" height="29" fill="#000"/>
    <rect x="0" y="61" width="34" height="39" fill="#000"/><rect x="0" y="71" width="40" height="29" fill="#000"/>
  </mask>
  <rect x="18" y="18" width="64" height="64" rx="17" fill="${t.mint}" mask="url(#${id})"/>`;
}

const svg = (w, h, body, title, style = "") =>
  `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${title}">
<title>${title}</title>${style ? `\n<style>${style}</style>` : ""}
${body}
</svg>
`;

const W = 1200, M = 48;
const NAME = { size: 116, weight: 600, tracking: -0.02 };
const LEDE = "A fork of llama.cpp that shares one gaming GPU with the desktop you keep using.";

// ---- governor ----------------------------------------------------------------------------------------------------
// One RX 7800 XT (16368 MiB) running Tiel-Coder-35B-A3B at 32k context (weights 10856 + KV/recurrent state 440 +
// compute 364 MiB, from the server log), the governor's 1.5 GiB floor + 256 MiB buffer, a desktop with a browser.
// The expert cache is what is left; it is drawn as 256 MiB banks.
function governor(t) {
  const H = 204, TOTAL = 16368, px = W / TOTAL;
  const MODEL = 10856 + 440 + 364, KEPT = 1536 + 256, DESK = 1100, BANK = 256;
  const banks = Math.floor((TOTAL - MODEL - KEPT - DESK) / BANK);              // 7
  const cache = banks * BANK, spare = TOTAL - MODEL - KEPT - DESK - cache;      // spare stays with the desktop
  const xModel = MODEL * px, pitch = BANK * px;
  const xKept = xModel + banks * pitch, wKept = KEPT * px;
  const xDesk = xKept + wKept, TRIM = 5, shift = TRIM * pitch;                  // the desktop takes 5 banks: 1.25 GiB
  const barY = 54, barH = 66;

  // The free VRAM is the desktop's room: an empty space outlined in the desktop's colour (dashed: empty, not a block), from
  // the end of the expert cache to the desktop. Its left edge follows the cache; the desktop, drawn over it, grows into it.
  // Timeline, 12 s: the desktop grows into its room at 1.5 s, the governor gives back five banks at 2 s and the room is
  // whole again; the desktop lets go at 5.5 s (the room is wider than it has to be), the cache waits for a calm 2 s and
  // grows back one bank every 0.4 s until the room is back to the floor plus the buffer.
  const T = 12, pc = (s) => `${((s / T) * 100).toFixed(3)}%`, slide = 0.15;
  const jump = 1.5, grown = 1.8, trim = 2.0, letGo = 5.5, regrow = [7.5, 7.9, 8.3, 8.7, 9.1];
  const tx = (v) => `{transform:translateX(${v ? `-${v.toFixed(1)}px` : 0})}`;
  const deskKF = `0%,${pc(jump)}${tx(0)}${pc(grown)},${pc(letGo)}${tx(shift)}${pc(letGo + 0.3)},100%${tx(0)}`;
  let roomKF = `0%,${pc(trim)}${tx(0)}${pc(trim + slide)}${tx(shift)}`;
  regrow.forEach((s, i) => { roomKF += `${pc(s)}${tx(shift - i * pitch)}${pc(s + slide)}${tx(shift - (i + 1) * pitch)}`; });
  roomKF += `100%${tx(0)}`;
  let style = `.m{animation-duration:${T}s;animation-iteration-count:infinite;animation-timing-function:linear}
.desk{animation-name:desk}@keyframes desk{${deskKF}}
.room{animation-name:room}@keyframes room{${roomKF}}`;
  let bankRects = "";
  for (let i = 0; i < banks; i++) {
    const x = xModel + i * pitch;
    const k = i - (banks - TRIM);                                                // 0..4 for the banks that go
    let cls = "";
    if (k >= 0) {
      const back = regrow[k];
      style += `\n.b${i}{animation-name:b${i}}@keyframes b${i}{0%,${pc(trim)}{opacity:1}${pc(trim + slide)},${pc(back)}{opacity:0}${pc(back + slide)},100%{opacity:1}}`;
      cls = ` class="m b${i}"`;
    }
    // banks touch: the gap between two of them is the ground showing through, 3 px
    bankRects += `<rect${cls} x="${x.toFixed(1)}" y="${barY}" width="${(pitch - 3).toFixed(1)}" height="${barH}" fill="${t.cache}"/>`;
  }
  style += `\n@media (prefers-reduced-motion: reduce){.m{animation:none}}`;

  // the room's outline: top, left and bottom edges; its right side runs on under the desktop, which closes it
  const sw = 2.5, top = barY + sw / 2, bottom = barY + barH - sw / 2, far = W + shift + 20;
  const room = `<path class="m room" d="M ${far} ${top} H ${(xKept + sw / 2).toFixed(1)} V ${bottom} H ${far}" fill="none" ` +
               `stroke="${t.room}" stroke-width="${sw}" stroke-dasharray="8 6"/>`;
  const lab = 21, mid = barY + barH / 2;
  const body = `<rect width="${W}" height="${H}" fill="${t.ground}"/>
${text("To scale: one 16 GiB card, Tiel-Coder-35B-A3B at 32k context", { x: M, y: barY - 14, size: lab, fill: t.ink })}
${text("expert cache", { x: xModel, y: barY - 14, size: lab, fill: t.ink })}
${text("desktop", { x: W - M, y: barY - 14, size: lab, fill: t.ink, anchor: "end" })}
<rect x="0" y="${barY}" width="${xModel.toFixed(1)}" height="${barH}" fill="${t.model}"/>
${text("Model weights, KV cache and compute: 11.4 GiB", { x: M, y: mid + 7.5, size: lab, fill: t.onModel })}
${bankRects}
${room}
<rect class="m desk" x="${xDesk.toFixed(1)}" y="${barY}" width="${(W - xDesk + shift + 4).toFixed(1)}" height="${barH}" fill="${t.desk}"/>
${text("free: the desktop takes it at once", { x: xDesk, y: barY + barH + 32, size: lab, fill: t.ink, anchor: "end" })}
${text("1.5 GiB floor + 256 MiB buffer", { x: xDesk, y: barY + barH + 58, size: 18, fill: t.ink, anchor: "end" })}`;
  return { svg: svg(W, H, body, "The VRAM governor: the desktop takes its free room at once, the expert cache gives the room back", style),
           facts: { banks, cache, spare, xModel, xKept, xDesk, shift, pitch,
                    scaleNoteEnd: M + measure("To scale: one 16 GiB card, Tiel-Coder-35B-A3B at 32k context", lab) } };
}

// ---- quiet -------------------------------------------------------------------------------------------------------
function quiet(t, theme) {
  const H = 280, size = 168, top = (H - size) / 2, x = M + size + 44;
  const lines = ["A fork of llama.cpp that shares one gaming GPU", "with the desktop you keep using."];
  const w = Math.ceil(x + Math.max(measure("shura.cpp", 124, NAME.weight, NAME.tracking), ...lines.map((l) => measure(l, 27))) + M);
  const body = `<rect width="${w}" height="${H}" fill="${t.ground}"/>
<g transform="translate(${M} ${top}) scale(${size / 100})">${mark(t, `quiet-${theme}`)}</g>
${text("shura.cpp", { x: x - 4, y: 142, ...NAME, size: 124, fill: t.ink })}
${lines.map((l, i) => text(l, { x, y: 196 + i * 38, size: 27, fill: t.ink })).join("\n")}`;
  return { svg: svg(w, H, body, "shura.cpp: " + LEDE) };
}

const report = {};
for (const theme of ["light", "dark"]) {
  const t = THEMES[theme];
  const g = governor(t);
  writeFileSync(join(OUT, `governor-${theme}.svg`), g.svg);
  report.governor = g.facts;
  writeFileSync(join(OUT, `banner-${theme}.svg`), quiet(t, theme).svg);
}
console.log(JSON.stringify(report));
