# shura.cpp icon: two mint forms share one square, the seam between them is an s.
# Regenerates icon.svg, icon-512.png, favicon-32.png, favicon-16.png and favicon.ico next to this file.
# Needs python3 with playwright (chromium) and Pillow; run from any directory.
import asyncio, pathlib
HERE = pathlib.Path(__file__).resolve().parent
TILE, MINT = "#0F4A55", "#7DE3CB"
# the seam is drawn wider at small sizes, or it disappears
TIERS = {"big": dict(r=16, w=10, rx=17), "small": dict(r=16.5, w=12, rx=18), "tiny": dict(r=15, w=14, rx=20)}
def tier(size): return "tiny" if size <= 16 else "small" if size <= 64 else "big"
def svg(size, t):
    r, w, rx = t["r"], t["w"], t["rx"]; h = w / 2
    seam = f"M 100 {50-r} H {50+r} A {r} {r} 0 1 0 50 50 A {r} {r} 0 1 1 {50-r} {50+r} H 0"
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 100 100">
  <rect x="2" y="2" width="96" height="96" rx="22" fill="{TILE}"/>
  <defs><mask id="seam"><rect width="100" height="100" fill="white"/>
    <path d="{seam}" fill="none" stroke="black" stroke-width="{w}"/>
    <rect x="{50+r}" y="0" width="{50-r}" height="{50-r+h}" fill="black"/>
    <rect x="60" y="0" width="40" height="{50-r-h}" fill="black"/>
    <rect x="0" y="{50+r-h}" width="{50-r}" height="{50-r+h}" fill="black"/>
    <rect x="0" y="{50+r+h}" width="40" height="{50-r-h}" fill="black"/>
  </mask></defs>
  <rect x="18" y="18" width="64" height="64" rx="{rx}" fill="{MINT}" mask="url(#seam)"/>
</svg>
'''
async def main():
    from playwright.async_api import async_playwright
    from PIL import Image
    (HERE / "icon.svg").write_text(svg(512, TIERS["big"]))
    out = {512: "icon-512.png", 32: "favicon-32.png", 16: "favicon-16.png"}
    async with async_playwright() as p:
        b = await p.chromium.launch()
        for size, name in out.items():
            tmp = HERE / f".icon-{size}.svg"; tmp.write_text(svg(size, TIERS[tier(size)]))
            pg = await b.new_page(viewport={"width": size, "height": size}, device_scale_factor=1)
            await pg.goto(tmp.as_uri()); await pg.wait_for_timeout(100)
            await pg.screenshot(path=str(HERE / name), omit_background=True); await pg.close(); tmp.unlink()
        await b.close()
    i16 = Image.open(HERE / "favicon-16.png").convert("RGBA"); i32 = Image.open(HERE / "favicon-32.png").convert("RGBA")
    i32.save(HERE / "favicon.ico", format="ICO", sizes=[(32, 32), (16, 16)], append_images=[i16])
if __name__ == "__main__":
    asyncio.run(main())
