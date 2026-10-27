"""Render the shura.cpp wordmark candidates from banner.html to PNG (light and dark) into candidates/ next to this file.
The chosen one (V2, the memory bar) is copied to wordmark-light.png and wordmark-dark.png.

Needs Playwright with Chromium (`pip install playwright && playwright install chromium`). The fonts are system fonts:
Montserrat, Source Code Pro, Droid Arabic Kufi.
"""
import asyncio, pathlib
from playwright.async_api import async_playwright

HERE = pathlib.Path(__file__).resolve().parent
SRC = (HERE / "banner.html").as_uri()
OUT = HERE / "candidates"

async def main():
    OUT.mkdir(exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        for v in ("1", "2", "3"):
            for theme in ("light", "dark"):
                h = 360 if v == "2" else 320
                page = await browser.new_page(viewport={"width": 1200, "height": h}, device_scale_factor=2)
                await page.goto(f"{SRC}?v={v}&theme={theme}")
                await page.wait_for_timeout(300)
                await page.screenshot(path=str(OUT / f"v{v}-{theme}.png"), clip={"x": 0, "y": 0, "width": 1200, "height": h})
                await page.close()
        await browser.close()

asyncio.run(main())
