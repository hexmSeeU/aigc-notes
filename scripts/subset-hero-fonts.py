"""Subset official OFL fonts for the existing homepage hero only.

Usage: python scripts/subset-hero-fonts.py /path/Caveat[wght].ttf /path/LongCang-Regular.ttf
Requires FontTools. See static/fonts/README.md for sources and their hashes.
"""
import json
from pathlib import Path
import re
import sys
import tomllib

from fontTools import subset
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

root = Path(__file__).resolve().parents[1]
config = tomllib.loads((root / "hugo.toml").read_text())["params"]["homeInfoParams"]
text = " ".join(config.values()) + (root / "layouts/_partials/home_info.html").read_text() + (root / "static/js/home-cover.js").read_text()
chinese = set(re.findall(r"[\u2e80-\u9fff\uff00-\uffef]", text))
latin = set(chr(code) for code in range(32, 127))
coverage = {}

for source, chars, name in [
    (sys.argv[1], latin, "caveat-hero.woff"),
    (sys.argv[2], chinese, "long-cang-hero.woff"),
]:
    font = TTFont(source)
    if "fvar" in font:
        font = instantiateVariableFont(font, {"wght": 600}, inplace=True)
    missing = {ord(char) for char in chars} - set(font.getBestCmap())
    if missing:
        raise ValueError(f"Missing source glyphs: {''.join(chr(code) for code in sorted(missing))}")
    options = subset.Options()
    options.layout_features = ["kern", "liga"]
    options.hinting = False
    subsetter = subset.Subsetter(options=options)
    subsetter.populate(text="".join(sorted(chars)))
    subsetter.subset(font)
    font.flavor = "woff"
    destination = root / "static/fonts" / name
    font.save(destination)
    coverage[name] = "".join(sorted(chars))
    print(f"{name}: {len(chars)} characters, {destination.stat().st_size} bytes")

(root / "static/fonts/hero-font-coverage.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2) + "\n")
