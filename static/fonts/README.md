# Self-hosted hero handwriting

Only the existing homepage hero uses these fonts. Article text, navigation,
layout, illustration, and animation are unchanged. There are no requests to
third-party font services.

- Caveat, copyright 2014 The Caveat Project Authors, SIL OFL 1.1
  - Source: https://github.com/google/fonts/tree/main/ofl/caveat
  - Original `Caveat[wght].ttf` SHA-256: `0bdb6b660482d31531b3945849fba5916b3ef8695da7024a9e6b9ee3c4157988`
  - Instantiated at weight 600 and subset to printable ASCII
- Long Cang, copyright 2018 The Long Cang Project Authors, SIL OFL 1.1
  - Source: https://github.com/google/fonts/tree/main/ofl/longcang
  - Original `LongCang-Regular.ttf` SHA-256: `e5bf2c3f24ef2327c6f136d8f73e2f9dfdf44896fdbeb35a9515f44777bb91bc`
  - Subset to Chinese characters and fullwidth punctuation in the hero copy and controls

Both official source files were downloaded on 2026-10-12. Licenses are included
unchanged apart from trailing whitespace normalization. Reproduce the subsets
with FontTools:

    python scripts/subset-hero-fonts.py /path/Caveat[wght].ttf /path/LongCang-Regular.ttf

The script verifies source glyph coverage and saves the covered characters in
`hero-font-coverage.json`. Run it when editing hero lettering.
