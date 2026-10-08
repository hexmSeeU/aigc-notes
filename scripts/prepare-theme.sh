#!/usr/bin/env bash
set -euo pipefail
commit=d3768854d00ad003b0a8dbdba254ce9224377a01
if [ ! -d themes/PaperMod/.git ]; then
  git clone https://github.com/adityatelange/hugo-PaperMod.git themes/PaperMod
fi
git -C themes/PaperMod checkout --detach "$commit"
