# Nonpublished legacy source audit

Base: `7a965a8d1957236e9d4f9b08697bb8fd83cc450b` (current main fetched for this change).

These two Markdown files are byte-for-byte copies of the former `content/posts/*-derivation/index.md` files. They are outside Hugo's content/static/assets directories and are never built. They remain available for preservation review; the public URLs are aliases of the corresponding collections.

| Source | SHA-256 | Published coverage |
| --- | --- | --- |
| `ddim-derivation.md` | `e08d7977eb56774343b6f7cf3bac71df68fa69c751df78a0823b8285a74c2f5e` | All 327 nonblank, non-heading substantive lines occur verbatim across `ddim-1` through `ddim-5`. Only the legacy-page navigation notice is excluded. |
| `ddpm-derivation.md` | `d033109b9d6550dca11323c635f89667c5aa35c42734307c5a060a8c71536641` | All 226 original audited blocks remain in `ddpm-1` through `ddpm-4`, using the existing explicit cross-reference/section-number corrections in `tests/fixtures/ddpm-preservation.json`. |

The DDIM coverage test checks from the old source toward the published episodes (not just the reverse). DDPM's existing preservation test checks every audited block. Historical mathematical/body snapshots still run against the unchanged archival sources, and against live split articles with only the explicitly reviewed link substitutions reversed in the test helper. No mathematical snapshots were regenerated.

An audit of current content and all fetched Git history found no ELBO monolithic legacy page: only `elbo-1`, `elbo-2`, `elbo-3`, their collection, and the unrelated introductory `ae-to-vae` note. No speculative ELBO URL is created and the standalone VAE note/diagrams remain unchanged.

Browser tests cover all split pages' formulas, final summary/self-check ordering, and exactly one footer previous/next navigation. Redirect tests cover JavaScript enabled/disabled and root/subpath hosting. Built-output checks exclude legacy paths from all other HTML, XML and JSON outputs (including RSS, sitemap and any search indexes).
