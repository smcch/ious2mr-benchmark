# SynDiff arm

First-party code for the adversarial-diffusion family of the benchmark:

* `backbones3d.py` — 3D NCSN++ generator and time-conditioned 3D discriminator
* `eval_resvit_protocol.py`, `eval_all_epochs.py` — evaluation under the unified metric protocol
* `build_unified_final.py` — assembles the SynDiff arm of the results tables

## Upstream dependency (run this first)

The 2D/2.5D training loop builds on upstream SynDiff, which in turn derives from
NVIDIA's DDGAN. That code is **not redistributed here** because it is licensed for
**non-commercial research use only** and this repository is Apache-2.0. Fetch it at the
pinned commit with:

```bash
python scripts/setup_syndiff_upstream.py     # -> src/syndiff/_upstream/
```

The released SynDiff **weights** carry the same non-commercial restriction. See
[`../../THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md).
