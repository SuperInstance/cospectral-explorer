# cospectral-explorer

**Can conservation ratios distinguish cospectral graphs? — breaking the "can you hear the shape of a drum?" problem for graphs.**

Cospectral graphs share the same Laplacian eigenvalue spectrum but have different edge structure. This experiment tests whether conservation ratios (computed from node attributes and the tension graph Laplacian) can break the ambiguity that eigenvalues alone cannot resolve.

## What This Gives You

- **Cospectral pair finder** — generate random graphs, group by Laplacian spectrum
- **Edge overlap metric** — Jaccard similarity of edge sets
- **Conservation ratio discrimination** — can CR distinguish cospectral graphs?
- **Attribute sensitivity** — test with random, degree-based, and structural attributes
- **Visualization** — side-by-side graph layouts with spectral and conservation comparison

## Quick Start

```bash
pip install numpy networkx matplotlib
python cospectral_explorer.py
```

## How It Fits

Part of the SuperInstance ecosystem:

- **[fiedler-universal](https://github.com/SuperInstance/fiedler-universal)** — Fiedler vector benchmarking
- **cospectral-explorer** — Cospectral graph discrimination (this repo)

## License

MIT
