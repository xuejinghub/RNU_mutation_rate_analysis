# Introduction

This repository contains the main datasets and scripts used in the analyses described in  **"Cross-species analysis identifies APOBEC3-associated spliceosomal snRNA hypermutability"**.


## Repository Structure

Numbering follows RNU manuscript.

| Figure | Content | Final plotting script |
| --- | --- | --- |
| Fig. 1 | Cross-species rare variation | `scripts/plot_fig1.py` |
| Fig. 2 | Mutation features and APOBEC3 repertoire | `scripts/plot_fig2.py` |
| Fig. 3 | A3B-associated ssDNA and R-loops | `scripts/plot_fig3.py` |
| Supplementary Figs. 1, 2, 5, 6 | Human/macaque; chimpanzee/baboon; mouse/rat; chicken annotation | `scripts/plot_annotations.py` |
| Supplementary Fig. 3 | Human-cohort densities, enrichments and profiles | `scripts/plot_figs3.py` |
| Supplementary Fig. 4 | Germline DNMs | `scripts/plot_figs4.py` |
| Supplementary Fig. 7 | Species-matched downsampling | `scripts/plot_figs7.py` |
| Supplementary Figs. 8-11 | EAS, SAS, FIN and random2000 mutation features | `scripts/plot_cohorts.py` |

```text
scripts/                         Final plotting entry points
processing/                      Final-related data processing and dependencies
figures/Fig1..Fig3/data/          Main-figure inputs
figures/Supplementary/FigS1..11/  Manuscript-numbered supplementary inputs
figures/reference/               Original final-layout PDFs, renumbered only
figures/final/                   Regenerated PDF/SVG/300-dpi PNG outputs
gene_sets/                       Final coordinate sets
```

## Run

Tested plotting environment: Python 3.10.8, NumPy 1.24.4, pandas 1.5.3, SciPy 1.14.1 and Matplotlib 3.9.1. Matplotlib also uses Pillow for PNG export. `environment.yml` and `requirements.txt` describe the plotting environment.

```bash
# Compile the Python sources.
python -m compileall -q scripts
python -m compileall -q processing

# Install/use licensed Arial; no font files are distributed here.
export RNU_FONT_DIR=/path/to/licensed/arial/fonts
python scripts/reproduce.py --figure all
```

Use `--figure Fig1`, `Fig2`, `Fig3` or `Supplementary` to render a subset. The plotting command reads bundled processed tables and writes PDF, SVG and 300-dpi PNG files under `figures/final/`.
The plotting command directly renders saved results. 

`processing/` contains the final-related processing modules. Actual processing requires the corresponding VCFs, reference assemblies, genomic tracks and intermediate tables, with paths configured for the local installation. Depending on the stage, additional Python packages include `pysam`, `cyvcf2`, `pyBigWig` and MuSiCal (`musical`); command-line tools include bedtools, bcftools and samtools. These production dependencies are separate from the plotting environment.

## License

This repository is available under the [MIT License](LICENSE.md). Third-party software and datasets retain their respective licenses and terms of use.

## Contact

If you encounter any issues with this repository or have questions beyond the scope of the provided documentation, please feel free to contact us via **[e-mail](mailto:jingxue1992@gmail.com)** or open an **[issue on GitHub]()**.
