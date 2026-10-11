"""Render manuscript Supplementary Figures 8-11 from frozen cohort tables.

Uses the shared portable mutation-feature renderer; no refitting or resampling.
The original final-layout PDFs are retained separately under figures/reference.
"""
from plot_fig2 import main

if __name__ == '__main__':
    for cohort in ['EAS', 'SAS', 'FIN', 'random2000']:
        main(cohort)
