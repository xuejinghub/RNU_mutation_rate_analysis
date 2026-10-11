"""Final FigS4 renderer extracted from its frozen production layout."""
from rnu_common import ROOT, read, apply_journal_style, save_figure
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from scipy.stats import chi2, norm
GROUPS=['pol2_spliceRNUs','functional_U6s','RNU_pseudogenes','genome_wide_neutral']
COLORS=['#2077b5','#ff800d','#2ba02b','#777777']
FILL_ALPHA=.70
style=apply_journal_style
OUT=ROOT/'results'
DATA=ROOT/'figures/Supplementary/FigS4/data'
SOURCE=DATA
LABELS=['POLII\nspliceRNUs','U6/U6ATAC','PseudoRNUs','Control']
BRACKETS=[(0,1),(1,2),(2,3),(0,2),(1,3),(0,3)]
WIDTH_MM,HEIGHT_MM=85,100

def p_text(value):
    if value < .001:
        mantissa, exponent = f"{value:.1e}".split("e")
        return rf"{mantissa}\times 10^{{{int(exponent)}}}"
    return f"{value:.3f}".rstrip("0").rstrip(".")

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths=[SOURCE/"locus_DNM_data.tsv",
           SOURCE/"aggregate_rates.tsv", SOURCE/"unadjusted_score_tests.tsv"]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    data=read(paths[0])
    summary=read(paths[1]).query("panel == 'b'").copy().set_index("analysis_group").loc[GROUPS]
    raw=read(paths[2]).copy()
    tests=raw[["group_1","group_2","count_1","count_2","exposure_1","exposure_2",
               "pooled_rate_ratio_1_over_2","poisson_score_z","poisson_score_p"]].copy()
    assert len(tests)==6
    assert data.proband_count.eq(9652).all()
    assert np.array_equal(data.exposure, data.callable_bp*9652*2)
    checks=[]
    for group in GROUPS:
        frame=data[data.analysis_group.eq(group)]
        n, exposure=frame.event_count.sum(),frame.exposure.sum()
        rate=1e8*n/exposure
        low=1e8*.5*chi2.ppf(.025,2*n)/exposure if n else 0
        high=1e8*.5*chi2.ppf(.975,2*(n+1))/exposure
        frozen=summary.loc[group]
        assert n==frozen.total_events and exposure==frozen.total_exposure
        assert np.allclose([rate,low,high],frozen[["pooled_rate","pooled_rate_ci_low","pooled_rate_ci_high"]].to_numpy(float),rtol=1e-13,atol=0)
        checks.append(dict(check=f"{group}: counts, exposure, rate, exact CI",status="PASS"))
    for test in tests.itertuples():
        pooled=(test.count_1+test.count_2)/(test.exposure_1+test.exposure_2)
        z=(test.count_1/test.exposure_1-test.count_2/test.exposure_2)/np.sqrt(pooled*(1/test.exposure_1+1/test.exposure_2))
        p=2*norm.sf(abs(z))
        ratio=(test.count_1/test.exposure_1)/(test.count_2/test.exposure_2)
        assert np.isclose(p,test.poisson_score_p,rtol=1e-12,atol=0)
        assert np.isclose(ratio,test.pooled_rate_ratio_1_over_2,rtol=1e-13)
    style()
    fig,ax=plt.subplots(figsize=(WIDTH_MM/25.4,HEIGHT_MM/25.4))
    fig.subplots_adjust(left=.18,right=.985,bottom=.18,top=.88)
    for i,group in enumerate(GROUPS):
        record=summary.loc[group]
        y=float(record.pooled_rate)
        errors=np.array([[y-record.pooled_rate_ci_low],[record.pooled_rate_ci_high-y]])
        # Preserve the point-and-CI design; match the Fig. 1 category fill colours.
        ax.errorbar(i,y,yerr=errors,fmt="o",markersize=4.8,
                    markerfacecolor=mpl.colors.to_rgba(COLORS[i], FILL_ALPHA),
                    markeredgecolor="#333333",markeredgewidth=.7,ecolor=COLORS[i],elinewidth=1.1,
                    capsize=3,capthick=.8,zorder=5)
        ax.text(i,record.pooled_rate_ci_high+1,f"{int(record.total_events)} events",ha="center",va="bottom",fontsize=6.5)
    lookup=tests.set_index(["group_1","group_2"])
    annotations=[]
    for level,(i,j) in enumerate(BRACKETS):
        row=lookup.loc[(GROUPS[i],GROUPS[j])]
        y=37+level*3.9
        ax.plot([i,i,j,j],[y-.5,y,y,y-.5],color="#444444",linewidth=.65)
        text=rf"RR = {row.pooled_rate_ratio_1_over_2:.2f}×; $\mathit{{P}} = {p_text(row.poisson_score_p)}$"
        ax.text((i+j)/2,y+.28,text,ha="center",va="bottom",fontsize=6.2,fontweight="normal",color="#333333")
        annotations.append(dict(group_1=GROUPS[i],group_2=GROUPS[j],rate_ratio=row.pooled_rate_ratio_1_over_2,p_value=row.poisson_score_p,label=text))
    ax.set_xticks(range(4),[label+f"\nn = {int(summary.loc[g,'n_units']):,}" for label,g in zip(LABELS,GROUPS)])
    ax.set_xlim(-.55,3.55)
    ax.set_ylim(0,60)
    ax.set_yticks(range(0,61,10))
    ax.set_ylabel(r"DNM per $10^{8}$ callable site-generations")
    ax.spines[["right","top"]].set_visible(False)
    ax.tick_params(pad=2)
    ax.set_title("De novo mutation rate\nPalsson 2025 · 9,652 probands",loc="left",fontsize=7.5,fontweight="normal",pad=8)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for text in fig.findobj(mpl.text.Text):
        if text.get_visible() and text.get_text():
            assert text.get_fontweight() in ["normal",400]
            box=text.get_window_extent(renderer)
            assert box.x0>=-2 and box.y0>=-2 and box.x1<=fig.bbox.width+2 and box.y1<=fig.bbox.height+2,text.get_text()
    save_figure(fig, 'FigS4')

if __name__=='__main__': main()
