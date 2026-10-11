"""Final FigS3 renderer extracted from its frozen production layout."""
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
DATA=ROOT/'figures/Supplementary/FigS3/data'
SOURCE=DATA
COHORTS=['EAS','SAS','FIN','shuf2000']
NAMES=['EAS','SAS','FIN','random2000']
LABELS=['POLII\nsplice\nRNUs','U6/\nU6ATAC','Pseudo\nRNUs','Control']
WIDTH_MM,HEIGHT_MM=183,152

def p_text(value):
    if value < .001:
        mantissa, exponent = f"{value:.1e}".split("e")
        return rf"{mantissa}\times10^{{{int(exponent)}}}"
    return f"{value:.3f}".rstrip("0").rstrip(".")

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = read(DATA / "FigS3_locus_density.tsv.gz")
    bars = read(DATA / "FigS3_control_relative_means.tsv")
    tests = read(DATA / "FigS3_pairwise_statistics.tsv")
    profiles = read(DATA / "FigS3_profile_summary.tsv")
    assert len(profiles) == 1200 and len(bars) == 16
    style()
    mpl.rcParams.update({'xtick.labelsize':6, 'ytick.labelsize':6})
    fig, axes = plt.subplots(3, 4, figsize=(WIDTH_MM/25.4, HEIGHT_MM/25.4),
                             gridspec_kw={"height_ratios": [1.8, 1, 1.15]})
    fig.subplots_adjust(left=.105, right=.985, top=.925, bottom=.14, wspace=.21, hspace=.63)
    box_max = np.ceil(data.rate.max()*1.48/100)*100
    bar_max = np.ceil(bars.mean_density_ratio_to_control.max()*1.20/5)*5
    profile_max = np.ceil(profiles.density_ratio_to_control.max()*1.08/5)*5
    annotations = []
    for col, cohort in enumerate(COHORTS):
        selected = data[data.cohort.eq(cohort)]
        vectors = [selected[selected.analysis_group.eq(group)].rate.to_numpy() for group in GROUPS]
        ax = axes[0, col]
        boxes = ax.boxplot(vectors, positions=range(4), widths=.58, patch_artist=True, showfliers=False,
                           boxprops={"edgecolor": "#555555", "linewidth": .7},
                           medianprops={"color": "#222222", "linewidth": 1.1},
                           whiskerprops={"color": "#444444", "linewidth": .7},
                           capprops={"color": "#444444", "linewidth": .7})
        rng = np.random.default_rng(20260916+col)
        for i, (values, patch, color) in enumerate(zip(vectors, boxes["boxes"], COLORS)):
            # Match Fig. 1 fill opacity while preserving the original border.
            patch.set_facecolor(mpl.colors.to_rgba(color, FILL_ALPHA))
            patch.set_edgecolor(mpl.colors.to_rgba("#555555", .45))
            ax.scatter(i+rng.uniform(-.18, .18, len(values)), values, s=2.1, color=color,
                       alpha=.32 if len(values)<300 else .15, edgecolors="none", rasterized=False, zorder=2)
            ax.scatter(i, values.mean(), marker="D", s=13, color="#303030", edgecolors="white", linewidths=.4, zorder=5)
        for level, group in enumerate(GROUPS[1:]):
            test = tests[tests.cohort.eq(cohort) & tests.group_1.eq(GROUPS[0]) & tests.group_2.eq(group)].iloc[0]
            y, j = box_max*(.745+.095*level), level+1
            ax.plot([0,0,j,j], [y-box_max*.012,y,y,y-box_max*.012], linewidth=.6, color="#555555")
            label = rf"$\mathit{{P}} = {p_text(test.mann_whitney_p)}$"
            ax.text(j/2, y+box_max*.01, label, ha="center", va="bottom", fontsize=6)
            annotations.append(dict(cohort=cohort, group_1=GROUPS[0], group_2=group, p_value=test.mann_whitney_p, label=label))
        ax.set_xticks(range(4), [v+"\n("+str(len(selected[selected.analysis_group.eq(g)]))+")" for v,g in zip(LABELS,GROUPS)])
        ax.set_xlim(-.60,3.60)
        ax.set_ylim(0,box_max)
        ax.yaxis.set_major_locator(MultipleLocator(250))
        ax.set_title(NAMES[col], fontsize=8, pad=11)
        pos=ax.get_position()
        fig.text(pos.x0-.025, pos.y1+.025, "abcd"[col], fontsize=10, weight="bold")
        ax=axes[1,col]
        values=bars[bars.cohort.eq(cohort)].set_index("analysis_group").loc[GROUPS].mean_density_ratio_to_control.to_numpy()
        assert values[-1] == 1
        ax.bar(range(4), values, width=.58,
               color=[mpl.colors.to_rgba(color, FILL_ALPHA) for color in COLORS],
               edgecolor="none", zorder=2)
        ax.axhline(1,color=COLORS[-1],linestyle=(0,(3,2)),linewidth=.65,zorder=1)
        for i,value in enumerate(values):
            ax.text(i,value+bar_max*.025,f"{value:.2f}×" if i<3 else "1×",ha="center",va="bottom",fontsize=6)
        ax.set_xticks(range(4),LABELS)
        ax.set_xlim(-.60,3.60)
        ax.set_ylim(0,bar_max)
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax=axes[2,col]
        ax.axvspan(0,2000,color="#E8F2F8",zorder=0)
        for group,color in zip(GROUPS[:3],COLORS[:3]):
            frame=profiles[profiles.cohort.eq(cohort)&profiles.analysis_group.eq(group)].sort_values("bin_index")
            ax.plot((frame.display_start+frame.display_end)/2,frame.density_ratio_to_control,color=color,linewidth=.85)
        ax.axhline(1,color=COLORS[-1],linestyle=(0,(3,2)),linewidth=.65)
        for x in [0,2000]:
            ax.axvline(x,color="#888888",linestyle=(0,(3,2)),linewidth=.6)
        ax.set_xlim(-2000,4000)
        ax.set_xticks([-2000,0,2000,4000],["−2 kb","TSS","TES","+2 kb"])
        ax.set_ylim(0,profile_max)
        ax.yaxis.set_major_locator(MultipleLocator(5))
        for ax in axes[:,col]:
            ax.spines[["right","top"]].set_visible(False)
            ax.tick_params(pad=2)
            if col:
                ax.tick_params(labelleft=False)
    axes[0,0].set_ylabel("# rare SNVs per callable kb")
    axes[1,0].set_ylabel("Mean rare SNVs density\nrelative to control")
    axes[2,0].set_ylabel("Rare SNVs density\nrelative to control")
    handles=[Line2D([],[],color=c,lw=1.1,linestyle="--" if i==3 else "-") for i,c in enumerate(COLORS)]
    fig.legend(handles,["POLII spliceRNUs", "U6/U6ATAC", "PseudoRNUs", "Control"],
               loc="lower center",bbox_to_anchor=(.54,.035),ncol=4,frameon=False,fontsize=6.5,
               columnspacing=1.1,handlelength=1.4,handletextpad=.4)
    for ax in axes[:2, :].flat:
        assert len(ax.patches) == 4
        for patch, color in zip(ax.patches, COLORS):
            assert np.allclose(patch.get_facecolor(), mpl.colors.to_rgba(color, FILL_ALPHA))
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    clipped=[]
    for item in fig.findobj(mpl.text.Text):
        if item.get_visible() and item.get_text():
            bounds=item.get_window_extent(renderer)
            if bounds.x0 < -2 or bounds.y0 < -2 or bounds.x1 > fig.bbox.width+2 or bounds.y1 > fig.bbox.height+2:
                clipped.append(item.get_text())
    assert not clipped,clipped
    for ax in axes[:2,:].flat:
        bounds = [label.get_window_extent(renderer) for label in ax.get_xticklabels()]
        assert all(left.x1 < right.x0 for left,right in zip(bounds,bounds[1:])), "Overlapping category labels"
    save_figure(fig, 'FigS3')

if __name__=='__main__': main()
