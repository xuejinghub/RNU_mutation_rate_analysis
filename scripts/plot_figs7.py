"""Final FigS7 renderer extracted from its frozen production layout."""
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
DATA=ROOT/'figures/Supplementary/FigS7/data'
SOURCE=DATA
LABELS=['POLII\nspliceRNUs','U6/\nU6ATAC','PseudoRNUs','Control']
WIDTH_MM,HEIGHT_MM=180,255
DATASETS=['shuf500_r1','rhesus_original','shuf200_r1','mouse_original','shuf100_r1','rat_current']
NAMES=dict(zip(DATASETS,['Human random500','Macaque','Human random200','Mouse','Human random100','Rat']))
MERGE_BOX_BAR_PANELS=READABLE_SHARED_X=SAFE_BRACKET_LAYOUT=True
JITTER_SEEDS=dict(zip(['shuf200_r1','mouse_original','shuf500_r1','rhesus_original','shuf100_r1','rat_current'],range(20260916,20260922)))
def p_text(value):
    if value<.001:
        a,b=f'{value:.1e}'.split('e')
        return rf'{a}\times 10^{{{int(b)}}}'
    return f'{value:.3f}'.rstrip('0').rstrip('.')

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = read(DATA / "FigS7_four_group_density.tsv.gz")
    means = read(DATA / "FigS7_control_relative_means.tsv")
    tests = read(DATA / "FigS7_pairwise_unadjusted_P.tsv")
    assert set(data.dataset) == set(DATASETS)
    assert len(means) == len(DATASETS)*4 and len(tests) == len(DATASETS)*6
    n_pairs = len(DATASETS)//2
    assert n_pairs*2 == len(DATASETS)
    style()
    if READABLE_SHARED_X:
        mpl.rcParams.update({'font.size':8,'axes.labelsize':8,'xtick.labelsize':7.5,
                             'ytick.labelsize':7.5,'axes.titlesize':9})
    if MERGE_BOX_BAR_PANELS:
        fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
        outer = fig.add_gridspec(n_pairs, 1, left=.125 if READABLE_SHARED_X else .115,
                                right=.985, top=.950 if READABLE_SHARED_X else .955,
                                bottom=.075 if READABLE_SHARED_X else .055,
                                hspace=.32 if SAFE_BRACKET_LAYOUT else (.40 if READABLE_SHARED_X else .42))
        axes = np.empty((n_pairs*2, 2), dtype=object)
        for pair in range(n_pairs):
            inner = outer[pair].subgridspec(2, 2, height_ratios=[2.1, 1] if SAFE_BRACKET_LAYOUT else ([1.8, 1] if READABLE_SHARED_X else [1.65, 1]),
                                          hspace=.15 if READABLE_SHARED_X else .45, wspace=.28)
            for row in range(2):
                for col in range(2):
                    axes[pair*2+row, col] = fig.add_subplot(inner[row, col])
    else:
        fig, axes = plt.subplots(n_pairs*2, 2, figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4),
                                 gridspec_kw={"height_ratios": [1.65, 1]*n_pairs})
        fig.subplots_adjust(left=.115, right=.985, top=.955, bottom=.055, hspace=.60, wspace=.28)
    annotations = []
    bracket_artists = {}
    pair_maxima = []
    bar_max = np.ceil(means.mean_density_relative_to_control.max() * 1.18 / 2) * 2
    for pair_index in range(n_pairs):
        datasets = DATASETS[pair_index * 2:pair_index * 2 + 2]
        maximum = data[data.dataset.isin(datasets)].rare_snv_density_per_kb.max()
        pair_maxima.append(float(maximum))
        box_max = np.ceil(maximum * (2.05 if READABLE_SHARED_X else 1.68) / 50) * 50
        for column, dataset in enumerate(datasets):
            box_ax, bar_ax = axes[pair_index * 2, column], axes[pair_index * 2 + 1, column]
            frame = data[data.dataset.eq(dataset)]
            vectors = [frame[frame.analysis_group.eq(group)].rare_snv_density_per_kb.to_numpy() for group in GROUPS]
            boxes = box_ax.boxplot(vectors, positions=range(4), widths=.53, patch_artist=True, showfliers=False,
                                   boxprops={"edgecolor": "#555555", "linewidth": .7},
                                   medianprops={"color": "#222222", "linewidth": 1.1},
                                   whiskerprops={"color": "#444444", "linewidth": .7},
                                   capprops={"color": "#444444", "linewidth": .7})
            rng = np.random.default_rng(JITTER_SEEDS.get(dataset, 20260916 + pair_index * 2 + column))
            for i, (vector, patch, color) in enumerate(zip(vectors, boxes["boxes"], COLORS)):
                # Match Fig. 1 fill opacity without changing the existing border.
                patch.set_facecolor(mpl.colors.to_rgba(color, FILL_ALPHA))
                patch.set_edgecolor(mpl.colors.to_rgba("#555555", .45))
                box_ax.scatter(i + rng.uniform(-.18, .18, len(vector)), vector, s=3, color=color,
                               alpha=.32 if len(vector) < 300 else .15, edgecolors="none", rasterized=False, zorder=2)
                box_ax.scatter(i, vector.mean(), marker="D", s=17, color="#303030",
                               edgecolors="white", linewidths=.45, zorder=5)
            # Keep the original three RNU contrasts and add POLII versus Control.
            # Fold estimates are now presented in the dedicated bar-chart rows.
            for level, (i, j) in enumerate([(0, 1), (1, 2), (0, 2), (0, 3)]):
                result = tests[tests.dataset.eq(dataset) & tests.group_1.eq(GROUPS[i]) & tests.group_2.eq(GROUPS[j])].iloc[0]
                y = box_max * ((.56 + level * .125) if READABLE_SHARED_X else (.665 + level * .097))
                line, = box_ax.plot([i, i, j, j], [y-box_max*.014, y, y, y-box_max*.014], color="#555555", linewidth=.65)
                label = rf"$\mathit{{P}} = {p_text(result.mann_whitney_p)}$"
                text = box_ax.text((i+j)/2, y+box_max*.008, label, ha="center", va="bottom", fontsize=7.5 if READABLE_SHARED_X else 6)
                bracket_artists.setdefault(box_ax, []).append((line,text,i,j))
                annotations.append(dict(dataset=dataset, group_1=GROUPS[i], group_2=GROUPS[j],
                                        p_value=result.mann_whitney_p, label=label))
            tick_labels = [label + f"\nn = {len(v):,}" for label, v in zip(LABELS, vectors)] if dataset not in ['shuf500_r1', 'shuf100_r1'] else LABELS
            box_ax.set_xticks(range(4), tick_labels)
            if READABLE_SHARED_X:
                box_ax.tick_params(axis='x',bottom=False,labelbottom=False)
            box_ax.set_ylim(0, box_max)
            box_ax.yaxis.set_major_locator(MultipleLocator(100))
            summary = means[means.dataset.eq(dataset)].set_index("analysis_group").loc[GROUPS]
            ratios = summary.mean_density_relative_to_control.to_numpy()
            assert ratios[-1] == 1
            bar_ax.bar(range(4), ratios, width=.57,
                       color=[mpl.colors.to_rgba(color, FILL_ALPHA) for color in COLORS],
                       edgecolor="none", zorder=2)
            bar_ax.axhline(1, color="#777777", linewidth=.65, linestyle=(0, (3, 2)), zorder=1)
            for i, ratio in enumerate(ratios):
                bar_ax.text(i, ratio+bar_max*.026, f"{ratio:.2f}×" if i < 3 else "1×", fontsize=8 if READABLE_SHARED_X else 6.5, ha="center", va="bottom")
            bar_ax.set_xticks(range(4), LABELS)
            if READABLE_SHARED_X:
                shared_labels = [label + ('\n' if '\n' not in label else '') + f'\nn = {len(v):,}'
                                 for label,v in zip(LABELS,vectors)]
                bar_ax.set_xticklabels(shared_labels)
            bar_ax.set_ylim(0, bar_max)
            bar_ax.yaxis.set_major_locator(MultipleLocator(5 if bar_max > 12 else 2))
            for ax in [box_ax, bar_ax]:
                ax.set_xlim(-.55, 3.55)
                ax.spines[["right", "top"]].set_visible(False)
                ax.tick_params(pad=2)
                if not MERGE_BOX_BAR_PANELS or ax is box_ax:
                    ax.set_title(NAMES[dataset], loc="left", fontsize=9 if READABLE_SHARED_X else 7,
                                 pad=10 if READABLE_SHARED_X else 5)
                if column == 1:
                    ax.tick_params(labelleft=False)
            if column == 0:
                box_ax.set_ylabel("# rare SNVs per callable kb")
                bar_ax.set_ylabel("Mean rare SNVs density\nrelative to control")
                if READABLE_SHARED_X:
                    box_ax.set_ylabel('# rare SNVs per\ncallable kb',labelpad=4)
                    bar_ax.set_ylabel('Mean rare SNVs\ndensity relative\nto control',labelpad=4)
    for row in range(n_pairs*2):
        if MERGE_BOX_BAR_PANELS and row % 2:
            continue
        position = axes[row, 0].get_position()
        label_index = row//2 if MERGE_BOX_BAR_PANELS else row
        fig.text(.020 if READABLE_SHARED_X else .025, position.y1+(.018 if READABLE_SHARED_X else .010),
                 "abcdefghijklmnopqrstuvwxyz"[label_index], fontsize=11 if READABLE_SHARED_X else 10, weight="bold")
    for ax in axes.flat:
        assert len(ax.patches) == 4
        for patch, color in zip(ax.patches, COLORS):
            assert np.allclose(patch.get_facecolor(), mpl.colors.to_rgba(color, FILL_ALPHA))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if SAFE_BRACKET_LAYOUT:
        # Place brackets in physical display units so superscripts cannot touch
        # the next bracket's horizontal line or downward hook.
        px_per_pt = fig.dpi/72
        text_gap, hook, next_clearance = 2*px_per_pt, 1*px_per_pt, 2.5*px_per_pt
        top_gap, data_gap = 2*px_per_pt, 4*px_per_pt
        for pair in range(n_pairs):
            pair_axes = axes[pair*2]
            heights = [max(bracket_artists[ax][level][1].get_window_extent(renderer).height
                           for ax in pair_axes) for level in range(4)]
            offsets = [0.0]
            for level in range(3):
                offsets.append(offsets[-1]+text_gap+heights[level]+next_clearance+hook)
            area = pair_axes[0].get_window_extent(renderer)
            first_y = area.y1-top_gap-heights[-1]-text_gap-offsets[-1]
            available_fraction = (first_y-hook-data_gap-area.y0)/area.height
            assert available_fraction>.20, 'Insufficient height for data and legible brackets'
            limit = np.ceil(pair_maxima[pair]/available_fraction/50)*50
            for ax in pair_axes:
                ax.set_ylim(0,limit)
                inverse = ax.transData.inverted()
                for offset,(line,text,i,j) in zip(offsets,bracket_artists[ax]):
                    y = first_y+offset
                    line.set_ydata([inverse.transform((0,v))[1] for v in [y-hook,y,y,y-hook]])
                    text.set_position(((i+j)/2,inverse.transform((0,y+text_gap))[1]))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        from matplotlib.transforms import Bbox
        text_line_collisions = []
        minimum_clearance_pt = []
        for artists in bracket_artists.values():
            for own_line,text,_,_ in artists:
                bounds = text.get_window_extent(renderer)
                padded = Bbox.from_extents(bounds.x0-1.5*px_per_pt,bounds.y0-1.5*px_per_pt,
                                          bounds.x1+1.5*px_per_pt,bounds.y1+1.5*px_per_pt)
                own_points = own_line.get_transform().transform(own_line.get_path().vertices)
                minimum_clearance_pt.append((bounds.y0-own_points[:,1].max())/px_per_pt)
                for other_artists in bracket_artists.values():
                    for line,_,_,_ in other_artists:
                        path = line.get_transform().transform_path(line.get_path())
                        if path.intersects_bbox(padded,filled=False):
                            text_line_collisions.append(text.get_text())
        (OUT/'text_bracket_overlap_check.json').write_text(json.dumps(dict(
            text_line_collisions=text_line_collisions,text_clearance_padding_pt=1.5,
            minimum_label_to_own_bracket_gap_pt=min(minimum_clearance_pt),
            bracket_placement='rendered text height plus 2.5-pt next-bracket clearance'),indent=2)+'\n')
        assert not text_line_collisions,text_line_collisions
    clipped = []
    for item in fig.findobj(mpl.text.Text):
        if item.get_visible() and item.get_text():
            extent = item.get_window_extent(renderer)
            if extent.x0 < -2 or extent.y0 < -2 or extent.x1 > fig.bbox.width+2 or extent.y1 > fig.bbox.height+2:
                clipped.append(item.get_text())
    assert not clipped, clipped
    if READABLE_SHARED_X:
        # Check rendered text boxes, including neighbouring panel annotations.
        # Locators create ticks outside the view limits that Axis.draw skips.
        off_view_ticks = set()
        for ax in axes.flat:
            for axis,limits in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
                for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                    if not min(limits)<=tick.get_loc()<=max(limits):
                        off_view_ticks.update([tick.label1,tick.label2])
        visible = [t for t in fig.findobj(mpl.text.Text) if t.get_visible() and t.get_text() and t not in off_view_ticks]
        collisions = []
        for i,first in enumerate(visible):
            a = first.get_window_extent(renderer)
            for second in visible[i+1:]:
                b = second.get_window_extent(renderer)
                if min(a.x1,b.x1)-max(a.x0,b.x0)>1 and min(a.y1,b.y1)-max(a.y0,b.y0)>1:
                    collisions.append([first.get_text(),second.get_text()])
        (OUT/'text_overlap_check.json').write_text(json.dumps({'text_collisions':collisions},indent=2)+'\n')
        assert not collisions,collisions
    save_figure(fig, 'FigS7')

if __name__=='__main__': main()
