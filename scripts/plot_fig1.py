"""Mechanically extracted plotting functions for the saved Fig. 1 results.
Only repository-local inputs are read; statistical values are not recalculated here.
"""
from rnu_common import ROOT, apply_journal_style
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from rnu_common import ROOT, apply_journal_style

from types import SimpleNamespace
from unittest.mock import patch
from matplotlib.transforms import Bbox
from matplotlib.ticker import FixedLocator, FixedFormatter, MaxNLocator
from matplotlib import _mathtext
from matplotlib.backends.backend_pdf import RendererPdf, pdfRepr
def clean_axis(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(pad=2)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4,min_n_ticks=3))
base=SimpleNamespace(mpl=mpl,plt=plt,clean_axis=clean_axis)
DATA=ROOT/'figures/Fig1/data'
OUT=ROOT/'figures/final'
STEM='Fig1'
SPECIES=('Human','Chimpanzee','Rhesus_macaque','Baboon','Mouse','Rat','Chicken')
GROUPS=('pol2_spliceRNUs','functional_U6s','RNU_pseudogenes','genome_wide_neutral')
COLORS=dict(zip(GROUPS,['#2077b5','#ff800d','#2ba02b','#777777']))
HEADERS=dict(zip(SPECIES,['Human','Chimpanzee','Macaque','Baboon','Mouse','Rat','Chicken']))
COHORTS=dict(zip(SPECIES,[(1736,.001),(198,.01),(514,.002),(110,.02),(148,.01),(89,.02),(191,.01)]))
WIDTH_MM,HEIGHT_MM=180,160
FULL_GROUP_NAMES=dict(zip(GROUPS,['PolII spliceRNUs','U6/U6ATAC','PseudoRNUs','Control']))
P_LABEL_TEXT={}
class ArialMathConstants(_mathtext.FontConstantsBase):
    # The installed Arial PCLT x-height gives an excessive default rise.
    # This yields a 2.55 pt rise with 6.4 pt text and 4.48 pt superscripts.
    sup1 = .40

def p_label(value: float) -> str:
    """One editable Arial math expression; the complete exponent is superscript."""
    if value < .001:
        mantissa, exponent = f"{value:.1e}".split("e")
        label=rf"$\mathit{{P}} = {mantissa}\times10^{{{int(exponent)}}}$"
        superscript=str(int(exponent)).translate(str.maketrans("-0123456789","⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
        P_LABEL_TEXT[label]=f"P = {mantissa} × 10{superscript}"
    else:
        shown=f"{value:.3f}".rstrip("0").rstrip(".")
        label=rf"$\mathit{{P}} = {shown}$"
        P_LABEL_TEXT[label]=f"P = {shown}"
    return label

def place_significance_labels(fig,axes):
    """Let the math layout engine size and raise superscripts consistently."""
    for axis in axes:
        for x,y,value in axis.significance_specs:
            label=axis.text(x,y,"small n" if value is None else p_label(value),
                            fontfamily="Arial",fontsize=6.4,ha="center",va="bottom")
            axis.significance_labels.append([label])

def save_figure(fig):
    OUT.mkdir(parents=True,exist_ok=True)
    paths=[]
    original_draw_mathtext=RendererPdf.draw_mathtext

    def draw_accessible_mathtext(renderer,gc,x,y,s,prop,angle):
        actual=P_LABEL_TEXT.get(s)
        if actual is not None:
            renderer.file.write(b"/Span << /ActualText "+pdfRepr(actual)+b" >> BDC\n")
        original_draw_mathtext(renderer,gc,x,y,s,prop,angle)
        if actual is not None:
            renderer.file.write(b"EMC\n")

    for ext in ("pdf","svg","png"):
        path=OUT/f"{STEM}.{ext}"
        with patch.object(RendererPdf,"draw_mathtext",draw_accessible_mathtext):
            fig.savefig(path,dpi=300,facecolor="white",transparent=False)
        paths.append(path)
    return paths

def density_panel(ax,sp,data,stats,seed):
    sub=data[data.species.eq(sp)]
    rng=np.random.default_rng(seed)
    active=[g for g in GROUPS if (sub.analysis_group==g).any()]
    xs=[GROUPS.index(g) for g in active]
    vectors=[sub[sub.analysis_group.eq(g)].rare_snv_density_per_kb.dropna().to_numpy(float) for g in active]
    boxes=ax.boxplot(vectors,positions=xs,widths=.58,patch_artist=True,showfliers=False,
                     boxprops={"edgecolor":"#555555","linewidth":.7},
                     medianprops={"color":"#222222","linewidth":1.1},
                     whiskerprops={"color":"#444444","linewidth":.7},
                     capprops={"color":"#444444","linewidth":.7})
    for x,g,v,p in zip(xs,active,vectors,boxes["boxes"]):
        p.set_facecolor(base.mpl.colors.to_rgba(COLORS[g],.70))
        ax.scatter(x+rng.uniform(-.18,.18,len(v)),v,s=2.3,c=COLORS[g],
                   alpha=.32 if len(v)<300 else .15,edgecolors="none",zorder=2)
        ax.scatter(x,v.mean(),marker="D",s=17,c="#303030",edgecolors="white",linewidths=.45,zorder=5)
    maximum=max(1,max(float(v.max()) for v in vectors if len(v)))
    ax.set_xlim(-.32,3.32);ax.set_ylim(0,maximum*1.58)
    ax.significance_specs=[];ax.significance_labels=[]
    if GROUPS[0] in active:
        for level,g in enumerate(GROUPS[1:]):
            row=stats[(stats.species==sp)&(stats.group_1==GROUPS[0])&(stats.group_2==g)]
            if row.empty: continue
            r=row.iloc[0]
            y=maximum*(1.06+.15*level)
            x=GROUPS.index(g)
            ax.plot([0,0,x,x],[y-maximum*.018,y,y,y-maximum*.018],color="#555555",lw=.65,clip_on=False)
            value=None if min(int(r.n1),int(r.n2))<3 else float(r.p_value)
            ax.significance_specs.append((max(x/2,1.18),y+maximum*.018,value))
    ax.set_xticks([])
    base.clean_axis(ax)

def add_unit_tick(ax):
    """Label the true ratio 1 separately; lift its text clear of the zero label."""
    ax.yaxis.set_minor_locator(FixedLocator([1.0]))
    ax.yaxis.set_minor_formatter(FixedFormatter(["1"]))
    ax.tick_params(axis="y",which="minor",labelsize=6,pad=2,length=2.6,width=.6)
    for label in ax.get_yticklabels(minor=True):
        label.set_verticalalignment("bottom")

def bar_panel(ax,sp,bars,ymax):
    sub=bars[bars.species==sp].set_index("analysis_group")
    counts={g:int(sub.loc[g,"n_units"]) for g in GROUPS if g in sub.index}
    for g in GROUPS:
        if g not in sub.index: continue
        x=GROUPS.index(g);v=float(sub.loc[g,"mean_density_ratio_to_control"])
        ax.bar(x,v,width=.57,color=base.mpl.colors.to_rgba(COLORS[g],.7),edgecolor="white",lw=.4)
        ax.text(x+(.18 if x==0 else 0),v+ymax*.018,f"{v:.2f}",
                ha="center",va="bottom",fontsize=6)
    ax.axhline(1,color="#555555",linestyle=(0,(3,2)),lw=.65)
    ax.set_xticks(range(4),[f"{FULL_GROUP_NAMES[g]}\n({counts[g]})" for g in GROUPS])
    ax.set_xlim(-.32,3.32);ax.set_ylim(0,ymax)
    base.clean_axis(ax);ax.tick_params(axis="x",labelsize=6,pad=1)
    ax.spines["left"].set_position(("outward",1))
    add_unit_tick(ax)
    for label in ax.get_xticklabels():
        label.set_rotation(60)
        label.set_rotation_mode("anchor")
        label.set_horizontalalignment("right")
        label.set_multialignment("center")

def profile_panel(ax,sp,profile,upper):
    ax.axvspan(0,2000,color="#E8F2F8",zorder=0)
    for g in GROUPS[:3]:
        sub=profile[(profile.species==sp)&(profile.analysis_group==g)].sort_values("bin_index")
        if sub.empty: continue
        x=(sub.relative_start+sub.relative_end)/2
        x=np.where(x<0,x,np.where(x<=1000,x*2,x+1000))
        ax.plot(x,sub.density_ratio_to_control,color=COLORS[g],lw=.95,
                label={GROUPS[0]:"POLII spliceRNUs",GROUPS[1]:"U6/U6ATAC",GROUPS[2]:"PseudoRNUs"}[g])
    ax.set_xlim(-2000,4000);ax.set_ylim(0,upper)
    ax.set_xticks([-2000,0,2000,4000],["−2","TSS","TES","+2"])
    base.clean_axis(ax);ax.tick_params(axis="x",labelsize=6)
    add_unit_tick(ax)

def tree(fig,axes):
    ax=fig.add_axes([0,0,1,1],zorder=0)
    ax.set_axis_off();ax.patch.set_visible(False);ax.set_xlim(0,1);ax.set_ylim(0,1)
    xs=[a.get_position().x0+a.get_position().width/2 for a in axes]
    h,ch,m,b,mouse,rat,bird=xs
    apes=(h+ch)/2;ow=(m+b)/2;prim=(apes+ow)/2;rod=(mouse+rat)/2
    mammals=(prim+rod)/2;root=(mammals+bird)/2
    lines=[([root,root],[.984,.968]),([mammals,bird],[.968,.968]),
           ([mammals,mammals],[.968,.951]),([prim,rod],[.951,.951]),
           ([prim,prim],[.951,.937]),([apes,ow],[.937,.937]),
           ([apes,apes],[.937,.923]),([h,ch],[.923,.923]),
           ([ow,ow],[.937,.923]),([m,b],[.923,.923]),
           ([rod,rod],[.951,.923]),([mouse,rat],[.923,.923])]
    lines.extend(([x,x],[.923,.909]) for x in xs[:6]);lines.append(([bird,bird],[.968,.909]))
    compact_y=lambda y: .9475+(y-.909)*(.040/.075)
    for x,y in lines: ax.plot(x,[compact_y(v) for v in y],color="#50565e",lw=.9,clip_on=False)
    # Species silhouettes omitted: separate asset-license clearance required.


def main():
    apply_journal_style()
    names=("panel_a_gene_locus_density","panel_a_pairwise_statistics",
           "panel_b_control_mean_ratios","panel_c_profile_control_ratios")
    data,stats,bars,profile=[pd.read_csv(DATA/f"{name}.tsv",sep="\t",float_precision="round_trip") for name in names]
    assert set(data.species)==set(SPECIES)
    fig,axes=base.plt.subplots(3,7,figsize=(WIDTH_MM/25.4,HEIGHT_MM/25.4))
    fig.subplots_adjust(left=.063,right=.98,top=.84,bottom=.0796875,wspace=.29,hspace=.01)
    # Preserve the preceding panel heights (44.54, 26.01, 30.60 mm).
    # The 10 mm reduction comes from the header, row gaps and bottom whitespace.
    for row,(bottom,height) in enumerate(((.561625,.278375),(.3834375,.1625625),(.0796875,.19125))):
        for axis in axes[row]:
            position=axis.get_position()
            axis.set_position([position.x0,bottom,position.width,height])
    ymax=int(np.ceil(bars.mean_density_ratio_to_control.max()*1.22/5))*5
    upper=float(profile.density_ratio_to_control.max())+2
    for i,sp in enumerate(SPECIES):
        density_panel(axes[0,i],sp,data,stats,20260924+i)
        bar_panel(axes[1,i],sp,bars,ymax)
        profile_panel(axes[2,i],sp,profile,upper)
    name_labels=[];cohort_labels=[]
    for i,sp in enumerate(SPECIES):
        x=axes[0,i].get_position().x0+axes[0,i].get_position().width/2
        n,af=COHORTS[sp]
        name_labels.append(fig.text(x,.890625,HEADERS[sp],fontsize=6.6,ha="center",va="center"))
        cohort_labels.append(fig.text(x,.8675,f"n={n:,}; AF≤{af:g}",fontsize=6,ha="center",va="center"))
    axes[0,0].set_ylabel("# rare SNVs per callable kb",labelpad=4)
    axes[1,0].set_ylabel("Mean rare SNV density\nrelative to control",labelpad=4)
    axes[2,0].set_ylabel("Rare SNV density\nrelative to control",labelpad=4)
    tree(fig,axes[0])
    fig.text(.01,.974,"a",fontsize=10,fontweight="bold",va="bottom")
    fig.text(.01,axes[2,0].get_position().y1-.005,"b",fontsize=10,fontweight="bold",va="bottom")
    x_axis_title=fig.text(.53,.04125,"Profile distance (kb)",fontsize=7,ha="center",va="center")
    handles,labels=axes[2,0].get_legend_handles_labels()
    legend=fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.53,0),ncol=3,
                      fontsize=6.5,handlelength=1.5,columnspacing=1.5,handletextpad=.4)
    place_significance_labels(fig,axes[0])
    fig.canvas.draw();renderer=fig.canvas.get_renderer();clipped=[]
    for axis in axes[1:].flat:
        minor=axis.get_yticklabels(minor=True)
        assert len(minor)==1 and minor[0].get_text()=="1"
        for major in axis.get_yticklabels():
            assert not minor[0].get_window_extent(renderer).overlaps(major.get_window_extent(renderer)),(
                "ratio 1 tick overlaps major tick",major.get_text())
    tick_overlaps=[]
    for row,axis_row in enumerate(axes):
        for col,axis in enumerate(axis_row):
            labels=[t for t in axis.get_xticklabels() if t.get_visible() and t.get_text()]
            for left_label,right_label in zip(labels,labels[1:]):
                left_box=left_label.get_window_extent(renderer)
                right_box=right_label.get_window_extent(renderer)
                if row!=1 and left_box.x1 > right_box.x0:
                    tick_overlaps.append((row,col,left_label.get_text(),right_label.get_text()))
    for kind,artists in (("species",name_labels),("cohort",cohort_labels)):
        for col,(left_label,right_label) in enumerate(zip(artists,artists[1:])):
            if left_label.get_window_extent(renderer).x1 > right_label.get_window_extent(renderer).x0:
                tick_overlaps.append((kind,col,left_label.get_text(),right_label.get_text()))
    x_title_box=x_axis_title.get_window_extent(renderer)
    c_tick_bottom=min(label.get_window_extent(renderer).y0 for ax in axes[2] for label in ax.get_xticklabels())
    legend_top=legend.get_window_extent(renderer).y1
    assert c_tick_bottom>x_title_box.y1,("profile ticks and title",c_tick_bottom,x_title_box.y1)
    assert x_title_box.y0>legend_top,("profile title and legend",x_title_box.y0,legend_top)
    for col,axis in enumerate(axes[0]):
        boxes=[Bbox.union([label.get_window_extent(renderer) for label in block])
               for block in axis.significance_labels]
        for block,box in zip(axis.significance_labels,boxes):
            text_value="".join(label.get_text() for label in block)
            assert box.y1 < axis.bbox.y1 and box.y0 > axis.bbox.y0,("significance height",col,text_value)
            assert box.x0 > axis.bbox.x0 and box.x1 < axis.bbox.x1,("significance width",col,text_value)
        for i,left_box in enumerate(boxes):
            for right_box in boxes[i+1:]:
                assert not left_box.overlaps(right_box),("significance overlap",col)
    for col,axis in enumerate(axes[1]):
        profile_top=axes[2,col].bbox.y1
        assert all(tick.get_window_extent(renderer).y0 > profile_top for tick in axis.get_xticklabels()),col
        assert axis.texts[0].get_window_extent(renderer).x0 > axis.spines["left"].get_window_extent(renderer).x1+.25,(
            "first bar value and y axis",col,
            axis.texts[0].get_window_extent(renderer).x0,
            axis.spines["left"].get_window_extent(renderer).x1)
        for left_index,left_value in enumerate(axis.texts):
            for right_value in axis.texts[left_index+1:]:
                assert not left_value.get_window_extent(renderer).overlaps(
                    right_value.get_window_extent(renderer)),(
                    "bar value labels overlap",col,left_value.get_text(),right_value.get_text(),
                    left_value.get_window_extent(renderer).bounds,
                    right_value.get_window_extent(renderer).bounds)
    for item in fig.findobj(base.mpl.text.Text):
        if not item.get_visible() or not item.get_text():continue
        assert item.get_fontfamily()[0]=="Arial",("unexpected font",item.get_text(),item.get_fontfamily())
        box=item.get_window_extent(renderer)
        if box.x0 < -2 or box.y0 < -2 or box.x1 > fig.bbox.width+2 or box.y1 > fig.bbox.height+2:
            clipped.append((item.get_text(),tuple(round(v,1) for v in box.bounds),
                            tuple(round(v,1) for v in fig.bbox.bounds)))
    assert not clipped,clipped
    assert not tick_overlaps,tick_overlaps
    paths=save_figure(fig);base.plt.close(fig)
    (OUT/"Fig1_layout.json").write_text(json.dumps(dict(width_mm=WIDTH_MM,height_mm=HEIGHT_MM,
        dpi=300,species=SPECIES,display_species=HEADERS,ratio_axis_unit_tick=1,
        panel_heights_mm=[44.54,26.01,30.6],superscript_layout={"engine":"matplotlib.mathtext",
            "font":"Arial","body_pt":6.4,"superscript_pt":4.48,"scale":0.7,
            "note":"Superscript-only size exception requested in the 2026-09-26 revision"},
        cohorts={sp:{"individuals":COHORTS[sp][0],"af_max":COHORTS[sp][1]} for sp in SPECIES},
        bar_group_names=[FULL_GROUP_NAMES[g] for g in GROUPS],
        profile_upper=upper,bar_upper=ymax,
        chicken_group_counts=data[data.species=="Chicken"].groupby("analysis_group").size().to_dict(),
        clipped_text=clipped,tick_label_overlaps=tick_overlaps),indent=2)+"\n")
    print("Inputs:",*[str(DATA/f"{n}.tsv") for n in names])
    print("Outputs:",*[str(x) for x in paths])

if __name__=="__main__": main()
