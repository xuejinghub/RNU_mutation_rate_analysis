"""Mechanically extracted plotting functions for the saved Fig. 3 results.
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

from matplotlib.collections import PathCollection
from matplotlib.patches import Patch
from matplotlib.text import Text
FINAL=ROOT/'figures/final'
DATA=ROOT/'figures/Fig3/data'
GROUPS=['POL2_spliceRNUs_strict41','functional_U6_U6ATAC_116','RNU_pseudogene_operational_complement']
NAMES=['POLII spliceRNUs','U6/U6ATAC','PseudoRNUs']
TICKS=['POLII\nspliceRNUs','U6/\nU6ATAC','PseudoRNUs']
COLORS=['#2077b5','#ff800d','#2ba02b']
OVERLAP_GROUPS=['pol2_spliceRNUs','functional_U6s','RNU_pseudogenes']
LABEL='A3B-binding ssDNA signal fold enrichment\n(IP/input; T47D cells)'
RLOOP_LABEL='R-loop R-ChIP fold enrichment\n(IP/Input; HEK293 cells)'
WIDTH,HEIGHT=180,225
def prepare_data():
    return {p.stem:pd.read_csv(p,sep='\t',float_precision='round_trip') for p in DATA.glob('*.tsv')}
def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(direction='out',length=3)

def p_math(p):
    if p<.001:
        exponent=int(np.floor(np.log10(p)))
        return rf'\mathit{{P}} = {p/10**exponent:.2f}\times10^{{{exponent}}}'
    return rf'\mathit{{P}} = {p:.3g}'

def bracket(ax,x1,x2,y,h,text):
    ax.plot([x1,x1,x2,x2],[y,y+h,y+h,y],color='#333333',lw=.7)
    ax.text((x1+x2)/2,y+h+h*.45,text,ha='center',va='bottom',fontsize=7.5,fontweight='normal')

def draw_boxes(ax,values,stats,panel):
    arrays=[values.loc[values.gene_set.eq(g),'FE'].dropna().to_numpy() for g in GROUPS]
    boxes=ax.boxplot(arrays,positions=range(3),widths=.52,whis=1.5,showfliers=False,patch_artist=True,
        medianprops={'color':'#111111','linewidth':1,'zorder':5},
        boxprops={'edgecolor':'#888888','linewidth':.7},
        whiskerprops={'color':'#555555','linewidth':.7},capprops={'color':'#555555','linewidth':.7})
    rng=np.random.default_rng(1709 if panel=='a' else 1710)
    displayed=[];audits=[]
    for i,(g,v,color) in enumerate(zip(GROUPS,arrays,COLORS)):
        boxes['boxes'][i].set_facecolor(color);boxes['boxes'][i].set_alpha(.70)
        points=v[v<=200] if panel=='c' else v
        displayed.extend(points)
        ax.scatter(i+rng.uniform(-.23,.23,len(points)),points,s=[7,5,3][i],
            alpha=.32 if len(points)<300 else .15,color=color,edgecolors='none',zorder=3)
        audits.append(dict(panel=panel,gene_set=g,n_tested=len(v),n_dots=len(points),n_hidden_high=len(v)-len(points)))
    upper=max(displayed)*1.06
    idx=stats.set_index(['left_gene_set','right_gene_set'])
    for level,(i,j) in enumerate([(0,1),(1,2),(0,2)]):
        row=idx.loc[(GROUPS[i],GROUPS[j])]
        text=rf'${row.median_fold_left_over_right:.2f}\times;\, '+p_math(row.p_two_sided_asymptotic)+'$'
        bracket(ax,i,j,upper*(1.03+level*.20),upper*.018,text)
    ax.set_ylim(0,upper*1.65);ax.set_xlim(-.6,2.6)
    labels=[f'{t}\n({len(v):,})' for t,v in zip(TICKS,arrays)] if panel=='a' else TICKS
    ax.set_xticks(range(3),labels);ax.set_ylabel(LABEL if panel=='a' else RLOOP_LABEL)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5,steps=[1,2,5,10]));clean(ax)
    return audits

def draw_profile(ax,legend_ax,data,rloop=False):
    ax.axvspan(0,1000,color='#EEEEEE',zorder=0)
    ax.axhline(1,color='#AAAAAA',linestyle=(0,(2,2)),linewidth=.6)
    for g,color,label in zip(GROUPS,COLORS,NAMES):
        p=data[data.gene_set.eq(g)].sort_values('bin_index')
        ax.plot(p.display_x,p.mean_fold_enrichment,color=color,lw=1.2,label=label)
    ax.set_xlim(-1000,2000)
    ax.set_ylim(data.mean_fold_enrichment.min(),data.mean_fold_enrichment.max()*1.12)
    ax.set_xticks([-1000,0,1000,2000],['−2 kb','TSS','TES','+2 kb'])
    ax.get_xticklabels()[0].set_ha('left');ax.get_xticklabels()[-1].set_ha('right')
    ax.set_ylabel(RLOOP_LABEL if rloop else LABEL);clean(ax)
    handles,labels=ax.get_legend_handles_labels()
    legend_ax.legend(handles,labels,loc='center',bbox_to_anchor=(.5,.32),ncol=2,
        handlelength=1.2,columnspacing=.8,handletextpad=.35,labelspacing=.35)

def draw_rnase(ax,header,values,stats):
    rng=np.random.default_rng(1717)
    conditions=[('lipid_control','Control',-.18,'#80B1D3'),('rnaseh1','+RNaseH1',.18,'#FDB462')]
    maxima=[];audit=[]
    for i,g in enumerate(GROUPS):
        maximum=0
        for key,label,offset,color in conditions:
            v=values.loc[values.gene_set.eq(g)&values.condition.eq(key),'spi_seq_fold_enrichment'].dropna().to_numpy()
            boxes=ax.boxplot([v],positions=[i+offset],widths=.29,whis=1.5,showfliers=False,patch_artist=True,
                medianprops={'color':'#111111','linewidth':.9,'zorder':5},boxprops={'edgecolor':'#555555','linewidth':.65},
                whiskerprops={'color':'#555555','linewidth':.65},capprops={'color':'#555555','linewidth':.65})
            boxes['boxes'][0].set_facecolor(color)
            ax.scatter(i+offset+rng.uniform(-.12,.12,len(v)),v,s=[7,5,3][i],alpha=[.6,.4,.2][i],color=color,edgecolors='none',zorder=3)
            maximum=max(maximum,float(v.max()))
            audit.append(dict(panel='e',gene_set=g,condition=key,n_tested=len(v),n_dots=len(v),n_hidden_high=0))
        maxima.append(maximum)
    height=max(maxima)*1.18
    for i,g in enumerate(GROUPS):
        r=stats.set_index('gene_set').loc[g]
        label=f'{r.median_fold_rnaseh1_over_control:.2f}×\n$'+p_math(r.p_two_sided_asymptotic)+'$'
        bracket(ax,i-.18,i+.18,maxima[i]+height*.025,height*.014,label)
    ax.set_ylim(0,height);ax.set_xlim(-.6,2.6);ax.set_xticks(range(3),TICKS)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5,steps=[1,2,5,10]))
    ax.set_ylabel(LABEL);clean(ax)
    header.legend(handles=[Patch(facecolor=c,label=label) for key,label,offset,c in conditions],loc='center right',ncol=2)
    return audit

def draw_overlap(ax,header,summary,stats):
    for species,offset,color,marker in [('Human',-.14,'#0072B2','o'),('Mouse',.14,'#D55E00','s')]:
        for i,g in enumerate(OVERLAP_GROUPS):
            r=summary[summary.analysis_group.eq(g)&summary.species.eq(species)].iloc[0]
            ax.errorbar(i+offset,r.percent,yerr=[[r.percent-r.ci95_low_percent],[r.ci95_high_percent-r.percent]],
                fmt=marker,color=color,markeredgecolor='white',markeredgewidth=.45,markersize=4.2,
                elinewidth=.9,capsize=1.8,capthick=.75,label=species if i==0 else None,zorder=3)
    for i,g in enumerate(OVERLAP_GROUPS):
        y=min(summary.loc[summary.analysis_group.eq(g),'ci95_high_percent'].max()+2.5,96.3)
        p=stats.loc[stats.analysis_group.eq(g),'fisher_exact_p_two_sided'].iloc[0]
        star='****' if p<.0001 else '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'ns'
        bracket(ax,i-.14,i+.14,y-1.3,1.3,star)
    ax.set_xlim(-.6,2.6);ax.set_ylim(0,104);ax.set_yticks([0,25,50,75,100])
    ax.set_xticks(range(3),['POLII\nspliceRNUs','U6/U6ATAC','PseudoRNUs'])
    ax.set_ylabel('R-loop-overlapping genes (%)');ax.yaxis.grid(True,color='#D9D9D9',lw=.5);clean(ax)
    handles,_=ax.get_legend_handles_labels()
    header.legend(handles,['Human (RLBase v2)','Mouse (9 mESC R-loop datasets)'],loc='center left',ncol=1)

def save_figure(fig):
    for artist in fig.findobj(PathCollection):artist.set_rasterized(False)
    target=FINAL/'Fig3'
    for ext in ['pdf','svg','png']:
        fig.savefig(target.with_suffix('.'+ext),dpi=300,facecolor='white',transparent=False)

def main():
    FINAL.mkdir(parents=True,exist_ok=True)
    tables=prepare_data();apply_journal_style(7.5);mpl.rcParams.update({"axes.labelsize":8})
    fig=plt.figure(figsize=(WIDTH/25.4,HEIGHT/25.4),facecolor='white')
    outer=fig.add_gridspec(3,1,left=.095,right=.988,top=.975,bottom=.055,height_ratios=[1.12,1.12,1],hspace=.07)
    axes=[];headers=[];legends=[]
    for row in range(3):
        grid=outer[row].subgridspec(3,2,height_ratios=[.065,.74,.195] if row<2 else [.115,.79,.095],
            hspace=.035,wspace=.35 if row<2 else .32)
        for col in range(2):
            header=fig.add_subplot(grid[0,col]);header.set_axis_off();headers.append(header)
            axes.append(fig.add_subplot(grid[1,col]))
            leg=fig.add_subplot(grid[2,col]);leg.set_axis_off();legends.append(leg)
    audits=draw_boxes(axes[0],tables['a_gene_values'],tables['a_pairwise'],'a')
    draw_profile(axes[1],legends[1],tables['b_profile'])
    audits+=draw_boxes(axes[2],tables['c_gene_values'],tables['c_pairwise'],'c')
    draw_profile(axes[3],legends[3],tables['d_profile'],True)
    audits+=draw_rnase(axes[4],headers[4],tables['e_gene_values'],tables['e_paired'])
    draw_overlap(axes[5],headers[5],tables['f_summary'],tables['f_comparisons'])
    for label,ax,header in zip('abcdef',axes,headers):
        fig.text(ax.get_position().x0-.036,header.get_position().y1,label,fontsize=10,fontweight='bold',va='top')
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=[];skip=set()
    for ax in axes:
        for axis,lim in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if not min(lim)<=tick.get_loc()<=max(lim):skip.update([tick.label1,tick.label2])
    for tx in fig.findobj(Text):
        if tx.get_visible() and tx.get_text() and tx not in skip:
            b=tx.get_window_extent(renderer)
            assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height,tx.get_text()
            assert tx.get_fontsize()>=7.5
            bounds.append(dict(text=tx.get_text(),font_pt=tx.get_fontsize(),bounds=[b.x0,b.y0,b.x1,b.y1]))
    save_figure(fig)
    plt.close(fig)
    print('Generated Fig3: 180 x 225 mm; all scientific drawing functions retained.')

if __name__=="__main__": main()
