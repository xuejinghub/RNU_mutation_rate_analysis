"""Portable numerical reproduction of Fig2 (NOT the pixel-identical manuscript layout).

Renders current gene counts, corrected strand scores, legacy corrected densities,
SBS96 pseudo-counts, frozen signature fits/proportions and TCW rates. No refitting.
The species-count bar replaces the manuscript cladogram; all values are retained.
"""
from rnu_common import ROOT, read, apply_journal_style, save_figure
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

GROUPS=['Pol II splice-RNUs','U6','Pseudo-RNUs']
LABELS=['Pol II splice-RNUs','U6/U6ATAC','Pseudo-RNUs']
COLORS=['#2077B5','#FF800D','#2BA02B']
MUTATIONS=['C>A','C>G','C>T','T>A','T>C','T>G']
SBS_COLORS=['#56B4E9','#222222','#D55E00','#999999','#009E73','#CC79A7']


def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(direction='out',pad=2)


def label(fig,ax,letter):
    p=ax.get_position()
    fig.text(p.x0-.035,p.y1+.009,letter,weight='bold',fontsize=9,va='bottom')


def main(cohort=None):
    apply_journal_style()
    number={'EAS':8,'SAS':9,'FIN':10,'random2000':11}
    base=f'figures/Supplementary/FigS{number[cohort]}/data/' if cohort else 'figures/Fig2/data/'
    asym=read(base+'panel_a_asymmetry.tsv')
    densities=read(base+'panel_a_density.tsv')
    matrix=read(base+'panel_b_matrix.tsv').set_index('Mutation Types')
    fits=read(base+'panel_c_fits.tsv')
    activities=read(base+'panel_c_activities.tsv')
    rates=read(base+'panel_d_rates.tsv')
    fig=plt.figure(figsize=(180/25.4,225/25.4))
    # Fixed final-size placement; legends occupy dedicated whitespace.
    if cohort is None:
        counts=read(base+'apobec3_gene_counts.tsv')
        ax=fig.add_axes([.17,.757,.21,.20])
        ax.barh(np.arange(len(counts)),counts.gene_count,color='#777777',height=.65)
        ax.set_yticks(np.arange(len(counts)),counts.display)
        ax.invert_yaxis();ax.set_xlabel('APOBEC3 genes');ax.set_xlim(0,10.5)
        for i,r in enumerate(counts.itertuples()):ax.text(r.gene_count+.2,i,str(r.gene_count),va='center')
        clean(ax);label(fig,ax,'a')
        rectangles=[[.49,.805,.46,.152],[.49,.696,.205,.068],[.76,.696,.19,.068]]
    else:
        rectangles=[[.12,.805,.83,.152],[.12,.696,.35,.068],[.60,.696,.35,.068]]
    axes=[fig.add_axes(rect) for rect in rectangles]
    x=np.arange(6)
    for j,(g,color) in enumerate(zip(GROUPS,COLORS)):
        sub=asym[asym.group.eq(g)].set_index('mutation').reindex(MUTATIONS)
        axes[0].bar(x+(j-1)*.23,sub.displayed_score,width=.22,color=color,label=LABELS[j])
        # Non-finite historical scores are explicitly marked; zero is not symmetry.
        for xi,r in zip(x+(j-1)*.23,sub.itertuples()):
            if not np.isfinite(r.corrected_log2_asymmetry):axes[0].text(xi,0,'×',ha='center',va='bottom')
    axes[0].axhline(0,color='#555555',lw=.6)
    axes[0].set_xticks(x,MUTATIONS);axes[0].set_ylabel('Corrected log2(NTS/TS)')
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,1.24),ncol=3,handlelength=.8,columnspacing=.5)
    label(fig,axes[0],'a' if cohort else 'b')
    for ax,site in zip(axes[1:],['non-CpG','CpG']):
        for j,(g,col) in enumerate(zip(GROUPS,COLORS)):
            part=densities[densities.group.eq(g)&densities.site_class.eq(site)]
            values=[part.loc[part.mutation_pair.str.startswith(m),'notebook_merged_density'].iloc[0]
                    if part.mutation_pair.str.startswith(m).any() else np.nan for m in MUTATIONS]
            ax.bar(x+(j-1)*.23,values,width=.22,color=col)
        ax.set_xticks(x,MUTATIONS,rotation=55);ax.set_title(site,pad=3)
        ax.set_ylabel('Corrected density',labelpad=1)
    for ax in axes:clean(ax)
    colors=[SBS_COLORS[MUTATIONS.index(s[s.index('[')+1:s.index(']')])] for s in matrix.index]
    for i,(g,title) in enumerate(zip(GROUPS,LABELS)):
        ax=fig.add_axes([.12,.548-i*.077,.83,.050])
        values=matrix[g]/matrix[g].sum()*100
        ax.bar(np.arange(96),values,color=colors,width=.9,linewidth=0)
        ax.set_xlim(-1,96);ax.set_ylabel('%',labelpad=3)
        ax.text(.01,.97,title,transform=ax.transAxes,va='top')
        if i==2:
            contexts=[s[0]+s[s.index('[')+1]+s[-1] for s in matrix.index]
            # All 96 bars are drawn; every fourth context is labeled for 7 pt readability.
            ax.set_xticks(range(0,96,4),contexts[::4],rotation=90,fontsize=7)
            ax.tick_params(axis='x',length=1,pad=1)
        else:ax.set_xticks([])
        clean(ax)
        if i==0:label(fig,ax,'b' if cohort else 'c')
    fig.legend([Patch(color=c) for c in SBS_COLORS],MUTATIONS,loc='center',
               bbox_to_anchor=(.54,.626),ncol=6,handlelength=.9,columnspacing=1.5)
    fitax=fig.add_axes([.12,.174,.35,.135])
    models=['base','base_apobec','all_cosmic']
    mcolors=['#777777','#0072B2','#E69F00']
    for j,(model,col) in enumerate(zip(models,mcolors)):
        part=fits[fits.model.eq(model)].set_index('sample').reindex(GROUPS)
        fitax.bar(np.arange(3)+(j-1)*.24,part.cosine_similarity,width=.23,color=col)
    fitax.set_xticks(range(3),['Pol II','U6','Pseudo']);fitax.set_ylabel('Cosine similarity');fitax.set_ylim(0,1)
    clean(fitax);label(fig,fitax,'c' if cohort else 'd')
    comp=fig.add_axes([.12,.058,.35,.065])
    bottom=np.zeros(3)
    signatures=['SBS5','SBS1','SBS2ss*','SBS13'];sc=['#777777','#56B4E9','#D55E00','#CC79A7']
    for sig,col in zip(signatures,sc):
        part=activities[activities.model.eq('base_apobec')&activities.signature.eq(sig)].set_index('group')
        vals=part.reindex(GROUPS).proportion.fillna(0).to_numpy()*100
        comp.bar(range(3),vals,bottom=bottom,color=col,width=.55);bottom+=vals
    comp.set_xticks(range(3),['Pol II','U6','Pseudo']);comp.set_ylabel('Activity (%)');comp.set_ylim(0,100);clean(comp)
    fig.legend([Patch(color=c) for c in mcolors],['Base','Base + APOBEC','67-signature model'],loc='center',bbox_to_anchor=(.30,.339),ncol=1)
    fig.legend([Patch(color=c) for c in sc],signatures,loc='center',bbox_to_anchor=(.30,.018),ncol=2)
    tax=fig.add_axes([.61,.16,.34,.149])
    species=list(dict.fromkeys(rates.species))
    for j,(g,col) in enumerate(zip(GROUPS,COLORS)):
        part=rates[rates.group.eq(g)].set_index('species').reindex(species)
        y=part.tcw_percent.to_numpy()
        tax.errorbar(np.arange(len(species))+(j-1)*.20,y,
            yerr=[y-part.wilson_95ci_low_percent.to_numpy(),part.wilson_95ci_high_percent.to_numpy()-y],
            fmt='o',ms=3,color=col,capsize=2,lw=.8)
    tax.set_xticks(range(len(species)),[s.replace('Rhesus macaque','Macaque').replace('Rhesus_macaque','Macaque') for s in species],rotation=30,ha='right')
    tax.set_ylabel('TCW SNVs / eligible positions (%)');tax.set_ylim(bottom=0);clean(tax);label(fig,tax,'d' if cohort else 'e')
    fig.text(.60,.087,'Frozen nominal estimates.\nError bars: legacy Wilson 95% CI.',va='top',linespacing=1.4)
    save_figure(fig,f'FigS{number[cohort]}' if cohort else 'Fig2')


if __name__=='__main__':main()
