"""Final annotation diagrams: manuscript Supplementary Figures 1, 2, 5 and 6.

Drawing functions extracted from the 29 September final selection renderer.
Frozen memberships are checked against the current Fig. 1 before drawing.
"""
from rnu_common import ROOT, read, apply_journal_style, save_figure as export
import json
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
POL2=['U1','U2','U4','U5','U4atac','U11','U12']
ALL=['U1','U2','U4','U5','U6','U4atac','U6atac','U11','U12']
BLUE,ORANGE,GREY,LIGHT_BLUE='#0072B2','#E69F00','#E6E6E6','#DCEBF3'
CHECKS,BAR_ROWS,STAGE_ROWS=[],[],[]

def save_figure(fig,name,panels):
    for artist in fig.findobj(mpl.text.Text):
        artist.set_fontfamily('Arial')
        artist.set_fontsize(7)
    export(fig,name.split('_')[0])

def load():
    sets=[('human','Human','FigS1'),('macaque','Rhesus_macaque','FigS1'),
          ('chimp','Chimpanzee','FigS2'),('baboon','Baboon','FigS2'),
          ('mouse','Mouse','FigS5'),('rat','Rat','FigS5'),('chicken','Chicken','FigS6')]
    main=read('figures/Fig1/data/panel_a_gene_locus_density.tsv')
    out={}
    for key,species,figure in sets:
        data=read(f'figures/Supplementary/{figure}/data/{species}_final.tsv')
        ids=main.loc[main.species.eq(species)&main.analysis_group.eq('pol2_spliceRNUs'),'unit_id']
        normalize=lambda values:{str(v).split('.')[0] for v in values}
        check(species+' retained membership',normalize(data.analysis_unit_id),normalize(ids))
        out[key]=data
    out['chicken_pre']=read('figures/Supplementary/FigS6/data/Chicken_supported24.tsv')
    return out

def check(name, observed, expected=True):
    ok = observed == expected
    CHECKS.append(dict(check=name, observed=str(observed), expected=str(expected), passed=bool(ok)))
    if not ok:
        raise AssertionError((name, observed, expected))

def flow(ax, figid, panel, species, counts, boxes, filters, ys=None, height=None):
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    n = len(boxes)
    ys = np.linspace(.89, .09, n) if ys is None else ys
    height = (.105 if n >= 5 else .14) if height is None else height
    ax.flow_boxes, ax.flow_filters = [], []
    for i, (y, text) in enumerate(zip(ys, boxes)):
        patch = Rectangle((.045, y-height/2), .77, height, facecolor=LIGHT_BLUE if i==n-1 else GREY, edgecolor='none')
        ax.add_patch(patch)
        txt = ax.text(.43, y, text, ha='center', va='center', fontsize=7, linespacing=1.2)
        ax.flow_boxes.append((patch, txt))
        STAGE_ROWS.append(dict(figure=figid, panel=panel, species=species, step=i+1,
                               n=counts[i], definition=text.replace('\n', '; '),
                               incoming_filter='' if i==0 else filters[i-1].replace('\n', '; ')))
        if i:
            top, bottom = ys[i-1]-height/2, y+height/2
            ax.annotate('', xy=(.095,bottom+.006), xytext=(.095,top-.006),
                        arrowprops=dict(arrowstyle='->', lw=.85, color='#222222', shrinkA=0, shrinkB=0))
            txt = ax.text(.14, (top+bottom)/2, filters[i-1], ha='left', va='center', fontsize=7, linespacing=1.15)
            ax.flow_filters.append((txt, bottom, top))

def bars(ax, figid, panel, species, stage, frame, title, ymax, candidates=False, counts=False, legend=False):
    order = ALL if candidates else POL2
    c = frame.set_index('family')['count'] if counts else frame.plot_family.value_counts()
    vals = [int(c.get(f,0)) for f in order]
    expected = int(frame['count'].sum()) if counts else len(frame)
    check(f'{figid}{panel} bars sum', sum(vals), expected)
    x = np.arange(len(order))
    ax.bar(x, vals, width=.72, color=[ORANGE if f in ['U6','U6atac'] else BLUE for f in order], linewidth=0)
    for xi, fam, v in zip(x, order, vals):
        ax.text(xi, v+ymax*.019, str(v), ha='center', va='bottom', fontsize=7)
        BAR_ROWS.append(dict(figure=figid, panel=panel, species=species, stage=stage, family=fam, count=v))
    ax.set(xticks=x, xticklabels=order, ylim=(0,ymax), ylabel='Number of genes', xlabel='snRNA family')
    ax.tick_params(axis='x', labelrotation=45)
    for t in ax.get_xticklabels(): t.set_ha('right')
    step = 20 if ymax>=60 else 10 if ymax>=30 else 5 if ymax>=15 else 2
    ax.set_yticks(np.arange(0, ymax+.01, step))
    ax.spines[['top','right']].set_visible(False)
    ax.set_title(f'{title}\n(n = {expected})', pad=8, fontsize=7, linespacing=1.2)
    if legend:
        ax.text(.98,.91,'Pol II families', color=BLUE, ha='right', va='top', transform=ax.transAxes)
        ax.text(.98,.77,'U6 / U6atac (Pol III)', color='#8D6100', ha='right', va='top', transform=ax.transAxes)

def label(fig, txt, x, y):
    fig.text(x, y, txt, weight='bold', fontsize=7, va='top')

def paired_layout(top_title, bottom_title):
    fig = plt.figure(figsize=(180/25.4,188/25.4))
    axes = [fig.add_axes(r) for r in [[.055,.535,.46,.405],[.635,.615,.33,.255],
                                     [.055,.045,.46,.405],[.635,.125,.33,.255]]]
    for ch,x,y in [('a',.025,.974),('b',.565,.974),('c',.025,.484),('d',.565,.484)]: label(fig,ch,x,y)
    fig.text(.08,.965,top_title,va='top'); fig.text(.08,.475,bottom_title,va='top')
    return fig, axes

def draw(t):
    fig,(a,c1,c,c2)=paired_layout('Human · GRCh38','Rhesus macaque · rheMac10')
    flow(a,'S1','a','Human',[200,82,44,41],
         ['200 putatively functional snRNA genes\nLiterature-guided candidate set [1]',
          '82 candidate Pol II\nspliceosomal snRNA genes','44 POLR2A-supported loci\nwith harmonized GENCODE identities',
          '41 Pol II splice-RNUs\nretained for mutation analysis'],
         ['Exclude U6 and U6atac\n(118 genes)','Overlap POLR2A peaks\n(mean fold enrichment > 2)',
          'Autosomes; coverage AND mappability;\n≥50 effective callable bases'])
    bars(c1,'S1','b','Human','Final Pol II',t['human'],'Human Pol II splice-RNUs',30)
    flow(c,'S1','c','Rhesus macaque',[41,29,28],
         ['41 human Pol II splice-RNUs\nused as mapping queries','29 one-to-one\nsynteny relationships',
          '28 macaque Pol II splice-RNUs\nretained for mutation analysis'],
         ['Tiered gene-body / flanking synteny;\nfamily-aware candidate assignment;\nexclude non-unique synteny relationships',
          'Coverage AND mappability;\nmapped interval ≥50 callable bases;\ngene ≥50 effective callable bases'],ys=[.87,.48,.09])
    bars(c2,'S1','d','Rhesus macaque','Final Pol II',t['macaque'],'Macaque Pol II splice-RNUs',15)
    fig.text(.055,.012,'[1] Leitão et al., Nature Genetics (2026). doi:10.1038/s41588-026-02547-5')
    save_figure(fig,'FigS1_human_macaque_PolII_selection','a–d')

    fig,(a,c1,c,c2)=paired_layout('Chimpanzee · panTro5 (synteny)','Baboon · Panubis1.0')
    flow(a,'S2','a','Chimpanzee',[41,41,28],
         ['41 human Pol II splice-RNUs\nprocessed within their own query class',
          '41 positional candidate assignments\n36 core; 5 flank-supported',
          '28 one-to-one synteny relationships\n27 core; 1 flank-supported'],
         ['Macaque-style gene-body / flank synteny;\nfamily-compatible annotation;\nsequence-supported rescue prioritized',
          'Require unique targets within Pol II class;\nexclude 13 target-sharing queries'],ys=[.87,.48,.09])
    bars(c1,'S2','b','Chimpanzee','Final Pol II',t['chimp'],'Chimpanzee Pol II splice-RNUs',15)
    flow(c,'S2','c','Baboon',[41,27,23,18],
         ['41 human Pol II splice-RNUs\nused as mapping queries','27 one-to-one synteny relationships\nwith annotated snRNA targets',
          '23 autosomal\nPol II splice-RNU candidates','18 baboon Pol II splice-RNUs\nretained for mutation analysis'],
         ['Macaque-style core / flank synteny;\nEnsembl release 115 snRNA targets',
          'Restrict to chromosomes 1–20\n(exclude four scaffold loci)',
          'Coverage AND mappability;\n≥50 effective callable bases'])
    bars(c2,'S2','d','Baboon','Final Pol II',t['baboon'],'Baboon Pol II splice-RNUs',10)
    fig.text(.055,.012,'Chimpanzee variants use paired hg38 intervals; no independent chimpanzee callable mask.')
    save_figure(fig,'FigS2_chimpanzee_baboon_PolII_selection','a–d')

    fig,(a,c1,c,c2)=paired_layout('Mouse · mm10 / GRCm38','Rat · rn7 / mRatBN7.2')
    flow(a,'S5','a','Mouse',[1383,120,114,50,39],
         ['1,383 annotated snRNA genes\nGENCODE vM25; chr1–19, X and Y',
          '120 regulatory / named RNA candidates\n114 cCRE-overlapping OR 11 named',
          '114 spliceosomal candidates\n50 Pol II; 64 U6/U6atac',
          '50 candidate Pol II splice-RNUs\nseven spliceosomal families',
          '39 Pol II splice-RNUs\nretained for mutation analysis'],
         ['cCRE overlap OR named RNA list;\nalso adds Rnu73b (snoRNA)',
          'Exclude six non-spliceosomal RNAs\n(including U7 and Rnu73b)',
          'Archived family assignments;\nexclude U6 and U6atac (Pol III)',
          'Autosomes; coverage AND mappability;\n≥50 effective callable bases'],ys=[.90,.70,.50,.30,.10],height=.085)
    bars(c1,'S5','b','Mouse','Final Pol II',t['mouse'],'Mouse Pol II splice-RNUs',22)
    flow(c,'S5','c','Rat',[1512,1418,92,52,46],
         ['1,512 annotated snRNA genes\nEnsembl release 112','1,418 autosomal snRNA genes\nEnsembl / RGD + Rfam family assignment',
          '92 shared-ATAC spliceosomal candidates\n52 Pol II; 40 U6/U6atac',
          '52 candidate Pol II splice-RNUs\nseven spliceosomal families','46 Pol II splice-RNUs\nretained for mutation analysis'],
         ['Restrict to chromosomes 1–20',
          'Spliceosomal families; TSS ±50 bp\noverlaps the intersection of peak intervals\nfrom two rat ATAC-seq atlases',
          'Exclude U6 and U6atac (40 loci)',
          'Coverage AND mappability;\n≥50 effective callable bases'],ys=[.90,.70,.50,.30,.10],height=.085)
    bars(c2,'S5','d','Rat','Final Pol II',t['rat'],'Rat Pol II splice-RNUs',20)
    save_figure(fig,'FigS5_mouse_rat_PolII_selection','a–d')

    fig=plt.figure(figsize=(180/25.4,180/25.4))
    a=fig.add_axes([.055,.055,.46,.89]);c1=fig.add_axes([.635,.63,.33,.25]);c2=fig.add_axes([.635,.15,.33,.25])
    for txt,x,y in [('a',.025,.975),('b',.565,.975),('c',.565,.49)]:label(fig,txt,x,y)
    fig.text(.08,.968,'Chicken · GRCg7b',va='top')
    flow(a,'S6','a','Chicken',[82,74,72,24,15,14],
         ['82 annotated snRNA genes\nEnsembl release 115','74 autosomal\nsnRNA genes','72 autosomal\nnon-U7 snRNA genes',
          '24 supported spliceosomal candidates\n15 Pol II; 9 U6/U6atac',
          '15 candidate Pol II splice-RNUs\n14 PRO-seq-supported; 1 Ser5P-supported','14 Pol II splice-RNUs\nretained for mutation analysis'],
         ['Restrict to chromosomes 1–39\n(exclude eight Z/W-linked loci)','Exclude two U7 genes',
          'Pol II: (Ser5P + ATAC) OR\n(replicated sense PRO-seq + chromatin);\nU6/U6atac: active chromatin support',
          'Exclude U6 and U6atac\n(9 supported loci)',
          'Coverage AND mappability;\n≥50 effective callable bases;\nexclude one locus with no callable bases'],ys=[.90,.74,.58,.42,.26,.10],height=.085)
    bars(c1,'S6','b','Chicken','Supported candidates',t['chicken_pre'],'Supported spliceosomal candidates',12,True,False,True)
    bars(c2,'S6','c','Chicken','Final Pol II',t['chicken'],'Chicken Pol II splice-RNUs',10)
    fig.text(.055,.018,'Regulatory windows: ATAC / active states, TSS ±50 bp; histone marks, see legend.')
    save_figure(fig,'FigS6_chicken_PolII_selection','a–c')

if __name__=='__main__':
    apply_journal_style()
    draw(load())
    folder=ROOT/'results'
    folder.mkdir(exist_ok=True)
    import pandas as pd
    pd.DataFrame(BAR_ROWS).to_csv(folder/'annotation_family_counts.tsv',sep='\t',index=False)
    pd.DataFrame(STAGE_ROWS).to_csv(folder/'annotation_flow_counts.tsv',sep='\t',index=False)
    (folder/'annotation_checks.json').write_text(json.dumps(CHECKS,indent=2)+'\n')
