"""Repository-relative IO and deterministic publication export helpers."""
from pathlib import Path
import os
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / 'tmp'
TMP.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(TMP / 'matplotlib'))
os.environ.setdefault('TMPDIR', str(TMP))


def read(relative):
    import pandas as pd
    return pd.read_csv(ROOT / relative, sep='\t', float_precision='round_trip')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def apply_journal_style(size=7):
    """Require an installed/licensed Arial; never bundle or silently substitute it."""
    import matplotlib as mpl
    from matplotlib import font_manager, _mathtext
    font_dir = os.environ.get('RNU_FONT_DIR')
    if font_dir:
        for name in ('arial.ttf', 'arialbd.ttf', 'ariali.ttf', 'arialbi.ttf'):
            path = Path(font_dir) / name
            if not path.is_file():
                raise FileNotFoundError(f'Arial font missing: {path}')
            font_manager.fontManager.addfont(str(path))
    try:
        font_manager.findfont('Arial', fallback_to_default=False)
    except ValueError as exc:
        raise RuntimeError('Install licensed Arial or set RNU_FONT_DIR to its directory.') from exc
    mpl.rcParams.update({
        'font.family':'Arial', 'font.sans-serif':['Arial'], 'font.size':size,
        'axes.labelsize':size, 'axes.titlesize':size, 'xtick.labelsize':size,
        'ytick.labelsize':size, 'legend.fontsize':size, 'legend.frameon':False,
        'axes.linewidth':.65, 'xtick.major.width':.6, 'ytick.major.width':.6,
        'xtick.major.size':2.6, 'ytick.major.size':2.6,
        'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
        'savefig.dpi':300, 'savefig.facecolor':'white', 'figure.facecolor':'white',
        'mathtext.fontset':'custom', 'mathtext.rm':'Arial', 'mathtext.it':'Arial:italic',
        'mathtext.bf':'Arial:bold', 'mathtext.sf':'Arial', 'mathtext.tt':'Arial',
        'mathtext.cal':'Arial', 'mathtext.fallback':None, 'mathtext.default':'regular',
        'svg.hashsalt':'rnu-manuscript-20260929',
    })
    class ArialConstants(_mathtext.FontConstantsBase):
        sup1 = .40
    _mathtext._font_constant_mapping['Arial'] = ArialConstants


def save_figure(fig, name):
    import matplotlib.pyplot as plt
    out = ROOT / 'figures/final'
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(out / f'{name}.{ext}', dpi=300, facecolor='white', transparent=False)
    size = (fig.get_size_inches() * 25.4).tolist()
    (out / f'{name}_export.json').write_text(json.dumps({
        'dimensions_mm':size, 'png_dpi':300, 'font':'Arial',
        'layout':'portable reproduction from bundled processed inputs'}, indent=2)+'\n')
    plt.close(fig)
    print(f'{name}: PDF / SVG / PNG; {size[0]:g} x {size[1]:g} mm')
