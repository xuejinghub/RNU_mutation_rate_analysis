"""Render manuscript figures from the bundled processed inputs."""
from rnu_common import ROOT
import argparse
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--figure',choices=['all','Fig1','Fig2','Fig3','Supplementary'],default='all')
    args=p.parse_args()
    scripts={'Fig1':'plot_fig1.py','Fig2':'plot_fig2.py','Fig3':'plot_fig3.py'}
    for key,script in scripts.items():
        if args.figure in ['all',key]:
            subprocess.run([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,check=True)
    if args.figure in ['all','Supplementary']:
        for script in ['plot_annotations.py','plot_figs3.py','plot_figs4.py','plot_figs7.py','plot_cohorts.py']:
            subprocess.run([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,check=True)


if __name__=='__main__':main()
