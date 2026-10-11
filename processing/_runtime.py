"""Load processing components and dispatch their command-line stages.

Adapters reference functions defined directly in the merged modules. They never
open or execute legacy script files. Attribute writes update the same globals
read by the retained numerical functions, preserving production configuration.
"""
from pathlib import Path
import importlib
import json
import sys

PROCESSING_ROOT=Path(__file__).resolve().parent
_REGISTRY=json.loads((PROCESSING_ROOT/'component_registry.json').read_text())
_VIEWS={}


class StageState:
    """Expose one stage's functions and configuration from its merged module."""
    def __init__(self,key):
        object.__setattr__(self,'_key',key)

    def _module(self):
        path=_REGISTRY[self._key]['file']
        name='processing.'+str(Path(path).with_suffix('')).replace('/','.')
        return importlib.import_module(name)

    def __getattr__(self,name):
        spec=_REGISTRY[self._key]
        if name=='__file__':return str(PROCESSING_ROOT/spec['file'])
        target=spec['exports'].get(name,f"{spec['prefix']}_{name}")
        return getattr(self._module(),target)

    def __setattr__(self,name,value):
        spec=_REGISTRY[self._key]
        target=spec['exports'].get(name,f"{spec['prefix']}_{name}")
        setattr(self._module(),target,value)


def component(key):
    """Return a lazy component view using the shared registry."""
    if key not in _REGISTRY:raise KeyError(f'Unknown consolidated stage: {key}')
    if key not in _VIEWS:_VIEWS[key]=StageState(key)
    return _VIEWS[key]


def component_file(key):
    return PROCESSING_ROOT/_REGISTRY[key]['file']


def run_stages(stages,namespace,setup=None):
    """Select a requested stage, prepare its paths and invoke its entry point."""
    args=sys.argv[1:]
    if not args or args[0] in ['-h','--help']:
        print('Usage: python '+Path(sys.argv[0]).name+' STAGE [stage arguments]')
        print('Available stages: '+', '.join(stages))
        print('Requires configured production inputs; this is not the frozen-data plotting command.')
        return
    stage=args.pop(0)
    if stage not in stages:raise SystemExit(f'Unknown stage {stage!r}; choose from {list(stages)}')
    sys.argv=[sys.argv[0]]+args
    if setup is not None:setup()
    namespace[stages[stage]]()
