"""Test loaded route connectivity using the saved failure state, without motion."""
import os, runpy, sys
from pathlib import Path
import numpy as np

state = runpy.run_path(str(Path(__file__).with_name('probe_reorder_docking.py')))
check = state['check']
check.args.assets = Path(os.environ['MLSPACES_ASSETS_DIR'])
check.args.kitchen = True
check.args.nav_speed = .12
check.args.turn_speed = .08
for xy in ((-.09, 1.98), (-.19, 1.98), (-.19, 1.88)):
    try:
        route = check.plan_route(np.asarray(xy), carrying=True, face=0.0)
        print('CONNECTED', xy, [p.tolist() for p in route], flush=True)
        break
    except RuntimeError as exc:
        print('NO ROUTE', xy, str(exc), flush=True)
else:
    raise SystemExit(1)
