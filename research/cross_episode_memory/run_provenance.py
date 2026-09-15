"""Capture the code and settings used by an oracle validation run."""
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys


def capture_run(output, arguments):
    root = Path(__file__).resolve().parents[2]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    files = {Path(module.__file__).resolve() for module in tuple(sys.modules.values())
             if getattr(module, '__file__', None) and str(module.__file__).endswith('.py')}
    package = root / 'research/cross_episode_memory'
    files.update(package.glob('*.py'))
    files.update((package / 'tools').glob('*.py'))
    files.update(root / 'research/cross_episode_memory' / name for name in (
        'tools/run_reorder_chain.sh', 'tools/run_procthor_reorder.sh', 'SPEC.md',
        'CHECK_REORDER_CHAIN.md', 'CHECK_PROCTHOR_TABLES.md'))
    hashes = {}
    for path in sorted(files):
        if not path.is_relative_to(root) or not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts[0] not in ('research', 'molmo_spaces'):
            continue
        content = path.read_bytes()
        target = output / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        hashes[str(relative)] = hashlib.sha256(content).hexdigest()
    settings = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(arguments).items()}
    packages = {}
    for package in ('mujoco', 'torch', 'nvidia-curobo', 'numpy', 'scipy'):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            pass
    manifest = dict(source_sha256=hashes, arguments=settings, packages=packages,
                    python=sys.executable, environment={key: os.environ.get(key) for key in (
                        'CUDA_VISIBLE_DEVICES', 'MUJOCO_GL', 'OMP_NUM_THREADS',
                        'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'FRIDGE_SHELF')})
    identity = dict(source_sha256={path: digest for path, digest in hashes.items()
                                   if Path(path).suffix in ('.py', '.sh')}, packages=packages,
                    arguments={key: value for key, value in settings.items() if key != 'output'})
    manifest['run_fingerprint'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'run_fingerprint': manifest['run_fingerprint'],
                      'run_manifest': str(output / 'run_manifest.json')}), flush=True)
    return manifest['run_fingerprint']
