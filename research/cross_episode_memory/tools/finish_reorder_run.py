"""Package videos only after a specific full-run process exits.

Uses a Linux process-exit notification, without polling simulation progress.
Writes completion.json for the user; this does not send chat notifications.
"""
import argparse
import ctypes
import errno
import platform
import json
import os
from pathlib import Path
import select
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    folder = args.folder.resolve()
    repo = Path(__file__).resolve().parents[3]
    if not folder.is_relative_to(repo):
        parser.error('Output must remain inside molmospaces')
    try:
        if hasattr(os, 'pidfd_open'):
            fd = os.pidfd_open(args.pid)
        else:
            # The project's Python predates os.pidfd_open; Linux still exposes it.
            if platform.machine() not in ('x86_64', 'aarch64'):
                raise RuntimeError('Unsupported pidfd syscall architecture')
            libc = ctypes.CDLL(None, use_errno=True)
            libc.syscall.restype = ctypes.c_long
            fd = int(libc.syscall(434, args.pid, 0))
            if fd < 0:
                code = ctypes.get_errno()
                if code == errno.ESRCH:
                    raise ProcessLookupError(args.pid)
                raise OSError(code, os.strerror(code))
    except ProcessLookupError:
        fd = None
    if fd is not None:
        try:
            while not select.select([fd], [], [], 60)[0]:
                pass
        finally:
            os.close(fd)
    result = {'physics_success': False, 'video_packaged': False}
    try:
        report = json.loads((folder / 'report.json').read_text())
        result['physics_success'] = bool(report.get('success'))
        result['error'] = report.get('error')
        if result['physics_success']:
            if report.get('video_status') != 'complete':
                raise RuntimeError('Full-task video rendering did not complete')
            events = report['chain_events']
            result['changes'] = sum(e['kind'] == 'intervention' for e in events)
            result['transfers'] = sum(e['kind'] == 'work' for e in events)
            result['restore_request'] = next(e for e in events if e['kind'] == 'restore_request')
            for name in ('render_reorder_change.py', 'package_reorder_video.py'):
                subprocess.run([sys.executable, str(Path(__file__).with_name(name)), str(folder)],
                               cwd=repo, check=True)
            result['video_packaged'] = True
            result['video'] = str(folder / 'dynamic_revisit_restore.mp4')
    except Exception as exc:
        result['packaging_error'] = str(exc)
    temporary = folder / 'completion.json.tmp'
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(folder / 'completion.json')
    print(json.dumps(result), flush=True)
    return 0 if result['physics_success'] and result['video_packaged'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
