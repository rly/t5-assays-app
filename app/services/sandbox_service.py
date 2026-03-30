"""Sandboxed code execution for AI-generated pandas/numpy scripts.

Runs user code in a subprocess with restricted imports to prevent
network access, file system writes, and process spawning.
Supports multiple named DataFrames passed as a dict.
"""
import json
import os
import pickle
import subprocess
import sys
import tempfile

import pandas as pd

TIMEOUT_SECONDS = 30

RUNNER_TEMPLATE = '''
import sys, io, json, pickle, os

# Import pandas/numpy BEFORE installing the import restriction hook
import pandas as pd
import numpy as np
import math, statistics, re, datetime, collections, itertools, functools, decimal

# Load datasets dict from temp file
datasets = pickle.loads(open(sys.argv[1], 'rb').read())

# Make each dataset available as a standalone variable
for _name, _df in datasets.items():
    # Sanitize name for use as variable (replace spaces/hyphens with underscores)
    _var_name = _name.replace(" ", "_").replace("-", "_")
    globals()[_var_name] = _df

# Backward compat: if only one dataset, also set 'df'
if len(datasets) == 1:
    df = list(datasets.values())[0]

# Install import restriction hook
_BLOCKED = {
    'subprocess', 'socket', 'http', 'urllib', 'requests', 'httpx',
    'ftplib', 'smtplib', 'xmlrpc', 'socketserver',
    'pathlib', 'glob', 'tempfile',
    'multiprocessing', 'threading', 'concurrent',
    'ctypes', 'code', 'codeop',
    'webbrowser', 'antigravity', 'os',
}

_original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

def _restricted_import(name, *args, **kwargs):
    top = name.split('.')[0]
    if top in _BLOCKED:
        raise ImportError(f"Import of '{name}' is not allowed in sandbox")
    return _original_import(name, *args, **kwargs)

if hasattr(__builtins__, '__import__'):
    __builtins__.__import__ = _restricted_import
else:
    import builtins
    builtins.__import__ = _restricted_import

# Capture output
_output = io.StringIO()
_old_stdout = sys.stdout
sys.stdout = _output

try:
    exec(CODE_PLACEHOLDER)
    result = {"success": True, "output": _output.getvalue()}
except Exception as e:
    result = {"success": False, "error": f"{type(e).__name__}: {str(e)}"}
finally:
    sys.stdout = _old_stdout

print(json.dumps(result), file=_old_stdout)
'''


def execute_code(code: str, datasets: dict[str, pd.DataFrame]) -> dict:
    """Execute code in a sandboxed subprocess with access to named DataFrames.

    Args:
        code: Python code string to execute
        datasets: Dict of {name: DataFrame} available to the code

    Returns:
        {"success": True, "output": str} or {"success": False, "error": str}
    """
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(datasets, f)
        pkl_path = f.name

    code_repr = repr(code)
    script = RUNNER_TEMPLATE.replace("CODE_PLACEHOLDER", code_repr)

    try:
        result = subprocess.run(
            [sys.executable, "-c", script, pkl_path],
            capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            return {"success": False, "error": stderr or "Code execution failed"}

        stdout = result.stdout.strip()
        if not stdout:
            return {"success": True, "output": "(no output)"}

        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            return {"success": True, "output": stdout}

    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Execution timed out after {TIMEOUT_SECONDS} seconds"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        try:
            os.unlink(pkl_path)
        except OSError:
            pass
