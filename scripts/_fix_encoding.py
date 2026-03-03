"""Ensure stdout/stderr use UTF-8 encoding on Windows.

Import this at the top of any script that may run with redirected output
(e.g. Start-Process -RedirectStandardOutput), where Windows defaults to
the system ANSI codepage (GBK on Chinese systems) and crashes on any
non-ASCII character like emoji or em-dash.

Usage: at the very top of a script, add:
    import scripts._fix_encoding  # noqa: F401
"""

import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
