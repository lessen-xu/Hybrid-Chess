"""Shared pytest fixtures.

Resets the module-level ``_active_variant`` (Python) and ``g_rule_flags``
(C++) between tests so that variant settings from one test cannot leak into
the next. This addresses audit issue C1 at the test layer; the production
code still uses the same globals.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_rule_state():
    """Reset Python and C++ rule-flag globals before and after each test."""
    import hybrid.core.rules as _rules
    try:
        from hybrid.cpp_engine import hybrid_cpp_engine as _cpp
        _has_cpp = True
    except ImportError:
        _has_cpp = False

    _rules._active_variant = None
    if _has_cpp:
        _cpp.set_rule_flags(_cpp.RuleFlags())

    yield

    _rules._active_variant = None
    if _has_cpp:
        _cpp.set_rule_flags(_cpp.RuleFlags())
