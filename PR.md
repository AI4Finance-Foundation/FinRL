# Clarify survivorship bias in Dow 30 backtests

## Summary

This change warns users that `DOW_30_TICKER` is a static constituent snapshot, not a point-in-time Dow universe. Reusing it for historical periods introduces survivorship bias and makes results unsuitable to present as a historical Dow strategy.

## Changes

- Add a warning to the multiple-stock trading tutorial.
- Document the same limitation next to the 2014–2025 training example.

## Verification

- `git diff --check` passed.
- `pip wheel --no-deps . --wheel-dir /private/tmp/finrl-build` passed and produced `finrl-0.3.8-py3-none-any.whl`.
- `sphinx-build -b html docs/source /private/tmp/finrl-docs-build` exited successfully and rendered the warning. The full documentation build reports 55 warnings, including errors in unrelated existing pages; the changed tutorial page parsed successfully.

## How did AI help?

AI helped inspect issue #1458, trace the `DOW_30_TICKER` snapshot and the example training window, and draft the warning. I checked the current source files and dates, reviewed the diff, and ran the package and documentation builds listed above.
