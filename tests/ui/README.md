# UI regression checks

The app itself needs no Node build or browser packages. These optional tests
use Node 18+ and jsdom 26.1 (compatible with Node 18+), plus the project's
Python environment. Set `PYTHON` to select a different Python executable.

```sh
npm ci --prefix tests/ui
npm test --prefix tests/ui
```

The DOM suite starts an isolated Python HTTP server on an ephemeral loopback
port and stops it after testing. It checks presets, rule previews, language
switching, both armies, undo, refresh, request recovery, promotions, keyboard
play and old replay formats. The parser and coordinate tests run without a
server.

```sh
python -m pytest tests/test_web_server.py tests/test_ab_web.py -q
```

jsdom does not perform layout or paint. Also inspect `/play/` and `/replay/`
in a real browser at desktop, tablet and phone sizes, including keyboard
focus, dialog placement and the board's click targets after flipping.
