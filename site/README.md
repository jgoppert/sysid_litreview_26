# Benchmark Site

This directory is a static GitHub Pages application for browsing benchmark results.

Run locally from the repository root:

```bash
python3 -m http.server 8000 --directory site
```

Then open `http://localhost:8000`.

Refresh the data bundle with:

```bash
./results.py web-data
```

The app reads JSON from `site/public/data/`. It does not run benchmark methods in the browser; benchmark execution remains Python-native and the site is only an interactive results viewer.
