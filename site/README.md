# Hero's Journey — benchmark site

A single-page, dependency-free site that (1) lets readers explore how each task
works interactively and (2) shows the model leaderboard (Table 6 of the paper).

## Files

- `index.html` — the page: Overview (landing) + Task Explorer + Leaderboard. No build step.
- `data.js` — generated data: `window.HJ_TASKS` (example episodes for all 8
  tasks × {semantic, nonce}) and `window.HJ_LEADERBOARD` (Table 6 numbers).
- `hj_icon.png`, `hj_fig.png` — the mascot and Figure 1, used on the Overview page.
  Regenerated from the source PDFs with:
  ```bash
  pdftocairo -png -r 300 -transp hj_icon.pdf hj_icon && mv hj_icon-1.png hj_icon.png
  pdftocairo -png -r 200        hj.pdf      hj_fig  && mv hj_fig-1.png  hj_fig.png
  ```

## Viewing it

Just open `index.html` in a browser — it loads `data.js` via a `<script>` tag,
so it works from `file://` (double-click) with no server, and unchanged on
GitHub Pages.

To serve locally:

```bash
python -m http.server -d site 8000   # then open http://localhost:8000
```

## Regenerating the data

`data.js` is produced from the live task pipeline:

```bash
python experiments/build_site_data.py
```

Re-run this whenever task definitions, rules, or lexicons change. The leaderboard
numbers are transcribed in that script (`LEADERBOARD_ROWS`) from the paper's
Appendix D, Table 6 — edit there if the numbers change.
