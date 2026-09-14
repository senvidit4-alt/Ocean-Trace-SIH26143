# Ocean Trace Console — Frontend

Oil Spill Detection & Vessel Attribution console. This is the exact UI/UX built
during the design phase, exported from a single self-contained HTML file into
a normal, buildable frontend project so it can be version-controlled, run
locally, and wired up to a real backend.

## Tech stack

- **Plain HTML / CSS / JavaScript** — no framework (no React/Vue/etc). All
  rendering (map, charts, gauges, tables) is done with hand-written SVG/Canvas
  and DOM code.
- **Vite** — used only as a dev server + production bundler/minifier. There is
  no framework-specific tooling; Vite is swappable for any static file server
  if preferred.
- **Google Fonts** (Barlow, Barlow Condensed, Inter, JetBrains Mono) loaded via
  `<link>` tags in `index.html` — requires internet access at runtime, or vendor
  the font files locally if the deployment target is offline.
- No other third-party packages, charting libraries, or map libraries are used
  — the map, bathymetry basemap, and all charts are custom canvas/SVG code.

## Project structure

```
ocean-trace-frontend/
├── index.html        # Page shell: <head>, font links, full app markup, script tag
├── src/
│   ├── styles.css     # All CSS (extracted from the 3 <style> blocks)
│   └── app.js         # All application JS (extracted from the 4 <script> blocks)
├── package.json
├── vite.config.js
└── .gitignore
```

The HTML/CSS/JS content is unchanged from the working prototype — this export
only splits one big file into the three files above and adds npm tooling
around it. No visual or behavioral changes were made.

## Running it locally

Requires Node.js 18+.

```bash
npm install
npm run dev
```

This starts a dev server (default: http://localhost:5173) with the console
running exactly as it did in the artifact preview — including the login gate,
all pages (Dashboard, Add Scene, Detection, Drift & Forecast, Attribution,
Analysis, History), and the built-in demo/mock data.

Production build:

```bash
npm run build      # outputs static files to dist/
npm run preview    # serve the production build locally to sanity-check it
```

`dist/` can be deployed to any static host (Netlify, Vercel, S3+CloudFront,
GitHub Pages, nginx, etc.) — it's plain HTML/CSS/JS, no server runtime needed
for the frontend itself.

## Connecting a real backend

The frontend currently runs on three bundled mock fixtures (a validated Gulf
of Mexico detection, an alternate-region detection, and a "no detection"
case) so it's fully demoable without a backend. All backend integration goes
through **one JS entry point** exposed on `window.OceanTrace`, defined at the
bottom of `src/app.js`:

```js
window.OceanTrace = {
  loadResult(detectionJson, opts),   // main hook: feed a new detection result into the UI
  applyResult(detectionJson),        // parse a result into internal state without redrawing/recentring
  centreOn(lat, lon, spanKm),        // pan/zoom the map to a coordinate
  frameCase(),                       // fit the map to the current case's bounds
  get result(),                      // current raw payload
  get case(),                        // current normalized case object used by the UI
  DEFAULT_VIEW,                      // default camera target
  samples: { gulfOfMexico, alternateRegion, noDetection } // the 3 bundled fixtures, for reference
};
```

To wire in a real backend:

1. Have your backend return JSON shaped like the fixtures in `samples`
   (`scene`, `detection.slick_polygon` as GeoJSON + `detection.centroid`,
   `drift.origin` / `drift.hindcast_path` / `drift.forcing`, and a `vessels[]`
   array with `lat`/`lon`/`cog`/`sog`/`evidence`).
2. After the page loads (or on a poll/websocket event), call:
   ```js
   fetch("/api/detections/latest")
     .then(r => r.json())
     .then(data => window.OceanTrace.loadResult(data));
   ```
3. The console will re-render the case banner, KPI strip, Detection page,
   Drift & Forecast page, and recentre the map automatically.
4. Look at `normaliseResult()` / `applyResult()` / `buildFleetFromResult()` in
   `src/app.js` if the real backend's schema differs and the mapping needs
   adjusting.

The login gate (`viewLogin`) is currently a UI-only mock — swap its submit
handler for a real auth call (e.g. `fetch("/api/login", ...)`) when the auth
backend is ready.

## Notes

- All ship/vessel data, MMSIs, and tracks in the bundled samples are synthetic
  demo data and do not describe real ships, except the Gulf of Mexico
  Sentinel-1 scene metadata (Aug 29 2021) which is a real reference scene used
  for calibration.
- The world basemap (coastlines) is derived from Natural Earth 1:50m data
  (via the `world-atlas` dataset) and embedded as a compact encoded string in
  `app.js` — no runtime fetch or external map tile service is used.
