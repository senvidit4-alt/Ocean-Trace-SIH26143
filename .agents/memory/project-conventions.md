# Project Conventions & Deployment Safety Protocols

## 🔒 Strict Live Deployment Isolation Rule (MANDATORY & CRITICAL)
- **Local Development Branch:** `main` (or feature branches). All daily code changes, tests, experiments, and edits MUST strictly remain on `main`.
- **Live Deployment Branch:** `production`.
- **NEVER touch or push to `production` automatically.** Under NO circumstances should any change be pushed/merged to `production` without an explicit, direct command from the user (e.g., "okay now add the updated files to the live deployment", "deploy to production", etc.).
- The live deployments on Vercel and Render track `production` ONLY. Local edits on `main` will NEVER trigger or affect live deployments.
