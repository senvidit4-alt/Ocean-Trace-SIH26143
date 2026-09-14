# 🚀 OceanTrace Deployment & Live Release Protocol

This guide explains how OceanTrace is deployed live to the cloud with **safe branch isolation**, so local changes NEVER accidentally affect the live website until you explicitly promote them.

---

## 🛡️ Safe Deployment Architecture (Branch-Based Isolation)

To prevent unverified local edits from breaking production, deployments are tied strictly to the **`production`** branch:

```
[ Local Dev / Changes ]
          │
          ▼
    `main` branch (GitHub)  ──► Staging & Teammate sync (Safe development)
          │
          │  🔒 (Requires explicit command: "add updated files to live deployment")
          ▼
 `production` branch (GitHub)
     │                     │
     ▼                     ▼
Vercel (Frontend)     Render / Railway (Backend API)
(https://...)         (https://...)
```

---

## 🌐 1. Deploy Frontend to Vercel (1-Time Setup)

1. Go to **[Vercel Dashboard](https://vercel.com/new)** and click **"Add New Project"** $\to$ **"Import Git Repository"**.
2. Select repository: `senvidit4-alt/Ocean-Trace-v2.0`.
3. Configure project settings:
   - **Framework Preset**: `Vite`
   - **Root Directory**: `ocean-trace-frontend`
   - **Production Branch**: Select `production` *(Important!)*
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
4. Add Environment Variable:
   - `VITE_API_URL`: `<your-live-backend-url>` (e.g. `https://oceantrace-api.onrender.com`)
5. Click **Deploy**.

---

## 🛰️ 2. Deploy Backend to Render (1-Time Setup)

### Option A: 1-Click Blueprint
1. In **[Render Dashboard](https://dashboard.render.com/)**, click **"New +"** $\to$ **"Blueprint"**.
2. Connect `senvidit4-alt/Ocean-Trace-v2.0` (it will auto-detect `render.yaml`).
3. Set branch to `production` and click **Apply**.

### Option B: Manual Web Service
1. Click **"New +"** $\to$ **"Web Service"**.
2. Select repo `senvidit4-alt/Ocean-Trace-v2.0` $\to$ Branch: `production`.
3. **Runtime**: `Python 3`
4. **Build Command**: `pip install -r requirements.txt`
5. **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Add Environment Variable:
   - `PYTHON_VERSION`: `3.11.9`
   - `CORS_ORIGINS`: `*` (or your Vercel frontend domain)
7. Click **Create Web Service**.

---

## ⚡ How to Push Updates to Live Deployment

Whenever you make changes and are ready to deploy to production:

Simply ask:
> **`"okay now add the updated files to the live deployment"`**

This automatically performs the following safe release routine:
```bash
git checkout production
git merge main --ff-only
git push origin production
git checkout main
```

Upon receiving the push to `production`, **Vercel** and **Render** will automatically trigger a clean build and update your live site seamlessly!
