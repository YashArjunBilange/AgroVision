# Render Deploy - Deploy Phase Fix

**Status:** BUILD SUCCESS ✅ | Deploy fails on 'auto' start.

**Fix Procfile:**
web: streamlit run app.py --server.port ${PORT:=8501} --server.address 0.0.0.0 --server.headless true

**Critical: Render Dashboard Settings:**
1. Service → Settings
2. Build Command: `pip install -r requirements.txt`
3. Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
4. Save → Manual Deploy.

This forces commands, detects port.

Push Procfile update, set manual commands → LIVE!
