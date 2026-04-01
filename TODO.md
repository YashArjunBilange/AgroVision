# Agrovision Render Deployment TODO

## Previous Steps (Done):
- [x] Step 1: Create Procfile with Render start command for Streamlit
- [x] Step 2: Create runtime.txt specifying Python 3.11.9 (stable for torch/ultralytics)
- [x] Step 3: Verify requirements.txt (opencv-python-headless good; torch CPU fine)
- [x] Fix torch version error: CPU index + loose versions
- [x] Fix Streamlit Python 3.14 wheel error: Add .python-version (3.12.7)

## Remaining Steps:
- [ ] Step 4: Push changes & redeploy on Render
- [ ] Step 5: Mark complete

**Notes:** 
- .python-version forces Python 3.12.7 (wheels for streamlit/torch).
- opencv-headless safe.
- Push: `git add . && git commit -m "Fix Python version for deps" && git push`
- Render auto/manual deploy.
