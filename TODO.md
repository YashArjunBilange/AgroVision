# Agrovision Render Deployment TODO

## Previous Steps (Done):
- [x] Step 1: Create Procfile with Render start command for Streamlit
- [x] Step 2: Create runtime.txt specifying Python 3.11.9 (stable for torch/ultralytics)
- [x] Step 3: Verify requirements.txt (opencv-python-headless already used, good for headless Render env; torch CPU-only fine on Render)
- [x] Handle Render build error: Update requirements.txt for Python 3.14 compatibility

## Remaining Steps:
- [ ] Step 4: Provide Render deployment instructions
- [ ] Step 5: Mark complete

**Notes:** 
- Updated requirements.txt uses CPU torch index + loose versions (works on Render Python 3.14).
- opencv-python-headless handles no GUI.
- Torch CPU fine for free tier inference.
- best.pt deploys as static.
- Push changes & trigger Render rebuild.
