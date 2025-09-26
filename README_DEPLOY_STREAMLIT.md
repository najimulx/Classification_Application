Streamlit Cloud deployment instructions

1) Ensure the repository contains a top-level entry file called `streamlit_app.py`.
   - This repo includes `streamlit_app.py` which calls `aeroreach.ui.app.main()`.

2) requirements.txt
   - The repo already has a `requirements.txt`. Make sure it lists `streamlit` and any other required packages.

3) Data file
   - This app expects `AeroReach Insights.csv` at the repository root. Confirm it's committed to Git.

4) Deploy on Streamlit Cloud
   - On https://share.streamlit.io click New app, connect your GitHub repo and select the branch.
   - App file path: `streamlit_app.py`.
   - Click Deploy. Streamlit Cloud will create the virtual environment and install packages from `requirements.txt`.

5) Common issues & tips
   - Large requirements or binary packages may need extra build time or fail on Streamlit Cloud. If a dependency fails to install, consider using a lighter alternative or pre-building wheels.
   - If your CSV is large, consider moving it to a cloud storage (S3, Git LFS) and load it at runtime.
   - For private data keep it out of the repo and load using secrets or environment variables.

6) Advanced: runtime memory/CPU
   - Streamlit Cloud provides limited resources for free apps. If your model trains at runtime, prefer loading a pre-trained model artifact instead of retraining on each request.
