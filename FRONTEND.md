# THRP Frontend

This project now includes a Streamlit frontend with five tabs:

- THRP
- Decision Tree
- K-Nearest Neighbors
- Naive Bayes
- Support Vector Machine

## Run

From the project root:

```bash
./run_frontend.sh
```

Or run directly:

```bash
./.venv/bin/python3 -m streamlit run streamlit_app.py
```

## What it does

- Upload an aircraft image in any tab.
- The app runs the corresponding project command.
- The predicted output is shown with confidence and the raw command output.
- The THRP tab also shows the generated visualization image.
