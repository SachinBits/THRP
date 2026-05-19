# THRP — Two-Stage Hierarchical Recognition Pipeline

Comprehensive README with setup, run and reproduction commands for the THRP project. THRP implements a two-stage object recognition pipeline optimized for fine-grained aircraft classification: a generalist detector proposes bounding boxes and coarse superclasses, then specialist classifiers perform fine-grained subclass recognition on cropped ROIs.

Contents
- Project summary
- Requirements
- Setup and installation
- Model weights and dataset preparation
- Full command reference (inference, training, evaluation, embedding extraction, baselines)
- Project structure
- Reproducibility and experiment logging
- Troubleshooting
- License and contact

Project summary
THRP is designed to improve fine-grained aircraft recognition by dividing the task into detection (generalist) and subclass classification (specialists). This modular design reduces per-model complexity, enables targeted data augmentation and class-balanced training for difficult subclasses, and provides embedding extraction for fast baselines and diagnostics.

Requirements
- Python 3.10 or newer
- pip
- Optional GPU with CUDA 11.8+ for accelerated training or inference, or Apple MPS on macOS
- Recommended: 16 GB RAM, 50+ GB disk for datasets and checkpoints
- Install requirements with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Model weights and dataset preparation
- Models: pre-trained generalist and specialist weights are required for inference. Place weights in `models/` or download from project releases.
- Datasets: specialist datasets live under `yolo_specialist_datasets/` (one folder per superclass with `images/` and `labels/` in YOLO format). The generalist dataset is under `yolo_generalist_dataset/`.
- If you have large model files or full datasets, keep them off the git repository and use external storage or release assets. The repository `.gitignore` excludes common large artifacts.

Commands — Inference and utilities

- Single-image two-stage inference (annotated output saved):

```bash
python3 scripts/predict.py --image path/to/image.jpg --visualize
```

- Run inference and show all candidate detections (debug):

```bash
python3 scripts/predict.py --image path/to/image.jpg --all-detections
```

- Run the auto decision-tree predictor (embedding → tree):

```bash
python3 scripts/predict_auto_decision_tree.py --image path/to/image.jpg
```

- Batch inference on a folder:

```bash
python3 scripts/predict.py --input-folder path/to/images --out-dir outputs/predictions
```

Commands — Training

- Train a specialist (example — Fighter):

```bash
python3 scripts/finetune_specialists.py \
  --dataset yolo_specialist_datasets \
  --superclasses fighter \
  --out-models models \
  --out-results results/baselines/cnn \
  --epochs-head 3 \
  --epochs-finetune 2 \
  --batch-size 32 \
  --device auto
```

- Train the generalist detector:

```bash
python3 scripts/train_generalist_kaggle.py \
  --epochs 50 \
  --batch-size 32 \
  --imgsz 640 \
  --device auto
```

Commands — Embeddings & baselines

- Extract embeddings for specialist datasets:

```bash
python3 scripts/extract_embeddings.py \
  --dataset yolo_specialist_datasets \
  --out results/embeddings \
  --batch-size 32 \
  --device auto
```

- Train simple baselines (decision tree / retrain with embeddings):

```bash
python3 scripts/run_decision_tree_baseline.py \
  --dataset yolo_specialist_datasets \
  --out results/baselines \
  --superclasses fighter

python3 scripts/retrain_with_embeddings.py --emb results/embeddings --out results/baselines
```

Commands — Evaluation & reporting

- Run end-to-end ensemble evaluation and collect metrics:

```bash
python3 scripts/eval_ensemble_fast.py
```

- Generate comprehensive reports (aggregates and per-class summaries):

```bash
./generate_report.sh
```

Commands — Frontend (preview)

- Run the streamlit frontend locally:

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

Project structure (high-level)

```
THRP/
├─ scripts/                      # Inference, training, eval, utilities
├─ models/                       # Place pretrained weights here (ignored by git)
├─ yolo_specialist_datasets/     # Specialist datasets (images + labels)
├─ yolo_generalist_dataset/      # Generalist dataset
├─ results/                      # Generated metrics, embeddings, baselines
├─ outputs/                      # Visualizations and report artifacts
├─ runs/                         # Training run outputs (checkpoints, tensorboard)
├─ requirements.txt
├─ README.md
└─ KNowledge BAse THRP           # Conceptual knowledge base (non-code)
```

Reproducibility and experiment logging
- Use deterministic seeds and record the full run config for each experiment. The training scripts accept seed/config args; ensure you record the `models/` checkpoint used and the dataset version.
- Save per-run artifacts: `runs/` (checkpoints), `results/` (metrics JSON), and `outputs/` (visualizations). Archive or upload large checkpoints separately.

Best practices
- Keep large checkpoints, datasets, and logs out of git. Use release assets or cloud storage for artifacts.
- Use the included `.gitignore` to avoid accidentally committing large files.
- For production deployment, consider model quantization, batching crops for specialists, and conditional routing thresholds to reduce latency.

Troubleshooting
- No models loaded: ensure `models/` contains `generalist_model.pt` and `*_specialist.pt` files.
- Low accuracy on specific class: inspect per-class confusion matrices in `results/` and increase augmentation or collect more examples for that class.
- OOM during training: reduce `--batch-size`, reduce `--imgsz`, or use CPU mode via `--device cpu`.

Contact & Contribution
- To contribute: open issues or PRs in the project repo; for model/data updates, attach small repro datasets or links to hosted artifacts.

License
- MIT License — see the LICENSE file in the repo.

Acknowledgements
- Uses YOLO/Ultralytics tooling and PyTorch. See `requirements.txt` for full dependency list.


## 👥 Authors
Sachin S.

## 🤝 Contributing
Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Submit a pull request

---

## 📞 Support
For issues, questions, or feedback:
- Open an issue on GitHub
- Check existing documentation in `/docs`
- Review GITHUB_UPLOAD_GUIDE.md for detailed schema

---

*Last Updated: May 15, 2026*
