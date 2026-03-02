# Automated Sentiment Analysis ML Pipeline

**MLOps Proof-of-Concept** for AI Ops Manager applications

Fully automated training pipeline using GitHub Actions. Every code change triggers data loading → preprocessing → model training → evaluation → artifact creation.

![CI Pipeline](https://github.com/YOUR-USERNAME/aiops-sentiment-automation-poc/actions/workflows/train-model.yml/badge.svg)

### 🎮 Live Interactive Demo
Try the model right now!  
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Click%20Here-brightgreen?style=for-the-badge&logo=streamlit)]([https://YOUR-APP-NAME.onrender.com](https://aiops-sentiment-demo.onrender.com))

(Deployed on Render free tier – may take ~30s to wake up on first visit)

### Workflow Diagram
```mermaid
flowchart TD
    A["GitHub Push or Manual Trigger"] --> B["Checkout Code"]
    B --> C["Setup Python + Install deps"]
    C --> D["Run train.py"]
    D --> E["Load IMDb dataset (1k samples)"]
    E --> F["TF-IDF Vectorization"]
    F --> G["Train LogisticRegression"]
    G --> H["Evaluate & Generate Report"]
    H --> I["Save Model + Report"]
    I --> J["Upload Artifacts"]
