# 🐾 End-to-End Cat vs. Dog Image Classification Pipeline

An automated, production-grade Deep Learning system that classifies images of cats and dogs. This project transitions an image classification problem from an offline Jupyter notebook script into a fully containerized, microservice-based web architecture with automated CI/CD checks, remote cloud storage auditing, and an interactive user dashboard.

🔗 **Live Frontend Dashboard:** [catvsdog-psr.streamlit.app](https://catvsdog-psr.streamlit.app/)
🔗 **Production API Core Engine:** [catvsdog-rfzr.onrender.com](https://catvsdog-rfzr.onrender.com)

---

## 🏗️ System Architecture & Data Flow

The system is engineered using a decoupled microservice architecture to isolate the user interface, heavy model inference, and persistence logging layers.

```text
[ Streamlit Frontend ] ──(Public Internet HTTPS)──> [ FastAPI Core (Render Linux Container) ]
        │                                                      │
        ▼                                                      ▼
(Renders UI Components)                             ┌──────────────────────┐
                                                      │  • Custom PyTorch    │
                                                      │    CNN Inference     │
                                                      │  • Supabase Audit Log│
                                                      └──────────────────────┘
```

1. **Frontend Layer** — A clean, minimalist UI built with Streamlit accepts user drag-and-drop images and streams them across the public network.
2. **Compute & Inference Layer** — A FastAPI backend wrapped in an isolated Docker Linux container processes incoming image tensors through a pre-trained PyTorch Convolutional Neural Network (CNN).
3. **Storage & Audit Layer** — Every image evaluated by the production API is duplicated as a byte stream and archived asynchronously into a Supabase Storage Bucket for system telemetry and data-drift tracking.

---

## 🛠️ Technology Stack & Tooling

| Category | Tools |
|---|---|
| **Core AI / Deep Learning** | Python, PyTorch (`torch`, `torchvision`), Pillow (PIL) |
| **Backend Framework** | FastAPI, Uvicorn, Python-Multipart |
| **Infrastructure & Virtualization** | Docker |
| **Database & Cloud Storage** | Supabase (Object Storage) |
| **Automation & CI/CD** | GitHub Actions (YAML Syntax Verification & Isolated Docker Layer Compilation) |
| **Hosting Platforms** | Render (API Instance Engine), Streamlit Community Cloud (User Dashboard) |

---

## ⚡ Key Engineering & MLOps Implementations

### 📦 1. Production Containerization (Docker)
The backend service is completely isolated within an optimized Linux image using multi-command pipeline layers. It standardizes system paths, manages dependencies seamlessly, upgrades build engines (`pip`), and includes system compilation utilities (`build-essential`, `g++`, `gfortran`, `libopenblas-dev`) to ensure zero-dependency compilation on any remote server.

### 🛡️ 2. Secure Infrastructure Configuration
Utilizes strict separation of code and configuration. Production secrets, keys, and endpoint targets are stored inside protected system environment variables (`SUPABASE_URL`, `SUPABASE_KEY`, `RENDER_API_URL`). All credential structures are strictly isolated using local `.env` setups, completely tracked and protected using a `.gitignore` guardrail.

### ⚙️ 3. Continuous Integration Assembly Line (CI/CD)
Implemented a live, automated quality inspector inside GitHub Actions (`.github/workflows/ci.yml`). Every codebase update automatically triggers a virtual Linux cluster to execute two critical operational validation jobs:

- **`validate-python`** — Installs an isolated runtime environment and executes `flake8` to parse scripts for syntax-breaking bugs, trailing errors, and unhandled variables.
- **`validate-docker`** — Sets up remote Docker Buildx engines and compiles the `Dockerfile` to guarantee it builds cleanly before allowing deployment.

---

## 📂 Project Structure

```text
cat_dog_project/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI Automation script
├── src/
│   ├── dataset.py             # Custom Torchvision tensor image transformations
│   └── model.py               # Deep Learning CNN Architecture definition
├── weights/
│   └── model.pth              # Saved PyTorch model artifact (ignored by git)
├── app.py                     # FastAPI Asynchronous REST Endpoint service
├── dashboard.py                # Minimalist Streamlit client UI application
├── Dockerfile                  # Multi-layer Docker system environment recipe
├── requirements.txt            # Verified Python runtime library manifest
└── .gitignore                  # Exclusion configuration for heavy/secret tracking
```

---

## 🧠 Core Engineering Principles Learned

- **Decoupled Architecture** — Learned why separating frontends from backend processing engines ensures that spikes in user traffic never choke model execution performance.
- **Immutability via Containers** — Solved the classic "it worked on my machine" problem by enforcing exact hardware/software specifications inside a portable Docker image.
- **Defensive Secret Management** — Mastered token boundaries, ensuring enterprise cloud integration credentials never accidentally bleed into open source public codebases.
- **Automated Code Guardrails** — Embraced Continuous Integration principles, leveraging cloud runners to verify build stability before features reach end users.

---

## 🔗 Links

- **Frontend Dashboard:** [catvsdog-psr.streamlit.app](https://catvsdog-psr.streamlit.app/)
- **API Core Engine:** [catvsdog-rfzr.onrender.com](https://catvsdog-rfzr.onrender.com)

---

## 👤 Author

**Devprakash Singh Rawat**
