# DTMKFC

> 🧠 *Dynamic Topology-aware Multi-view Knowledge-guided Semi-supervised Clustering*  
> Official implementation of our clustering method designed for noisy, multi-view, semi-supervised scenarios.  
> 📄 Companion code to our paper (2025). Please see citation below.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Issues](https://img.shields.io/github/issues/yourname/dtmkfc)](https://github.com/yourname/dtmkfc/issues)

---

## 🌟 Key Features

- ✅ Multi-view clustering with view fusion and constraint propagation  
- 🔁 Dynamic graph refinement across iterations  
- 📊 Built-in metrics: Accuracy, NMI, Purity  
- ⚙️ Easily customizable: plug in your own graph, views, or fusion strategy  

---

## 🗂️ Project Structure

```
DTMKFC/
├── dtmkfc.py              # Core algorithm (DTMKFC class)
├── utils.py               # Graph builders, constraint samplers, metric functions
├── datasets.py            # Synthetic multi-view dataset generator
├── run_experiment.py      # Quick demo script
├── requirements.txt       # Dependencies
├── LICENSE
└── README.md              # You’re here
```

---

## 🚀 Quick Start

### 🔧 Installation

```bash
git clone https://github.com/yourname/dtmkfc.git
cd dtmkfc
pip install -r requirements.txt
```

### ▶️ Run Demo

```bash
python run_experiment.py
```

This runs DTMKFC on a synthetic 2-view noisy digits dataset.  
Modify `run_experiment.py` to plug in your own data or adjust parameters.

---

## 📈 Output Example

```
[INFO] Clustering completed.
[INFO] ACC    = 0.9123
[INFO] NMI    = 0.8456
[INFO] Purity = 0.9347
```

---

```

---

## 🤝 Contributing

We welcome issues, feature requests, and pull requests.  
To contribute, please fork the repo and open a PR with clear description of changes.


---

## 📝 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
