# CyberGraph: Intelligent Web Threat Analytics
Course project - **Graph and Web Analytics**

## 1. Problem
Web servers log millions of requests. Attacks such as scanning, brute force, SQL injection and
DDoS are usually hunted with per-IP thresholds, which miss **coordinated** attacks: a botnet of
40 IPs that each send only 40-80 requests looks harmless one-by-one.
CyberGraph models web traffic as **graphs** so that relationships between IPs, URLs and
user-agents expose threats that isolated statistics cannot.

## 2. Objectives
1. Convert web-server logs into graph structures.
2. Apply graph analytics (degree, PageRank, HITS, clustering, betweenness, Louvain communities).
3. Detect and classify threats with a hybrid unsupervised + rule-based model.
4. Prove the value of the graph approach through an ablation study.
5. Produce visualisations and an HTML report.

## 3. Method
| Stage | File | What it does |
|---|---|---|
| Data | `src/generate_data.py` | Synthetic logs: normal users, scanners, brute-forcers, SQLi, DDoS botnet, benign Googlebot crawlers (hard negatives). Ground-truth labels included. |
| Graphs | `src/graph_builder.py` | (a) IP-URL bipartite graph, (b) IP-UserAgent graph, (c) IP similarity graph (one-mode projection, cosine >= 0.90) |
| Analytics | `src/features.py` | 13 behavioural + 9 graph features per IP; URL centrality (PageRank, HITS authority, betweenness) |
| Detection | `src/detection.py` | Isolation Forest anomaly score + interpretable graph-aware rules -> risk score, attack type |
| Visuals | `src/visualize.py` | Similarity graph, bipartite graph, degree distribution, top-risk, timeline, confusion matrix, ablation |
| Report | `src/report.py` | Self-contained `outputs/report.html` |
| Dashboard | `dashboard.py`, `src/pipeline.py` | Interactive Streamlit app on the same pipeline |

Key idea: **community size** from Louvain on the similarity graph identifies botnets - many IPs with
near-identical URL profiles.

## 4. Results (default synthetic run, seed 42)
8,793 requests | 360 IPs | 3,281 IP-URL edges | 56 malicious IPs

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Isolation Forest only | 0.852 | 0.821 | 0.836 |
| Rules without graph features | 1.000 | 0.286 | 0.444 |
| **Full graph-aware model** | **1.000** | **1.000** | **1.000** |

Without graph features the DDoS botnet (40 of the 56 attackers) is missed - graph analytics is
what catches it.

**Honest limitation:** the data is synthetic and the rule thresholds were tuned on it, so the perfect
score will not carry over to real traffic. Use `--logs` with a real CSV and retune thresholds in
`src/detection.py`.

## 5. Run in VS Code
1. Install Python 3.9+ and the VS Code *Python* extension.
2. `File > Open Folder...` -> select the `CyberGraph` folder.
3. Open a terminal (`Ctrl + ~`) and run:
   ```bash
   python -m venv .venv
   # Windows:  .venv\Scripts\activate       macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   python main.py
   ```
   Or press **F5** (Run and Debug -> "CyberGraph: run full pipeline"), or open `main.py` and click the Run button.
4. Open `outputs/report.html` in a browser (right-click > *Open with Live Server*, or just double-click).

Use your own logs: `python main.py --logs my_logs.csv`
(columns: `timestamp, ip, method, url, status, user_agent` and optionally `label`).

## 5b. Interactive dashboard (Streamlit)
```bash
python -m pip install streamlit
python -m streamlit run dashboard.py
```
(On Windows use `python -m streamlit ...` because the `streamlit` command is often not on PATH.)
It opens at http://localhost:8501 with tabs: Overview, Graphs, Threat table (filter + CSV download),
IP drill-down, Targeted URLs, Evaluation. Sidebar sliders change the graph similarity threshold,
risk threshold and expected anomaly share live; you can also upload your own log CSV.

## 6. Outputs (`outputs/`)
`report.html`, `ip_threat_scores.csv`, `url_centrality.csv`, and figures `01`-`07`.

## 7. Future work
Temporal/dynamic graphs, Graph Neural Networks (GraphSAGE), real datasets (CIC-IDS2017, Apache/Nginx logs),
and streaming detection.
