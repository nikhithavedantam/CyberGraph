"""
CyberGraph: Intelligent Web Threat Analytics
Run:  python main.py            (optionally:  python main.py --logs path/to/your.csv)
"""
import argparse
import warnings
from pathlib import Path

import pandas as pd

from src.generate_data import generate
from src.graph_builder import build_ip_url_graph, build_ip_agent_graph, build_ip_similarity_graph
from src.features import behaviour_features, graph_features, url_centrality
from src.detection import detect, evaluate, FEATURES
from src.visualize import (plot_similarity_graph, plot_bipartite, plot_degree_distribution,
                           plot_top_risk, plot_timeline, plot_confusion, plot_ablation)
from src.report import build_report

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
DATA, OUT = ROOT / "data", ROOT / "outputs"
GRAPH_COLS = ["degree_urls", "pagerank", "hub_score", "community_size", "sim_degree", "clustering", "agent_share"]


def main(logs=None):
    DATA.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)

    print("[1/6] Loading / generating web logs ...")
    if logs:
        df = pd.read_csv(logs, parse_dates=["timestamp"])
    else:
        df = generate()
        df.to_csv(DATA / "web_logs.csv", index=False)
    print(f"      {len(df):,} requests, {df.ip.nunique()} unique IPs")
    has_truth = "label" in df.columns
    truth = df.groupby("ip").label.first() if has_truth else None

    print("[2/6] Building graphs (IP-URL bipartite, IP-Agent, IP similarity) ...")
    G_bip, G_agent, G_sim = build_ip_url_graph(df), build_ip_agent_graph(df), build_ip_similarity_graph(df)
    print(f"      bipartite: {G_bip.number_of_nodes()} nodes / {G_bip.number_of_edges()} edges | "
          f"similarity: {G_sim.number_of_edges()} edges")

    print("[3/6] Computing graph analytics + behavioural features ...")
    beh = behaviour_features(df)
    gf = graph_features(G_bip, G_sim, G_agent, beh.index)
    feat_base = beh.join(gf)
    url_tbl = url_centrality(G_bip)

    print("[4/6] Detecting threats (Isolation Forest + graph-aware rules) ...")
    feat = detect(feat_base)
    print(feat.predicted_type.value_counts().to_string())

    community_tbl = (feat[feat.community_size > 1].groupby("community_id")
                     .agg(size=("community_size", "first"),
                          avg_risk=("risk_score", "mean"),
                          dominant_type=("predicted_type", lambda s: s.value_counts().index[0]),
                          avg_requests=("n_requests", "mean")).round(3).reset_index()
                     .sort_values("size", ascending=False).head(10))

    print("[5/6] Evaluating ...")
    if has_truth:
        metrics, y_true = evaluate(feat, truth)
        # ablation: (a) Isolation Forest only, (b) no graph features, (c) full model
        iso_only = feat.copy(); iso_only["is_threat"] = (iso_only.anomaly_score >= 0.5).astype(int)
        no_graph = feat_base.copy()
        no_graph["community_size"], no_graph["sim_degree"], no_graph["clustering"] = 1, 0, 0
        no_graph = detect(no_graph)
        rows = {}
        for name, f in {"IsolationForest only": iso_only, "Rules w/o graph": no_graph, "Full graph-aware": feat}.items():
            m, _ = evaluate(f, truth); rows[name] = {k: m[k] for k in ("precision", "recall", "f1")}
        ablation = pd.DataFrame(rows).T
        print(ablation.round(3).to_string())
        print(f"      FULL MODEL -> precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} f1={metrics['f1']:.3f}")
        if "predicted_type" in feat:
            print(pd.crosstab(truth.reindex(feat.index).rename("actual"), feat.predicted_type).to_string())
    else:
        metrics = dict(precision=float("nan"), recall=float("nan"), f1=float("nan"), confusion=[[0, 0], [0, 0]])
        ablation = pd.DataFrame()

    print("[6/6] Saving figures, tables and HTML report ...")
    feat.sort_values("risk_score", ascending=False).round(4).to_csv(OUT / "ip_threat_scores.csv")
    url_tbl.to_csv(OUT / "url_centrality.csv", index=False)
    plot_similarity_graph(G_sim, feat, OUT / "01_similarity_graph.png")
    plot_bipartite(G_bip, feat, OUT / "02_bipartite_graph.png")
    plot_degree_distribution(G_bip, OUT / "03_degree_distribution.png")
    plot_top_risk(feat, OUT / "04_top_risk.png")
    plot_timeline(df, feat, OUT / "05_timeline.png")
    if has_truth:
        plot_confusion(metrics["confusion"], OUT / "06_confusion.png")
        plot_ablation(ablation, OUT / "07_ablation.png")
        stats = dict(requests=len(df), ips=df.ip.nunique(), edges=G_bip.number_of_edges(),
                     threat_ips=int(feat.is_threat.sum()))
        build_report(OUT, stats, feat, url_tbl, ablation, community_tbl, metrics)
        print(f"\nDone. Open  {OUT / 'report.html'}  in your browser.")
    else:
        print("\nDone (no ground-truth labels -> report skipped, see outputs/*.png and CSVs).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", help="CSV with columns timestamp,ip,method,url,status,user_agent[,label]")
    main(ap.parse_args().logs)
