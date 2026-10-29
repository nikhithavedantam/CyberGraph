"""
CyberGraph - interactive Streamlit dashboard.

Run:   python -m streamlit run dashboard.py
"""
import io

import pandas as pd
import streamlit as st

from src.generate_data import generate
from src.pipeline import run
from src.visualize import (plot_similarity_graph, plot_bipartite, plot_degree_distribution,
                           plot_top_risk, plot_timeline, plot_confusion, plot_ablation, COLORS)

st.set_page_config(page_title="CyberGraph", page_icon="🛡️", layout="wide")
st.title("🛡️ CyberGraph: Intelligent Web Threat Analytics")
st.caption("Graph and Web Analytics course project - explore web-log threats as graphs")

# ------------------------------------------------------------------ sidebar
st.sidebar.header("1. Data")
source = st.sidebar.radio("Log source", ["Generate sample data", "Upload CSV"])
n_normal, seed, upload = 300, 42, None
if source == "Generate sample data":
    n_normal = st.sidebar.slider("Normal users", 100, 800, 300, 50)
    seed = st.sidebar.number_input("Random seed", 0, 9999, 42)
else:
    upload = st.sidebar.file_uploader("CSV: timestamp,ip,method,url,status,user_agent[,label]", type="csv")

st.sidebar.header("2. Model settings")
sim_thr = st.sidebar.slider("Graph similarity threshold", 0.50, 0.99, 0.90, 0.01,
                            help="Cosine similarity needed to link two IPs in the similarity graph")
risk_thr = st.sidebar.slider("Risk threshold", 0.20, 0.90, 0.50, 0.05,
                             help="IPs above this risk score are flagged as threats")
contam = st.sidebar.slider("Expected anomaly share", 0.01, 0.40, 0.15, 0.01,
                           help="Isolation Forest contamination")


@st.cache_data(show_spinner="Loading logs ...")
def load(source, n_normal, seed, upload_bytes):
    if source == "Generate sample data":
        return generate(n_normal=n_normal, seed=int(seed))
    return pd.read_csv(io.BytesIO(upload_bytes), parse_dates=["timestamp"])


@st.cache_data(show_spinner="Building graphs and detecting threats ...")
def analyse(df, sim_thr, risk_thr, contam):
    return run(df, sim_thr, risk_thr, contam)


def fig_png(fn, *args):
    """Render one of the src.visualize plots into memory and return PNG bytes."""
    buf = io.BytesIO()
    fn(*args, buf)
    return buf.getvalue()


if source == "Upload CSV" and upload is None:
    st.info("Upload a log CSV in the sidebar, or switch to *Generate sample data*.")
    st.stop()

try:
    df = load(source, n_normal, seed, upload.getvalue() if upload else None)
    R = analyse(df, sim_thr, risk_thr, contam)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not process the data: {e}")
    st.stop()

feat, urls = R["feat"], R["urls"]

# ------------------------------------------------------------------ KPIs
c = st.columns(5)
c[0].metric("Requests", f"{len(df):,}")
c[1].metric("Unique IPs", feat.shape[0])
c[2].metric("Threat IPs", int(feat.is_threat.sum()))
c[3].metric("Graph edges", R["G_bip"].number_of_edges())
c[4].metric("F1 score", f"{R['metrics']['f1']:.3f}" if R["has_truth"] else "n/a (no labels)")

tabs = st.tabs(["📈 Overview", "🕸️ Graphs", "🚨 Threat table", "🔎 IP drill-down",
                "🎯 Targeted URLs", "✅ Evaluation"])

# ------------------------------------------------------------------ overview
with tabs[0]:
    l, r = st.columns([3, 2])
    l.image(fig_png(plot_timeline, df, feat), caption="Traffic timeline by detected type")
    counts = feat.predicted_type.value_counts().rename_axis("type").reset_index(name="IPs")
    r.subheader("Detected types")
    r.dataframe(counts, hide_index=True, use_container_width=True)
    r.image(fig_png(plot_top_risk, feat), caption="Top 15 riskiest IPs")

# ------------------------------------------------------------------ graphs
with tabs[1]:
    st.write("**Similarity graph** - IPs with near-identical URL profiles are linked. "
             "Dense clusters are botnets / scanners. Lower the threshold in the sidebar to link more IPs.")
    if R["G_sim"].number_of_edges() == 0:
        st.warning("No IPs pass the similarity threshold. Lower it in the sidebar.")
    else:
        st.image(fig_png(plot_similarity_graph, R["G_sim"], feat))
    st.write("**IP ↔ URL bipartite graph** - top threats plus a sample of normal users.")
    st.image(fig_png(plot_bipartite, R["G_bip"], feat))
    st.image(fig_png(plot_degree_distribution, R["G_bip"]), width=520)

# ------------------------------------------------------------------ threat table
with tabs[2]:
    types = sorted(feat.predicted_type.unique())
    pick = st.multiselect("Show types", types, default=[t for t in types if t != "normal"])
    show = feat[feat.predicted_type.isin(pick)].sort_values("risk_score", ascending=False)
    cols = ["predicted_type", "risk_score", "anomaly_score", "rule_score", "n_requests", "unique_urls",
            "err_rate", "auth_fail_rate", "payload_ratio", "peak_req_per_min", "community_id", "community_size"]
    st.dataframe(show[cols].round(3), use_container_width=True)
    st.download_button("⬇ Download as CSV", show[cols].round(4).to_csv().encode(), "threat_ips.csv", "text/csv")

# ------------------------------------------------------------------ drill-down
with tabs[3]:
    ranked = feat.sort_values("risk_score", ascending=False)
    ip = st.selectbox("Choose an IP (sorted by risk)", ranked.index)
    row = feat.loc[ip]
    a, b, c3 = st.columns(3)
    a.metric("Predicted type", row.predicted_type)
    b.metric("Risk score", f"{row.risk_score:.3f}")
    c3.metric("Community size", int(row.community_size))
    left, right = st.columns(2)
    left.write("**Top URLs requested**")
    left.dataframe(df[df.ip == ip].url.value_counts().head(10).rename("requests"), use_container_width=True)
    left.write("**Status codes**")
    left.bar_chart(df[df.ip == ip].status.value_counts().sort_index())
    right.write("**Features**")
    right.dataframe(row.drop(["predicted_type"]).astype(float).round(4).rename("value"), use_container_width=True)
    right.write("**Sample of raw requests**")
    right.dataframe(df[df.ip == ip].head(15)[["timestamp", "method", "url", "status", "user_agent"]],
                    hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ URLs
with tabs[4]:
    st.write("Centrality of URL nodes in the IP-URL graph (PageRank, HITS authority, betweenness).")
    st.dataframe(urls.round(4), use_container_width=True, hide_index=True)
    st.bar_chart(urls.set_index("url")[["requests", "error_requests"]])

# ------------------------------------------------------------------ evaluation
with tabs[5]:
    if not R["has_truth"]:
        st.info("Evaluation needs a `label` column (normal / scanner / bruteforce / sqli / ddos) in the CSV.")
    else:
        m = R["metrics"]
        k = st.columns(3)
        k[0].metric("Precision", f"{m['precision']:.3f}")
        k[1].metric("Recall", f"{m['recall']:.3f}")
        k[2].metric("F1", f"{m['f1']:.3f}")
        l, r = st.columns(2)
        l.image(fig_png(plot_confusion, m["confusion"]))
        r.image(fig_png(plot_ablation, R["ablation"]))
        st.dataframe(R["ablation"].round(3), use_container_width=True)
        st.caption("Ablation: removing graph features (community size, similarity degree, clustering) "
                   "lets the distributed DDoS botnet go undetected. Data is synthetic; thresholds were tuned on it.")
