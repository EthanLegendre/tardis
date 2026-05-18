import warnings
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(page_title="TARDIS", page_icon="🚄", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_dataset.csv")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

@st.cache_resource
def load_model(df):
    for path in ("model.joblib", "model.pkl"):
        try:
            return joblib.load(path), None
        except Exception:
            pass
    cols = ["Number of scheduled trains", "Month", "Departure station",
            "Arrival station", "Average journey time", "Service"]
    cats = ["Departure station", "Month", "Arrival station", "Service"]
    avail = [c for c in cols if c in df.columns]
    X = pd.get_dummies(df[avail], columns=[c for c in cats if c in avail], drop_first=True)
    y = df["Average delay of all trains at arrival"]
    mask = y.notna() & X.notna().all(axis=1)
    m = RandomForestRegressor(n_estimators=100, random_state=12)
    m.fit(X[mask], y[mask])
    return m, X.columns.tolist()


df = load_data()
model, fallback_cols = load_model(df)

st.sidebar.title("🚄 TARDIS")
st.sidebar.markdown("SNCF Delay Analytics")
st.sidebar.divider()

departures = sorted(df["Departure station"].dropna().unique())
arrivals = sorted(df["Arrival station"].dropna().unique())
services = sorted(df["Service"].dropna().unique()) if "Service" in df.columns else []

sel_dep = st.sidebar.multiselect("Departure station", departures, placeholder="All")
sel_arr = st.sidebar.multiselect("Arrival station", arrivals, placeholder="All")
sel_svc = st.sidebar.multiselect("Service", services, placeholder="All")

fdf = df.copy()
if sel_dep:
    fdf = fdf[fdf["Departure station"].isin(sel_dep)]
if sel_arr:
    fdf = fdf[fdf["Arrival station"].isin(sel_arr)]
if sel_svc:
    fdf = fdf[fdf["Service"].isin(sel_svc)]

st.title("TARDIS — SNCF Delay Dashboard")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg arrival delay", f"{fdf['Average delay of all trains at arrival'].mean():.1f} min")
c2.metric("Total scheduled trains", f"{int(fdf['Number of scheduled trains'].sum()):,}")
if "Rate delay train" in fdf.columns:
    c3.metric("Punctuality rate", f"{100 - fdf['Rate delay train'].mean():.1f}%")
if "Rate Cancel train" in fdf.columns:
    c4.metric("Avg cancellation rate", f"{fdf['Rate Cancel train'].mean():.1f}%")

st.divider()

tab1, tab2 = st.tabs(["📊 Overview", "🗺️ Stations"])

with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Delay distribution")
        fig, ax = plt.subplots()
        data = fdf["Average delay of all trains at arrival"].dropna()
        ax.hist(data, bins=35, color="#e63946", edgecolor="white", linewidth=0.3)
        ax.axvline(data.mean(), color="orange", linestyle="--", label=f"Mean: {data.mean():.1f} min")
        ax.set_xlabel("Delay (minutes)")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.subheader("Delay by month")
        if "Month" in fdf.columns:
            month_order = ["January", "February", "March", "April", "May", "June",
                           "July", "August", "September", "October", "November", "December"]
            m_avg = (fdf.groupby("Month")["Average delay of all trains at arrival"]
                     .mean()
                     .reindex([m for m in month_order if m in fdf["Month"].values]))
            fig, ax = plt.subplots()
            ax.bar(m_avg.index, m_avg.values, color="#e63946")
            ax.set_ylabel("Avg delay (min)")
            plt.xticks(rotation=40, ha="right", fontsize=8)
            st.pyplot(fig)
            plt.close()

    st.subheader("Correlation heatmap")
    num_cols = [c for c in [
        "Average delay of all trains at arrival",
        "Number of scheduled trains",
        "Number of trains delayed at arrival",
        "Rate delay train",
        "Rate Cancel train",
        "Average journey time",
    ] if c in fdf.columns]
    if len(num_cols) >= 3:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.heatmap(fdf[num_cols].corr(), annot=True, fmt=".2f",
                    cmap="coolwarm", ax=ax, linewidths=0.5)
        plt.xticks(rotation=25, ha="right", fontsize=8)
        st.pyplot(fig)
        plt.close()

with tab2:
    n = st.slider("Stations to show", 5, 25, 10)
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Top departure stations")
        top_dep = (fdf.groupby("Departure station")["Average delay of all trains at arrival"]
                   .mean().sort_values(ascending=False).head(n))
        fig, ax = plt.subplots(figsize=(6, max(3, n * 0.35)))
        ax.barh(top_dep.index[::-1], top_dep.values[::-1], color="#e63946")
        ax.set_xlabel("Avg delay (min)")
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.subheader("Top arrival stations")
        top_arr = (fdf.groupby("Arrival station")["Average delay of all trains at arrival"]
                   .mean().sort_values(ascending=False).head(n))
        fig, ax = plt.subplots(figsize=(6, max(3, n * 0.35)))
        ax.barh(top_arr.index[::-1], top_arr.values[::-1], color="#457b9d")
        ax.set_xlabel("Avg delay (min)")
        st.pyplot(fig)
        plt.close()
