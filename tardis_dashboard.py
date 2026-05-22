import warnings
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

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
    cols = [
        "Number of scheduled trains",
        "Month",
        "Departure station",
        "Arrival station",
        "Average journey time",
        "Service",
    ]
    cats = ["Departure station", "Month", "Arrival station", "Service"]
    avail = [c for c in cols if c in df.columns]
    X = pd.get_dummies(
        df[avail], columns=[c for c in cats if c in avail], drop_first=True
    )
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
c1.metric(
    "Avg arrival delay",
    f"{fdf['Average delay of all trains at arrival'].mean():.1f} min",
)
c2.metric("Total scheduled trains", f"{int(fdf['Number of scheduled trains'].sum()):,}")
if "Rate delay train" in fdf.columns:
    c3.metric("Punctuality rate", f"{100 - fdf['Rate delay train'].mean():.1f}%")
if "Rate Cancel train" in fdf.columns:
    c4.metric("Avg cancellation rate", f"{fdf['Rate Cancel train'].mean():.1f}%")

st.divider()

tab1, tab2, tab3 = st.tabs(["📊 Overview", "🗺️ Stations", "🤖 Predict"])

with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Average delay per month")
        if "Month" in fdf.columns:
            month_order = [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ]
            fdf_month = fdf.copy()
            fdf_month["Month"] = pd.Categorical(
                fdf_month["Month"], categories=month_order, ordered=True
            )
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(
                data=fdf_month,
                x="Month",
                y="Average delay of all trains at arrival",
                errorbar=None,
                color="red",
                ax=ax,
            )
            ax.set_xlabel("")
            ax.set_ylabel("Avg delay (min)")
            plt.xticks(rotation=35, ha="right", fontsize=7)
            st.pyplot(fig)
            plt.close()

    with col_b:
        st.subheader("Scheduled vs delayed trains")
        if "Number of trains delayed at arrival" in fdf.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.regplot(
                data=fdf,
                x="Number of scheduled trains",
                y="Number of trains delayed at arrival",
                color="red",
                scatter_kws={"s": 5, "alpha": 0.5},
                ax=ax,
            )
            ax.set_xlabel("Scheduled trains")
            ax.set_ylabel("Delayed trains")
            st.pyplot(fig)
            plt.close()

    st.subheader("Correlation heatmap")
    num_cols = [
        c
        for c in [
            "Average delay of all trains at arrival",
            "Number of scheduled trains",
            "Number of trains delayed at arrival",
            "Rate delay train",
            "Rate Cancel train",
            "Average journey time",
        ]
        if c in fdf.columns
    ]
    if len(num_cols) >= 3:
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.heatmap(
            fdf[num_cols].corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax,
            linewidths=0.5,
        )
        plt.xticks(rotation=25, ha="right", fontsize=8)
        st.pyplot(fig)
        plt.close()

with tab2:
    n = st.slider("Stations to show", 5, 25, 10)
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Top departure stations")
        top_dep = (
            fdf.groupby("Departure station")["Average delay of all trains at arrival"]
            .mean()
            .sort_values(ascending=False)
            .head(n)
        )
        fig, ax = plt.subplots(figsize=(6, max(3, n * 0.35)))
        ax.barh(top_dep.index[::-1], top_dep.values[::-1], color="#e63946")
        ax.set_xlabel("Avg delay (min)")
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.subheader("Top arrival stations")
        top_arr = (
            fdf.groupby("Arrival station")["Average delay of all trains at arrival"]
            .mean()
            .sort_values(ascending=False)
            .head(n)
        )
        fig, ax = plt.subplots(figsize=(6, max(3, n * 0.35)))
        ax.barh(top_arr.index[::-1], top_arr.values[::-1], color="#457b9d")
        ax.set_xlabel("Avg delay (min)")
        st.pyplot(fig)
        plt.close()

with tab3:
    st.subheader("Predict arrival delay")

    col_a, col_b = st.columns(2)
    with col_a:
        p_dep = st.selectbox("Departure station", departures)
        p_arr = st.selectbox("Arrival station", arrivals)
        p_svc = st.selectbox("Service", services if services else ["TGV"])
    with col_b:
        months = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        p_month = st.selectbox("Month", months)
        days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        p_day = st.selectbox("Day of the week", days)

    if st.button("Predict", type="primary"):
        if model is None:
            st.error(
                "No model file found. Please add model.joblib to the project folder."
            )
        else:
            try:
                row = pd.DataFrame(
                    [
                        {
                            "Month": p_month,
                            "Departure station": p_dep,
                            "Arrival station": p_arr,
                            "Service": p_svc,
                            "Day": p_day,
                        }
                    ]
                )
                cats = [
                    "Departure station",
                    "Month",
                    "Arrival station",
                    "Service",
                    "Day",
                ]
                row_enc = pd.get_dummies(row, columns=cats, drop_first=True)
                use = [
                    "Number of scheduled trains",
                    "Month",
                    "Departure station",
                    "Arrival station",
                    "Average journey time",
                    "Service",
                ]
                avail = [c for c in use if c in df.columns]
                X_ref = pd.get_dummies(
                    df[avail],
                    columns=[
                        c
                        for c in [
                            "Departure station",
                            "Month",
                            "Arrival station",
                            "Service",
                        ]
                        if c in avail
                    ],
                    drop_first=True,
                )
                row_enc = row_enc.reindex(columns=X_ref.columns.tolist(), fill_value=0)
                pred = max(0, model.predict(row_enc)[0])
                if pred < 15:
                    label = "Minimum delay ✅"
                elif pred < 30:
                    label = "Low delay 🟡"
                elif pred < 60:
                    label = "Medium delay 🟠"
                else:
                    label = "Significant delay 🔴"
                st.success(f"**Predicted delay: {pred:.1f} minutes** — {label}")
                if hasattr(model, "feature_importances_"):
                    st.subheader("Top features")
                    imp = (
                        pd.Series(model.feature_importances_, index=X_ref.columns)
                        .sort_values(ascending=False)
                        .head(10)
                    )
                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.barh(imp.index[::-1], imp.values[::-1], color="#e63946")
                    ax.set_xlabel("Importance")
                    st.pyplot(fig)
                    plt.close()
            except Exception as e:
                st.error(f"Prediction error: {e}")
