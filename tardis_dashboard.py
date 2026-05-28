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

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🗺️ Stations", "🤖 Predict", "🧠 Quiz"])

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

with tab4:
    st.subheader("🧠 Train World Quiz")
    st.markdown("Test your knowledge about world railways! Answer all questions then click **Submit**.")

    quiz = [
        {
            "question": "What is the longest railway line in the world?",
            "options": [
                "The Trans-Siberian Railway (Russia)",
                "The Indian Railway Main Line",
                "The Canadian Pacific Railway",
                "The Australian Indian Pacific",
            ],
            "answer": 0,
            "explanation": "The Trans-Siberian Railway links Moscow to Vladivostok over 9,289 km, crossing 8 time zones.",
            "stat": {
                "label": "Trans-Siberian Railway",
                "details": [
                    ("Total length", "9,289 km"),
                    ("Time zones crossed", "8"),
                    ("Journey duration", "~6 days"),
                    ("Countries", "Russia only"),
                    ("Opened", "1916"),
                ]
            }
        },
        {
            "question": "What is generally the most expensive train journey in the world?",
            "options": [
                "The Venice Simplon-Orient-Express (Europe)",
                "The Blue Train (South Africa)",
                "The Maharajas' Express (India)",
                "The Rovos Rail Pride of Africa (South Africa)",
            ],
            "answer": 3,
            "explanation": "The Rovos Rail Pride of Africa can cost over $10,000 per person for its longest itinerary.",
            "stat": {
                "label": "Most expensive train journeys",
                "details": [
                    ("Rovos Rail Pride of Africa", "~$10,000+/person"),
                    ("Maharajas' Express", "~$5,000–$23,000/person"),
                    ("Venice Simplon-Orient-Express", "~$2,000–$5,000/person"),
                    ("The Blue Train", "~$1,000–$2,000/person"),
                ]
            }
        },
        {
            "question": "What is the oldest steam locomotive still in working order?",
            "options": [
                "The Rocket (1829, UK)",
                "Puffing Billy (1813, UK)",
                "The Fairy Queen (1855, India)",
                "The Locomotion No.1 (1825, UK)",
            ],
            "answer": 2,
            "explanation": "The Fairy Queen (1855) holds the Guinness World Record as the oldest working steam locomotive.",
            "stat": {
                "label": "The Fairy Queen",
                "details": [
                    ("Built", "1855"),
                    ("Built by", "Kitson Thompson & Hewitson, UK"),
                    ("Now operating in", "India (Delhi → Alwar)"),
                    ("Record", "Guinness World Record — oldest working loco"),
                    ("Age", "~170 years old"),
                ]
            }
        },
        {
            "question": "Who is credited with inventing the first steam locomotive?",
            "options": [
                "James Watt",
                "George Stephenson",
                "Richard Trevithick",
                "Robert Fulton",
            ],
            "answer": 2,
            "explanation": "Richard Trevithick built the first full-scale working railway steam locomotive in 1804 in Wales, UK.",
            "stat": {
                "label": "Richard Trevithick",
                "details": [
                    ("Born", "1771, Cornwall, UK"),
                    ("First locomotive", "1804 — Merthyr Tydfil, Wales"),
                    ("Speed of first run", "~8 km/h"),
                    ("Passengers carried", "70 people on first run"),
                    ("Died", "1833"),
                ]
            }
        },
        {
            "question": "What is the fastest train in the world (commercial service)?",
            "options": [
                "TGV (France)",
                "Shinkansen N700S (Japan)",
                "Shanghai Maglev (China)",
                "Frecciarossa 1000 (Italy)",
            ],
            "answer": 2,
            "explanation": "The Shanghai Maglev reaches 430 km/h in commercial service.",
            "stat": {
                "label": "Speed comparison",
                "details": [
                    ("Shanghai Maglev (China)", "430 km/h"),
                    ("Frecciarossa 1000 (Italy)", "360 km/h"),
                    ("TGV (France)", "320 km/h (record: 574 km/h in tests)"),
                    ("Shinkansen N700S (Japan)", "285 km/h"),
                    ("Eurostar (UK/France)", "300 km/h"),
                ]
            }
        },
    ]
    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}

    with st.form("quiz_form"):
        for i, q in enumerate(quiz):
            st.markdown(f"**Question {i+1} — {q['question']}**")
            st.session_state.quiz_answers[i] = st.radio(
                label=f"q{i}",
                options=q["options"],
                index=None,
                label_visibility="collapsed",
                key=f"q_{i}",
            )
            st.markdown("")

        submitted = st.form_submit_button("✅ Submit my answers", type="primary")

    if submitted:
        score = 0
        st.divider()
        st.subheader("📋 Results")

        for i, q in enumerate(quiz):
            user_choice = st.session_state.quiz_answers.get(i)
            correct_option = q["options"][q["answer"]]

            col_result, col_stat = st.columns([1, 1])

            with col_result:
                if user_choice == correct_option:
                    score += 1
                    st.success(f"**Q{i+1}** ✅ Correct!\n\n_{q['explanation']}_")
                elif user_choice is None:
                    st.warning(f"**Q{i+1}** ⚠️ No answer.\n\nCorrect: *{correct_option}*")
                else:
                    st.error(f"**Q{i+1}** ❌ Wrong.\n\nYou answered: *{user_choice}*\n\n *{correct_option}* — {q['explanation']}")

            with col_stat:
                stat = q["stat"]
                st.markdown(f"**{stat['label']}**")
                for k, v in stat["details"]:
                    st.markdown(f"- **{k}:** {v}")

            st.divider()

        pct = int(score / len(quiz) * 100)
        if pct == 100:
            st.balloons()
            st.success(f"Perfect score! {score}/{len(quiz)} — You're a true rail expert!")
        elif pct >= 66:
            st.info(f"Good job! {score}/{len(quiz)} — You know your trains well.")
        elif pct >= 33:
            st.warning(f"Not bad! {score}/{len(quiz)} — A bit more study and you'll ace it.")
        else:
            st.error(f"{score}/{len(quiz)} — Time to hop on the Trans-Siberian and learn along the way!")