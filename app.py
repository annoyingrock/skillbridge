import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="SkillBridge", page_icon="📄", layout="wide")

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

body {
    background-color: #0f172a;
    color: #e5e7eb;
}

.card {
    background-color: #111827;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #1f2937;
    margin-bottom: 20px;
}

.highlight {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    padding: 20px;
    border-radius: 14px;
    color: white;
    margin-bottom: 20px;
}

.small-text {
    color: #9ca3af;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- LOGIN ----------------
if not st.session_state.logged_in:

    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("## SkillBridge")
        st.caption("Sign in to continue")
        st.markdown("---")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):
            if username == "admin" and password == "1234":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid credentials")
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- MAIN APP ----------------
else:

    st.markdown("## SkillBridge")
    st.caption("Smart Internship Matching Platform")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    data = pd.read_csv("internships.csv")

    st.sidebar.title("Explore")
    selected_role = st.sidebar.selectbox(
        "Filter by Role",
        ["All Internships"] + sorted(data["title"].unique())
    )

    if selected_role != "All Internships":
        data = data[data["title"] == selected_role]

    left, right = st.columns([2,1])

    # -------- LEFT: RESUME INPUT --------
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📄 Resume Input")
        resume_text = st.text_area(
            "Paste your resume content here",
            height=220,
            placeholder="Enter education, skills, experience..."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # -------- RIGHT: ACTION --------
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🚀 Action")
        run = st.button("Find Matches", use_container_width=True)
        st.markdown('<p class="small-text">Results are based on skill similarity.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if run:

        words = len(resume_text.split())
        if words < 5:
            st.warning("Please provide more details about your profile.")
            st.stop()

        descriptions = data["description"].tolist()
        descriptions.append(resume_text)

        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
        vectors = vectorizer.fit_transform(descriptions)

        similarity = cosine_similarity(vectors[-1], vectors[:-1])
        data["Match Score"] = similarity[0] * 100

        if data["Match Score"].max() < 5:
            st.error("No relevant internship matches found.")
            st.info("Try adding education, skills, or experience related to internships.")
            st.stop()

        results = data.sort_values(by="Match Score", ascending=False).reset_index(drop=True)
        top = results.iloc[0]

        # -------- BEST MATCH --------
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown(f"### 🎯 Best Match")
        st.markdown(f"**{top['title']}**")
        st.markdown(f"Match Score: **{top['Match Score']:.2f}%**")
        st.progress(int(top["Match Score"]))
        st.markdown('</div>', unsafe_allow_html=True)

        # -------- RESULTS + CAREER PATH --------
        colA, colB = st.columns(2)

        with colA:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📊 Top Matches")
            display = results.head(5).copy()
            display["Match Score"] = display["Match Score"].map(lambda x: f"{x:.2f}%")
            st.dataframe(display[["title", "Match Score"]], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with colB:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🧠 AI Career Path")

            role = top["title"].lower()
            if any(k in role for k in ["business", "operations", "reporting"]):
                path = ["Business Intern", "Business Analyst", "Operations Manager"]
            elif "data" in role:
                path = ["Data Analyst Intern", "Data Analyst", "Data Scientist"]
            elif "backend" in role or "python" in role:
                path = ["Backend Intern", "Backend Engineer", "Software Architect"]
            elif "frontend" in role or "ui" in role:
                path = ["Frontend Intern", "Frontend Developer", "UI Lead"]
            elif "cloud" in role or "devops" in role:
                path = ["Cloud Intern", "Cloud Engineer", "Cloud Architect"]
            else:
                path = ["General Intern", "Specialist", "Team Lead"]

            st.write("Entry Level:", path[0])
            st.write("Mid Level:", path[1])
            st.write("Advanced Level:", path[2])
            st.markdown('</div>', unsafe_allow_html=True)

        st.sidebar.download_button(
            "Download Results",
            display.to_csv(index=False),
            "skillbridge_results.csv",
            "text/csv"
        )

    st.markdown("---")
    st.caption("SkillBridge © 2026")
