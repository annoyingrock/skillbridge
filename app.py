import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="SkillBridge", page_icon="📄", layout="centered")

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

.login-card {
    background-color: #111827;
    padding: 40px;
    border-radius: 12px;
    max-width: 420px;
    margin: auto;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.4);
}

.login-title {
    text-align: center;
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 5px;
}

.login-subtitle {
    text-align: center;
    color: #9ca3af;
    margin-bottom: 30px;
}

.login-btn button {
    width: 100%;
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    color: white;
    border-radius: 8px;
    font-size: 16px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- LOGIN PAGE ----------------
if not st.session_state.logged_in:

    st.markdown("""
    <div class="login-card">
        <div class="login-title">SkillBridge</div>
        <div class="login-subtitle">Sign in to continue</div>
    """, unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    st.markdown('<div class="login-btn">', unsafe_allow_html=True)
    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.markdown('</div></div>', unsafe_allow_html=True)

# ---------------- MAIN APP ----------------
else:

    st.title("SkillBridge")
    st.markdown("Internship Matching Platform")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    data = pd.read_csv("internships.csv")

    st.sidebar.title("Explore Opportunities")
    selected_role = st.sidebar.selectbox(
        "Choose Internship Role",
        ["All Internships"] + sorted(data["title"].unique())
    )

    if selected_role != "All Internships":
        data = data[data["title"] == selected_role]

    resume_text = st.text_area("Paste your resume content here", height=200)

    if st.button("Find Matches"):

        if resume_text.strip() == "":
            st.warning("Please enter resume text.")
        else:

            descriptions = data["description"].tolist()
            descriptions.append(resume_text)

            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
            vectors = vectorizer.fit_transform(descriptions)

            similarity = cosine_similarity(vectors[-1], vectors[:-1])
            data["Match Score"] = similarity[0] * 100

            results = data.sort_values(by="Match Score", ascending=False).reset_index(drop=True)
            results.index = results.index + 1

            top_match = results.iloc[0]

            if top_match["Match Score"] < 10:
                st.warning("No strong internship match found. Try adding more skills or experience.")
                st.stop()

            st.success(f"Best Match: {top_match['title']} ({top_match['Match Score']:.2f}%)")
            st.progress(int(top_match["Match Score"]))

            results_display = results.copy()
            results_display["Match Score"] = results_display["Match Score"].map(lambda x: f"{x:.2f}%")

            st.subheader("Top Matching Internships")
            st.dataframe(results_display[["title", "Match Score"]].head(5))

            st.subheader("Resume Overview")
            st.write(f"Total Words in Resume: {len(resume_text.split())}")

            st.subheader("AI Career Path Suggestion")

            role = top_match["title"].lower()
            if "business" in role or "operations" in role or "reporting" in role:
                path = ["Business Analyst", "Operations Manager", "Strategy Lead"]
            elif "python" in role or "backend" in role:
                path = ["Backend Developer", "Senior Backend Engineer", "Software Architect"]
            elif "data" in role:
                path = ["Data Analyst", "Data Scientist", "AI Engineer"]
            else:
                path = ["Technical Intern", "Technology Specialist", "Technology Lead"]

            st.write("Entry Level:", path[0])
            st.write("Mid Level:", path[1])
            st.write("Advanced Level:", path[2])

            st.session_state["results"] = results_display[["title", "Match Score"]]

    if "results" in st.session_state:
        st.sidebar.download_button(
            "Download Results",
            st.session_state["results"].to_csv(index=False),
            "skillbridge_results.csv",
            "text/csv"
        )

    st.markdown("---")
    st.markdown("SkillBridge © 2026")
