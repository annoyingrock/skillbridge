import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="SkillBridge", page_icon="📄", layout="centered")

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- LOGIN ----------------
if not st.session_state.logged_in:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
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

# ---------------- MAIN APP ----------------
else:
    st.title("SkillBridge")
    st.caption("Internship Matching Platform")

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

        word_count = len(resume_text.split())

        if word_count < 5:
            st.warning("Please provide more information about your skills or background.")
            st.stop()

        descriptions = data["description"].tolist()
        descriptions.append(resume_text)

        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
        vectors = vectorizer.fit_transform(descriptions)

        similarity = cosine_similarity(vectors[-1], vectors[:-1])
        data["Match Score"] = similarity[0] * 100

        max_score = data["Match Score"].max()

        if max_score < 5:
            st.error("No relevant internship matches found for this profile.")
            st.info("Try adding education, skills, or experience related to internships.")
            st.stop()

        results = data.sort_values(by="Match Score", ascending=False).reset_index(drop=True)
        results.index = results.index + 1

        top_match = results.iloc[0]

        st.success(f"Best Match: {top_match['title']} ({top_match['Match Score']:.2f}%)")
        st.progress(int(top_match["Match Score"]))

        display = results.copy()
        display["Match Score"] = display["Match Score"].map(lambda x: f"{x:.2f}%")

        st.subheader("Top Matching Internships")
        st.dataframe(display[["title", "Match Score"]].head(5))

        st.subheader("Resume Overview")
        st.write(f"Total Words in Resume: {word_count}")

        st.subheader("AI Career Path Suggestion")

        role = top_match["title"].lower()

        if any(k in role for k in ["reporting", "business", "operations", "management"]):
            path = ["Business Intern", "Business Analyst", "Operations Manager"]
        elif any(k in role for k in ["data", "analyst"]):
            path = ["Data Analyst Intern", "Data Analyst", "Data Scientist"]
        elif any(k in role for k in ["backend", "api", "server"]):
            path = ["Backend Intern", "Backend Engineer", "Software Architect"]
        elif any(k in role for k in ["frontend", "ui", "web"]):
            path = ["Frontend Intern", "Frontend Developer", "UI Lead"]
        elif any(k in role for k in ["cloud", "devops"]):
            path = ["Cloud Intern", "Cloud Engineer", "Cloud Architect"]
        else:
            path = ["General Intern", "Specialist", "Team Lead"]

        st.write("Entry Level:", path[0])
        st.write("Mid Level:", path[1])
        st.write("Advanced Level:", path[2])

        st.session_state["results"] = display[["title", "Match Score"]]

    if "results" in st.session_state:
        st.sidebar.download_button(
            "Download Results",
            st.session_state["results"].to_csv(index=False),
            "skillbridge_results.csv",
            "text/csv"
        )

    st.markdown("---")
    st.caption("SkillBridge © 2026")
