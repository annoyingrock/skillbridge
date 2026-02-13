import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="SkillBridge", page_icon="📄", layout="centered")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.title("SkillBridge Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid credentials")

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

    resume_text = st.text_area(
        "Paste your resume content here",
        height=200,
        key="resume"
    )

    if st.button("Find Matches"):

        if resume_text.strip() == "":
            st.warning("Please enter resume text.")
        else:

            descriptions = data["description"].tolist()
            descriptions.append(resume_text)

            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform(descriptions)

            similarity = cosine_similarity(vectors[-1], vectors[:-1])
            data["Match Score"] = similarity[0] * 100

            results = (
                data.sort_values(by="Match Score", ascending=False)
                .reset_index(drop=True)
            )
            results.index = results.index + 1

            top_match = results.iloc[0]

            st.success(
                f"Best Match: {top_match['title']} ({top_match['Match Score']:.2f}%)"
            )
            st.progress(int(top_match["Match Score"]))

            results_display = results.copy()
            results_display["Match Score"] = results_display["Match Score"].map(
                lambda x: f"{x:.2f}%"
            )

            st.subheader("Top Matching Internships")
            st.dataframe(results_display[["title", "Match Score"]].head(5))

            st.subheader("Resume Overview")
            st.write(f"Total Words in Resume: {len(resume_text.split())}")

            # ---------- AI CAREER PATH GENERATOR ----------
            st.subheader("AI Career Path Suggestion")

            role = top_match["title"].lower()

            if "python" in role or "backend" in role:
                path = ["Backend Developer", "Senior Backend Engineer", "Software Architect"]
            elif "data" in role:
                path = ["Data Analyst", "Data Scientist", "AI Engineer"]
            elif "machine learning" in role or "ai" in role:
                path = ["ML Engineer", "Senior ML Engineer", "AI Researcher"]
            elif "cloud" in role or "devops" in role:
                path = ["Cloud Engineer", "DevOps Engineer", "Cloud Architect"]
            elif "it" in role or "systems" in role:
                path = ["Systems Engineer", "Infrastructure Lead", "IT Architect"]
            else:
                path = ["Technical Intern", "Technology Specialist", "Technology Lead"]

            st.write("Entry Level:", path[0])
            st.write("Mid Level:", path[1])
            st.write("Advanced Level:", path[2])

            st.session_state["results"] = results_display[["title", "Match Score"]]

    if "results" in st.session_state:
        st.sidebar.download_button(
            label="Download Match Results",
            data=st.session_state["results"].to_csv(index=False),
            file_name="skillbridge_results.csv",
            mime="text/csv"
        )

    st.markdown("---")
    st.markdown("SkillBridge © 2026")
