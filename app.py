import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="SkillBridge", page_icon="📄", layout="centered")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("SkillBridge")
st.markdown("### Smart Internship Matching Platform")
st.markdown("Match your resume with internships based on skill similarity.")

st.divider()

try:
    data = pd.read_csv("internships.csv")
except:
    st.error("Could not load internship data.")
    st.stop()

st.sidebar.header("User Tools")

if st.sidebar.button("Clear Resume"):
    st.session_state["resume"] = ""

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

        st.session_state["results"] = results_display[["title", "Match Score"]]

st.sidebar.subheader("Download Results")

if "results" in st.session_state:
    st.sidebar.download_button(
        label="Download CSV",
        data=st.session_state["results"].to_csv(index=False),
        file_name="match_results.csv",
        mime="text/csv"
    )

st.divider()

st.markdown("SkillBridge © 2026")
