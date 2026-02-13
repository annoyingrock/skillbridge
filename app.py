import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="SkillBridge", page_icon="📄", layout="centered")

# ---------- HEADER ----------
st.title("SkillBridge")
st.markdown("### Smart Internship Matching Platform")

st.markdown("""
SkillBridge helps students discover internships that align with their skills.
Enter your resume content below and explore matching opportunities.
""")

st.divider()

# ---------- LOAD DATA ----------
try:
    data = pd.read_csv("internships.csv")
except:
    st.error("Could not load internship data.")
    st.stop()

# ---------- SIDEBAR ----------
st.sidebar.header("Filter Options")
selected_role = st.sidebar.selectbox(
    "Filter by Role (Optional)",
    ["All"] + list(data["title"].unique())
)

if selected_role != "All":
    data = data[data["title"] == selected_role]

# ---------- RESUME INPUT ----------
st.subheader("Resume Input")
resume_text = st.text_area("Paste your resume content here", height=200)

# ---------- MATCHING ----------
if st.button("Find Matches"):

    if resume_text.strip() == "":
        st.warning("Please enter resume text.")
    else:

        descriptions = data["description"].tolist()
        descriptions.append(resume_text)

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(descriptions)

        similarity = cosine_similarity(vectors[-1], vectors[:-1])

        data["Match Score (%)"] = similarity[0] * 100

        results = (
            data.sort_values(by="Match Score (%)", ascending=False)
            .reset_index(drop=True)
        )

        results.index = results.index + 1

        # Display Top Match
        top_match = results.iloc[0]

        st.success(
            f"Best Match: {top_match['title']} "
            f"({top_match['Match Score (%)']:.2f}%)"
        )

        st.progress(int(top_match["Match Score (%)"]))

        st.subheader("Top Matching Internships")

        # Format percentage display
        results_display = results.copy()
        results_display["Match Score (%)"] = results_display["Match Score (%)"].map(
            lambda x: f"{x:.2f}%"
        )

        st.dataframe(results_display[["title", "Match Score (%)"]].head(5))

        # Resume Summary
        st.subheader("Resume Overview")
        st.write(f"Total Words in Resume: {len(resume_text.split())}")

# ---------- FOOTER ----------
st.markdown("""
---
SkillBridge © 2026
""")

