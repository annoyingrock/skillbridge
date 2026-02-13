import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="SkillBridge", page_icon="📄")

st.title("SkillBridge - Internship Matcher")
st.write("Enter your resume content to find suitable internships.")

# Load internship data
try:
    data = pd.read_csv("internships.csv")
except:
    st.error("Could not load internship data.")
    st.stop()

resume_text = st.text_area("Resume Content", height=200)

if st.button("Find Matches"):

    if resume_text.strip() == "":
        st.warning("Please enter resume text.")
    else:

        # Combine descriptions + resume
        descriptions = data["description"].tolist()
        descriptions.append(resume_text)

        # Convert text to numeric vectors
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(descriptions)

        # Calculate similarity
        similarity = cosine_similarity(vectors[-1], vectors[:-1])

        # Convert to percentage
        data["Match Score (%)"] = similarity[0] * 100

        # Sort and reset index
        results = (
            data.sort_values(by="Match Score (%)", ascending=False)
            .reset_index(drop=True)
        )

        # Start numbering from 1
        results.index = results.index + 1

        # Add % symbol ONLY for display
        results["Match Score (%)"] = results["Match Score (%)"].map(
            lambda x: f"{x:.2f}%"
        )

        st.subheader("Top Matches")

        st.dataframe(
            results[["title", "Match Score (%)"]].head(5)
        )
