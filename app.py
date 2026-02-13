import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="SkillBridge", page_icon="📄", layout="wide")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
body {
    background-color: #f5f7fa;
}
.hero {
    background: linear-gradient(90deg, #4e73df, #1cc88a);
    padding: 40px;
    border-radius: 10px;
    color: white;
    text-align: center;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# -------- HERO SECTION --------
st.markdown("""
<div class="hero">
    <h1>SkillBridge</h1>
    <h3>Smart Internship Matching Platform</h3>
    <p>Match your resume with the right opportunities instantly.</p>
</div>
""", unsafe_allow_html=True)

st.write("")
st.write("")

try:
    data = pd.read_csv("internships.csv")
except:
    st.error("Unable to load internship data.")
    st.stop()

# -------- SIDEBAR --------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=120)
st.sidebar.title("Explore Opportunities")

selected_role = st.sidebar.selectbox(
    "Choose Internship Role",
    ["All Internships"] + sorted(data["title"].unique())
)

if selected_role != "All Internships":
    data = data[data["title"] == selected_role]

st.sidebar.markdown("---")
st.sidebar.title("Resume Actions")

if st.sidebar.button("Reset Resume"):
    st.session_state["resume"] = ""

# -------- MAIN CARD --------
st.markdown('<div class="card">', unsafe_allow_html=True)

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

if "results" in st.session_state:
    st.sidebar.download_button(
        label="Download Match Results",
        data=st.session_state["results"].to_csv(index=False),
        file_name="skillbridge_results.csv",
        mime="text/csv"
    )

st.markdown('</div>', unsafe_allow_html=True)

st.write("")
st.markdown(
    "<center><small>© 2026 SkillBridge</small></center>",
    unsafe_allow_html=True
)

