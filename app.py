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
}

.login-box {
    background: linear-gradient(135deg, #1e3a8a, #1e40af);
    padding: 30px;
    border-radius: 18px;
    color: white;
}

.card {
    background-color: #111827;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #1f2937;
    margin-bottom: 20px;
}

.small {
    color: #c7d2fe;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- LOGIN ----------------
if not st.session_state.logged_in:

    st.markdown("<br><br><br>", unsafe_allow_html=True)
    _, center, _ = st.columns([1,2,1])

    with center:
        # Start blue box
        st.markdown('<div class="login-box">', unsafe_allow_html=True)

        st.markdown("## SkillBridge")
        st.markdown("<p class='small'>Sign in to continue</p>", unsafe_allow_html=True)
        st.markdown("---")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):
            if username == "admin" and password == "1234":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid credentials")

        # End blue box
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- MAIN APP ----------------
else:
    st.markdown("## SkillBridge")
    st.caption("Smart Internship Matching Platform")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    data = pd.read_csv("internships.csv")

    left, right = st.columns([3,1])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📄 Resume Input")
        resume_text = st.text_area(
            "Paste your resume content here",
            height=300
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        run = st.button("🚀 Find Matches", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if run:
        if len(resume_text.split()) < 5:
            st.warning("Please provide more details.")
            st.stop()

        desc = data["description"].tolist() + [resume_text]
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
        mat = vec.fit_transform(desc)

        scores = cosine_similarity(mat[-1], mat[:-1])[0] * 100
        data["Match Score"] = scores

        if scores.max() < 5:
            st.error("No relevant matches found.")
            st.stop()

        best = data.sort_values("Match Score", ascending=False).iloc[0]

        st.success(f"Best Match: {best['title']} ({best['Match Score']:.2f}%)")

    st.caption("SkillBridge © 2026")
