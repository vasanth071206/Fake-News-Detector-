import streamlit as st
from google import genai
from dotenv import load_dotenv
import os
import time

# ---------------- LOAD API KEY ----------------
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# Gemini client
client = genai.Client(api_key=GOOGLE_API_KEY)

# Initialize session state
if "analysis_in_progress" not in st.session_state:
    st.session_state.analysis_in_progress = False

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fake News Detector AI",
    page_icon="📰",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("📰 AI Fake News Detector (Gemini Powered)")
st.write("Detect whether a news article is REAL or FAKE using AI.")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Project Info")
    st.write("AI Model: Gemini 2.5 Flash")
    st.write("Framework: Streamlit")
    st.write("Type: NLP + Generative AI Project")

# ---------------- GEMINI FUNCTION ----------------
def analyze_news(news):

    prompt = f"""
You are a strict Fake News Detection system.

Analyze the news and return EXACTLY in this format:

Verdict: REAL or FAKE

Confidence: 0-100

Reason:
- Maximum 3 short lines

News:
{news}
"""


    max_retries = 2
    retry_delay = 3
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text

        except Exception as e:
            error_msg = str(e).lower()
            # Only retry on service unavailable/rate limit errors
            if "503" in error_msg or "busy" in error_msg or "rate" in error_msg:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
            # For other errors, fail fast
            return f"⚠️ Error: {str(e)[:100]}"

    return "⚠️ API service is busy. Please try again in a moment."

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["Analyze News", "Result", "Tips"])

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("Paste News Article")

    news_input = st.text_area("Enter news text here", height=250, key="news_input")

    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_button = st.button("Analyze", key="analyze_btn")

    if analyze_button:
        if not news_input.strip():
            st.warning("Please enter a news article.")
        else:
            # Prevent duplicate calls
            if not st.session_state.analysis_in_progress:
                st.session_state.analysis_in_progress = True
                
                with st.spinner("Analyzing with AI..."):
                    result = analyze_news(news_input)
                    st.session_state.result = result
                
                st.session_state.analysis_in_progress = False
                st.success("Analysis Completed! Check the 'Result' tab.")
            else:
                st.warning("Analysis in progress... Please wait.")

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("AI Result")

    if "result" in st.session_state:
        result = st.session_state["result"]

        result = result.replace("Verdict:", "\n### Verdict\n")
        result = result.replace("Confidence:", "\n### Confidence\n")
        result = result.replace("Reason:", "\n### Reason\n")

        st.markdown(result)
    else:
        st.info("No analysis yet. Go to Analyze News tab.")

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("How to verify news manually")

    st.markdown("""
- Check trusted sources like BBC, Reuters, The Hindu  
- Verify publication date  
- Cross-check with multiple news websites  
- Avoid viral social media claims  
- Look for evidence and official statements  
""")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built using Streamlit + Gemini AI")
