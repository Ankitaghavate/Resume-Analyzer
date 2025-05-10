import nltk
import pickle
import re
import streamlit as st
from io import StringIO
from pdfminer.high_level import extract_text
import time
from streamlit_lottie import st_lottie
import json
import requests
from streamlit.components.v1 import html

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load models
@st.cache_resource
def load_models():
    clf = pickle.load(open('clf.pkl', 'rb'))
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))
    return clf, tfidf

clf, tfidf = load_models()

# Load Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Animation URLs
lottie_upload = "https://assets8.lottiefiles.com/packages/lf20_vnikrcia.json"
lottie_typing = "https://assets1.lottiefiles.com/packages/lf20_4kx2q32h.json"

# Clean resume text
def cleanResume(txt):
    cleantxt = re.sub('http\S+\s', ' ', txt)
    cleantxt = re.sub('@\S+', ' ', cleantxt)
    cleantxt = re.sub('#\S+', " ", cleantxt)
    cleantxt = re.sub('RT|CC', ' ', cleantxt)
    cleantxt = re.sub(r'[!"#&%\'()*+,-./:;<=>?@[\]^_`{|}~]', ' ', cleantxt)
    cleantxt = re.sub('[^a-zA-Z]', ' ', cleantxt)
    cleantxt = re.sub('\s+', ' ', cleantxt)
    cleantxt = cleantxt.lower()
    return cleantxt

# Extract text from PDF using pdfminer
def extract_text_from_pdf(file):
    return extract_text(file)

# Typing animation effect
def typing_animation(text: str, speed: int = 50):
    return f"""
    <script>
        const text = "{text}";
        const element = document.getElementById("typing-animation");
        let i = 0;
        function typeWriter() {{
            if (i < text.length) {{
                element.innerHTML += text.charAt(i);
                i++;
                setTimeout(typeWriter, {speed});
            }}
        }}
        typeWriter();
    </script>
    """

# Web App
def main():
    # Custom CSS for styling
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
        
        * {
            font-family: 'Poppins', sans-serif;
        }
        
        .main {
            background-color: white;
        }
        .stApp {
            background-color: white;
        }
        .title-text {
            color: black;
            text-align: center;
            font-size: 2.8em;
            margin-bottom: 0.2em;
            font-weight: 700;
        }
        .subheader {
            color: black;
            text-align: center;
            font-size: 1.4em;
            margin-bottom: 2em;
            font-weight: 400;
        }
        .upload-section {
            border: 2px dashed #ccc;
            border-radius: 8px;
            padding: 30px;
            text-align: center;
            margin: 20px 0;
        }
        .footer {
            text-align: center;
            color: black;
            font-size: 0.8em;
            margin-top: 3em;
            padding: 1em;
            border-top: 1px solid #eee;
        }
        .result-box {
            background-color: #f8f9fa;
            color: black;
            padding: 20px;
            border-radius: 8px;
            font-size: 1.5em;
            font-weight: bold;
            margin: 20px auto;
            max-width: 80%;
            text-align: center;
            border: 1px solid #ddd;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header with typing animation
    col1, col2, col3 = st.columns([1,3,1])
    
    with col2:
        
        st.markdown("""
    <div class="footer">
        <h3>Resume Screening Pro</h3>
        
    </div>
    """, unsafe_allow_html=True)

        
        # Add typing animation lottie
        lottie_typing_json = load_lottieurl(lottie_typing)
        if lottie_typing_json:
            st_lottie(lottie_typing_json, height=100, key="typing", speed=1)
        
        # Subheader
        st.markdown('<p class="subheader">Upload your resume in PDF or TXT format for instant analysis</p>', unsafe_allow_html=True)

    # File upload section
    

        
        
        uploaded_file = st.file_uploader('Browse files', type=['txt', 'pdf'], label_visibility="collapsed")
        
        if uploaded_file is not None:
            lottie_upload_json = load_lottieurl(lottie_upload)
            if lottie_upload_json:
                st_lottie(lottie_upload_json, height=150, key="upload", speed=1)

    # Processing and results
    if uploaded_file is not None:
        try:
            # Show loading animation
            with st.spinner('Analyzing your resume...'):
                # Add a progress bar
                progress_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(percent_complete + 1)
                
                if uploaded_file.type == "text/plain":
                    # Read text file directly
                    resume_text = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
                elif uploaded_file.type == "application/pdf":
                    # Extract text from PDF
                    resume_text = extract_text_from_pdf(uploaded_file)
                else:
                    st.error("Unsupported file type.")
                    return

                cleaned_resume = cleanResume(resume_text)
                input_feature = tfidf.transform([cleaned_resume])
                prediction_id = clf.predict(input_feature)[0]

                category_mapping = {
                    0: "Advocate", 1: "Arts", 2: "Automation Testing", 3: "Blockchain",
                    4: "Business Analyst", 5: "Civil Engineer", 6: "Data Science",
                    7: "Database", 8: "DevOps Engineer", 9: "DotNet Developer",
                    10: "ETL Developer", 11: "Electrical Engineering", 12: "HR",
                    13: "Hadoop", 14: "Health and fitness", 15: "Java Developer",
                    16: "Mechanical Engineer", 17: "Network Security Engineer",
                    18: "Operations Manager", 19: "PMO", 20: "Python Developer",
                    21: "SAP Developer", 22: "Sales", 23: "Testing", 24: "Web Designing"
                }

                category_name = category_mapping.get(prediction_id, "Unknown")
                
                # Show result
                progress_bar.empty()
                st.markdown(f"""
                <div class="result-box">
                    {category_name}
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("Please try uploading a different file.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>Powered by AI and Streamlit | Upload your resume to discover the best matching job category</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()