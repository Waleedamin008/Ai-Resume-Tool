"""
AI Resume Job Predictor - Streamlit Web Application
A production-ready ML application for predicting job roles from resumes
"""

import streamlit as st
import time
from src.predict import predict_top3
from src.pdf_utils import extract_text_from_pdf


# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="AI Resume Job Predictor | Career Path Analyzer",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
    <style>
    /* Import Google Fonts and Font Awesome */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background with animated gradient */
    .stApp {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #4facfe);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container with glassmorphism */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Hero Section */
    .hero-section {
        text-align: center;
        padding: 3rem 2rem;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 30px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        animation: fadeInUp 0.8s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .hero-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: white;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        line-height: 1.2;
    }
    
    .hero-gradient-text {
        background: linear-gradient(135deg, #fff 0%, #f0f0f0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Feature Cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 20px;
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.3);
        animation: fadeInUp 0.8s ease-out;
    }
    
    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
        background: rgba(255, 255, 255, 1);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        font-size: 0.95rem;
        color: #718096;
        line-height: 1.6;
    }
    
    /* Upload Section */
    .upload-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 2.5rem;
        border-radius: 25px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
        margin: 2rem 0;
        animation: fadeInUp 1s ease-out;
    }
    
    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Prediction Cards - Enhanced */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        animation: slideInRight 0.5s ease-out;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 100%);
        pointer-events: none;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.5);
    }
    
    .rank-badge {
        background: rgba(255, 255, 255, 0.25);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 1rem;
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    
    .job-title {
        font-size: 2rem;
        font-weight: 800;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .confidence-score {
        font-size: 3rem;
        font-weight: 800;
        margin-top: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .confidence-label {
        font-size: 1rem;
        opacity: 0.9;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Progress Bar Enhanced */
    .progress-bar {
        background: rgba(255, 255, 255, 0.3);
        height: 12px;
        border-radius: 10px;
        overflow: hidden;
        margin-top: 1.5rem;
        position: relative;
    }
    
    .progress-fill {
        background: white;
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease-out;
        box-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
    }
    
    /* Match Quality Indicator */
    .match-indicator {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.5rem;
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .stat-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.8);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Button Styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        padding: 1rem 2rem;
        border-radius: 15px;
        border: none;
        font-size: 1.2rem;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* File Uploader Styling */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 15px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
        background: rgba(255, 255, 255, 0.1);
    }
    
    /* Text Area */
    .stTextArea textarea {
        border-radius: 15px;
        border: 2px solid rgba(102, 126, 234, 0.3);
        font-size: 1rem;
        padding: 1rem;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        font-weight: 600;
        color: white;
    }
    
    /* Success/Error Messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 15px;
        padding: 1rem;
        font-weight: 600;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: white;
        margin-top: 4rem;
        padding: 2rem;
        font-size: 0.95rem;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .footer-links {
        margin-top: 1rem;
        display: flex;
        justify-content: center;
        gap: 2rem;
        flex-wrap: wrap;
    }
    
    .footer-link {
        color: white;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .footer-link:hover {
        transform: translateY(-2px);
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
    }
    
    /* Spinner Override */
    .stSpinner > div {
        border-top-color: white !important;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        .hero-subtitle {
            font-size: 1.1rem;
        }
        .feature-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
""", unsafe_allow_html=True)

# ==================== HEADER & HERO ====================
st.markdown("""
    <div class="hero-section">
        <div class="hero-icon">🎯</div>
        <h1 class="hero-title">
            <span class="hero-gradient-text">AI Resume Job Predictor</span>
        </h1>
        <p class="hero-subtitle">
            ✨ Discover your perfect career path powered by advanced machine learning ✨
        </p>
    </div>
""", unsafe_allow_html=True)

# ==================== FEATURES SECTION ====================
st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">🚀</div>
            <div class="feature-title">Instant Analysis</div>
            <div class="feature-desc">Get predictions in seconds using advanced NLP algorithms</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Top-3 Matches</div>
            <div class="feature-desc">See your best job role matches with confidence scores</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🔒</div>
            <div class="feature-title">100% Private</div>
            <div class="feature-desc">Your resume is never stored or shared anywhere</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <div class="feature-title">AI-Powered</div>
            <div class="feature-desc">Built with TF-IDF vectorization & Logistic Regression</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# ==================== FILE UPLOAD SECTION ====================
# st.markdown('<div class="upload-container">', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">📤 Upload Your Resume</h2>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload your resume in PDF format (max 200MB)",
        label_visibility="collapsed"
    )

with col2:
    st.markdown("""
        <div style="background: rgba(102, 126, 234, 0.1); padding: 1.5rem; border-radius: 15px; margin-top: 1rem;">
            <div style="font-size: 2rem; text-align: center; margin-bottom: 0.5rem;">📄</div>
            <div style="text-align: center; color: #4a5568; font-weight: 600; font-size: 0.9rem;">
                PDF Format<br/>Max 200MB
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Alternative text input
# st.markdown('<div class="upload-container" style="margin-top: 1.5rem;">', unsafe_allow_html=True)
with st.expander("✍️ Or paste your resume text here", expanded=False):
    resume_text_input = st.text_area(
        "Resume Text",
        height=250,
        placeholder="Paste your complete resume content here...\n\nInclude:\n• Work experience\n• Skills\n• Education\n• Projects\n• Certifications",
        label_visibility="collapsed"
    )
st.markdown("</div>", unsafe_allow_html=True)

# ==================== PREDICTION SECTION ====================
st.markdown('<div style="text-align: center; margin: 2rem 0;">', unsafe_allow_html=True)
predict_button = st.button("🔮 Analyze Resume & Predict Jobs", type="primary")
st.markdown('</div>', unsafe_allow_html=True)

if predict_button:
    resume_text = None
    
    # Get resume text from PDF or text input
    if uploaded_file is not None:
        try:
            with st.spinner("📄 Extracting text from PDF..."):
                time.sleep(0.5)  # Brief pause for better UX
                resume_text = extract_text_from_pdf(uploaded_file)
        except Exception as e:
            st.error(f"❌ Error reading PDF: {str(e)}")
            st.stop()
    
    elif resume_text_input.strip():
        resume_text = resume_text_input
    
    else:
        st.warning("⚠️ Please upload a PDF file or paste your resume text to continue.")
        st.stop()
    
    # Validate resume text
    if not resume_text or len(resume_text.strip()) < 50:
        st.error("❌ Resume text is too short. Please provide a valid resume with at least 50 characters.")
        st.stop()
    
    # Show processing animation
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Simulate processing steps for better UX
    steps = [
        ("🔍 Analyzing resume structure...", 0.3),
        ("🧠 Extracting key skills and experience...", 0.5),
        ("📊 Comparing with job market data...", 0.7),
        ("🎯 Calculating match probabilities...", 0.9),
        ("✅ Finalizing predictions...", 1.0)
    ]
    
    progress_bar = progress_placeholder.progress(0)
    
    for step_text, progress_val in steps:
        status_placeholder.info(step_text)
        progress_bar.progress(progress_val)
        time.sleep(0.4)
    
    # Make predictions
    try:
        predictions = predict_top3(resume_text)
        
        # Clear progress indicators
        progress_placeholder.empty()
        status_placeholder.empty()
        
        if not predictions:
            st.error("❌ Could not make predictions. Please check your resume content.")
            st.stop()
        
        # Display success message with animation
        st.balloons()
        st.success("✅ Analysis Complete! We found your perfect job matches!")
        
        # Display resume statistics
        st.markdown("---")
        word_count = len(resume_text.split())
        char_count = len(resume_text)
        
        st.markdown(f"""
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{word_count}</div>
                    <div class="stat-label">Words Analyzed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(predictions)}</div>
                    <div class="stat-label">Job Matches Found</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{predictions[0][1]*100:.0f}%</div>
                    <div class="stat-label">Best Match Score</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Display predictions
        st.markdown("---")
        st.markdown('<h2 style="text-align: center; color: white; font-size: 2rem; margin: 2rem 0;">🏆 Your Top Job Role Matches</h2>', unsafe_allow_html=True)
        
        rank_icons = ['🥇', '🥈', '🥉']
        rank_labels = ["BEST MATCH", "STRONG MATCH", "GOOD MATCH"]
        gradient_colors = [
            ("linear-gradient(135deg, #667eea 0%, #764ba2 100%)", "#667eea"),
            ("linear-gradient(135deg, #f093fb 0%, #f5576c 100%)", "#f093fb"),
            ("linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)", "#4facfe")
        ]
        
        for idx, (category, probability) in enumerate(predictions):
            confidence_pct = probability * 100
            
            # Determine match quality
            if confidence_pct >= 70:
                match_quality = "Excellent Match"
            elif confidence_pct >= 50:
                match_quality = "Strong Match"
            elif confidence_pct >= 30:
                match_quality = "Good Match"
            else:
                match_quality = "Potential Match"
            
            # Add delay for staggered animation
            time.sleep(0.2)
            
            # Create prediction card with custom gradient
            gradient, _ = gradient_colors[idx]
            st.markdown(f"""
                <div class="prediction-card" style="background: {gradient}; animation-delay: {idx * 0.2}s;">
                    <span class="rank-badge">{rank_icons[idx]} {rank_labels[idx]}</span>
                    <div class="job-title">{category}</div>
                    <div class="match-indicator">{match_quality}</div>
                    <div class="confidence-score">{confidence_pct:.1f}%</div>
                    <div class="confidence-label">Match Confidence</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {confidence_pct}%;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        # Additional insights
        st.markdown("---")
        st.markdown("""
            <div class="upload-container">
                <h3 style="color: #2d3748; margin-bottom: 1rem;">💡 What's Next?</h3>
                <div style="color: #4a5568; line-height: 1.8;">
                    <p>✅ Review each job match and see which aligns best with your career goals</p>
                    <p>✅ Research companies hiring for these roles in your area</p>
                    <p>✅ Tailor your resume to highlight skills relevant to your top match</p>
                    <p>✅ Consider additional certifications to strengthen your profile</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Show extracted text info
        with st.expander("📝 View Analyzed Resume Text"):
            st.text_area(
                "Resume Content",
                resume_text[:2000] + ("..." if len(resume_text) > 2000 else ""),
                height=200,
                disabled=True,
                label_visibility="collapsed"
            )
            st.caption(f"Total characters: {char_count} | Total words: {word_count}")
        
    except FileNotFoundError as e:
        st.error(f"❌ {str(e)}")
        st.info("💡 Please train the model first by running: `python src/train.py`")
    
    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {str(e)}")
        st.info("💡 Please ensure the model is trained and all dependencies are installed.")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown('<div class="footer">', unsafe_allow_html=True)

# Footer Title
st.markdown('''
<h3 style="margin-bottom: 1rem; font-size: 1.5rem; color: white; text-align: center;">
    🎯 AI Resume Job Predictor
</h3>
''', unsafe_allow_html=True)

st.markdown('''
<p style="font-size: 1.1rem; margin-bottom: 1rem; color: white; text-align: center;">
    <strong>✨ Empowering Career Decisions with Artificial Intelligence ✨</strong>
</p>
''', unsafe_allow_html=True)

# Technology Stack Box
st.markdown('''
<div style="margin: 1.5rem auto; padding: 1.5rem; background: rgba(255, 255, 255, 0.1); border-radius: 15px; max-width: 800px;">
    <p style="margin-bottom: 0.5rem; color: white; text-align: center;">
        🔬 <strong>Technology Stack</strong>
    </p>
    <p style="font-size: 0.95rem; opacity: 0.9; color: white; text-align: center;">
        TF-IDF Vectorization (10K features) • Logistic Regression • Scikit-learn • Python • Streamlit
    </p>
</div>
''', unsafe_allow_html=True)

# Footer Links
st.markdown('''
<div style="margin-top: 1.5rem; display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
    <span style="color: white; font-weight: 600;">
        🧠 Machine Learning
    </span>
    <span style="color: white; font-weight: 600;">
        🔒 Privacy First
    </span>
    <span style="color: white; font-weight: 600;">
        ⚡ Real-time Analysis
    </span>
    <span style="color: white; font-weight: 600;">
        🎯 High Accuracy
    </span>
</div>
''', unsafe_allow_html=True)

# Copyright
st.markdown('''
<p style="margin-top: 1.5rem; font-size: 0.9rem; opacity: 0.8; color: white; text-align: center;">
    Built with ❤️ using Python, Scikit-learn, and Streamlit
</p>
<p style="font-size: 0.85rem; opacity: 0.7; margin-top: 0.5rem; color: white; text-align: center;">
    © 2026 AI Resume Job Predictor. All rights reserved.
</p>
''', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
