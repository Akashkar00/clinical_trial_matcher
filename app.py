import streamlit as st
import tempfile
import os
from pipeline.graph import pipeline
from models.patient_profile import PatientProfile

st.set_page_config(page_title="Clinical Trial Matcher", page_icon="🧬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    line-height: 1.6;
}
.stApp {
    background: #f8fffe;
    background-image: radial-gradient(circle at 20% 50%, rgba(13,148,136,0.03) 0%, transparent 50%),
                      radial-gradient(circle at 80% 20%, rgba(59,130,246,0.03) 0%, transparent 50%);
}

/* Animations */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(13,148,136,0.4); }
    50% { box-shadow: 0 0 0 8px rgba(13,148,136,0); }
}
@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
.fade-in { animation: fadeInUp 0.6s ease-out forwards; }
.fade-in-delay { animation: fadeInUp 0.6s ease-out 0.2s forwards; opacity: 0; }

/* Header */
.hero-header {
    background: linear-gradient(135deg, #0d9488 0%, #0891b2 50%, #3b82f6 100%);
    padding: 2.5rem 3rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(13,148,136,0.25);
    animation: fadeInUp 0.6s ease-out;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M30 5 L30 55 M5 30 L55 30' stroke='rgba(255,255,255,0.06)' stroke-width='1' fill='none'/%3E%3Ccircle cx='30' cy='30' r='8' stroke='rgba(255,255,255,0.04)' fill='none'/%3E%3C/svg%3E");
    pointer-events: none;
}
.hero-header h1 {
    color: #ffffff;
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    position: relative;
}
.hero-header p {
    color: #ffffff;
    font-size: 1.1rem;
    font-weight: 500;
    margin: 0.6rem 0 0 0;
    opacity: 0.95;
    position: relative;
}
.hero-icons {
    position: absolute;
    right: 3rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 3.5rem;
    opacity: 0.2;
}

/* Upload area */
div[data-testid="stFileUploader"] section {
    border: none !important;
    padding: 0 !important;
}
div[data-testid="stFileUploader"] section > div:first-child {
    display: none !important;
}

/* Info card */
.info-card {
    background: linear-gradient(135deg, #ffffff, #f0fdfa);
    border: 1px solid #ccfbf1;
    border-radius: 14px;
    padding: 1.4rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.info-card h4 {
    color: #0d9488;
    font-weight: 700;
    font-size: 1rem;
    margin: 0 0 0.8rem 0;
}
.info-card .step {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.5rem;
    color: #2d3748;
    font-size: 0.9rem;
    font-weight: 500;
}
.info-card .step-num {
    background: #0d9488;
    color: white;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    flex-shrink: 0;
}

/* Button override */
.stButton > button[kind="primary"] {
    background: linear-gradient(120deg, #0d9488, #0891b2) !important;
    border: none !important;
    font-weight: 700 !important;
    padding: 0.75rem 2rem !important;
    font-size: 1.05rem !important;
    border-radius: 10px !important;
    letter-spacing: 0.3px !important;
    animation: pulse 2s infinite;
    transition: transform 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
    transform: scale(1.02) !important;
    animation: none;
}

/* Sidebar */
div[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 3px solid #0d9488;
}
div[data-testid="stSidebar"] * {
    color: #1a1a2e !important;
}

.sidebar-title {
    color: #0d9488 !important;
    font-size: 1.3rem;
    font-weight: 800;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e0f2f1;
    margin-bottom: 1rem;
}

.profile-item {
    background: #f0fdfa;
    border-left: 3px solid #0d9488;
    border-radius: 0 10px 10px 0;
    padding: 0.8rem 1rem;
    margin-bottom: 0.7rem;
    transition: transform 0.2s;
}
.profile-item:hover {
    transform: translateX(3px);
}
.profile-item .label {
    font-size: 0.75rem;
    font-weight: 700;
    color: #0d9488 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.2rem;
}
.profile-item .value {
    font-size: 0.95rem;
    font-weight: 600;
    color: #1a1a2e !important;
}

/* Stats */
.stats-row {
    display: flex;
    gap: 1rem;
    margin: 1.5rem 0;
    animation: fadeInUp 0.6s ease-out 0.3s forwards;
    opacity: 0;
}
.stat-card {
    flex: 1;
    background: #ffffff;
    border-radius: 14px;
    padding: 1.4rem;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    border-top: 4px solid;
    transition: transform 0.2s, box-shadow 0.2s;
}
.stat-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.1);
}
.stat-card.match { border-top-color: #10b981; }
.stat-card.partial { border-top-color: #f59e0b; }
.stat-card.no { border-top-color: #ef4444; }
.stat-card .icon { font-size: 1.5rem; margin-bottom: 0.3rem; }
.stat-card .number {
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1;
}
.stat-card.match .number { color: #10b981; }
.stat-card.partial .number { color: #f59e0b; }
.stat-card.no .number { color: #ef4444; }
.stat-card .label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 0.3rem;
}

/* Trial cards */
.trial-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    border-left: 5px solid;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    transition: transform 0.2s, box-shadow 0.2s;
}
.trial-card:hover {
    transform: translateX(4px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
}
.trial-card.match { border-left-color: #10b981; }
.trial-card.partial { border-left-color: #f59e0b; }
.trial-card.no { border-left-color: #ef4444; }

.verdict-pill {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.8rem;
    letter-spacing: 0.5px;
    color: #ffffff;
}
.pill-match { background: #10b981; }
.pill-partial { background: #f59e0b; }
.pill-no { background: #ef4444; }

.score-chip {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.8rem;
    margin-left: 0.5rem;
}
.chip-match { background: #d1fae5; color: #065f46; }
.chip-partial { background: #fef3c7; color: #92400e; }
.chip-no { background: #fee2e2; color: #991b1b; }

.trial-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #1a1a2e;
    margin: 0.6rem 0 0.4rem 0;
}
.trial-reason {
    font-size: 0.9rem;
    color: #4b5563;
    font-weight: 500;
    line-height: 1.5;
}
.trial-nct {
    font-size: 0.8rem;
    color: #6b7280;
    font-weight: 600;
}

/* Section header */
.section-title {
    color: #1a1a2e;
    font-size: 1.4rem;
    font-weight: 800;
    margin: 1.5rem 0 1rem 0;
}

/* Loading shimmer */
.loading-bar {
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(90deg, #e0f2fe 25%, #99f6e4 50%, #e0f2fe 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
}

/* TEXT VISIBILITY FIXES — light theme forced via config.toml */
/* Ensure no dark backgrounds leak through on any element */
div[data-testid="stExpander"] details,
div[data-testid="stExpander"] details summary,
div[data-testid="stExpander"] details div[data-testid="stExpanderDetails"],
div[data-testid="stMetric"],
div[data-testid="stMetricLabel"],
div[data-testid="stMetricValue"],
div[data-testid="metric-container"],
div[data-testid="column"],
div[data-testid="stVerticalBlock"],
div[data-testid="stHorizontalBlock"],
div[data-testid="stStatusWidget"],
div[data-testid="stFileUploader"] section {
    background-color: transparent !important;
    background: transparent !important;
}

/* Force dark text globally */
.stApp p, .stApp span, .stApp label,
div[data-testid="stMetricLabel"] label,
div[data-testid="stMetricLabel"] div,
div[data-testid="stMetricValue"] div,
div[data-testid="stExpander"] summary span,
div[data-testid="stExpander"] summary span p,
.stMarkdown p, .stMarkdown span,
div[data-testid="stStatusWidget"] p,
div[data-testid="stStatusWidget"] span {
    color: #1f2937 !important;
}

/* Metric label lighter */
div[data-testid="stMetricLabel"] label {
    color: #6b7280 !important;
    font-weight: 600 !important;
}
div[data-testid="stMetricValue"] div {
    color: #111827 !important;
    font-weight: 700 !important;
}

/* Sidebar */
div[data-testid="stSidebar"],
div[data-testid="stSidebar"] > div:first-child {
    background: #ffffff !important;
}

/* Link button */
.stLinkButton a {
    color: #0d9488 !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# Hero header
st.markdown("""
<div class="hero-header">
    <div class="hero-icons">🧬 💊 🔬</div>
    <h1>🏥 Clinical Trial Matcher</h1>
    <p>AI-powered patient-to-trial matching using semantic search & LLM clinical reasoning</p>
</div>
""", unsafe_allow_html=True)

# Upload section
col_upload, col_info = st.columns([3, 2])
with col_upload:
    uploaded_file = st.file_uploader("📄 Upload Patient Medical Report", type=["pdf"], label_visibility="visible")

with col_info:
    st.markdown("""
    <div class="info-card">
        <h4>🔬 How It Works</h4>
        <div class="step"><span class="step-num">1</span> Extract patient profile from PDF</div>
        <div class="step"><span class="step-num">2</span> Fetch trials from ClinicalTrials.gov</div>
        <div class="step"><span class="step-num">3</span> Semantic similarity retrieval</div>
        <div class="step"><span class="step-num">4</span> LLM eligibility scoring</div>
    </div>
    """, unsafe_allow_html=True)

# Action button
if uploaded_file:
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("🔍 Find Matching Trials", type="primary", use_container_width=True)
else:
    run = False

if uploaded_file and run:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        with st.status("🧪 Running AI Pipeline...", expanded=True) as status:
            st.write("📄 Extracting patient profile from PDF...")
            st.write("🌐 Fetching clinical trials...")
            st.write("🧠 Embedding & semantic retrieval...")
            st.write("⚖️ LLM eligibility scoring...")
            result = pipeline.invoke({
                "pdf_path": tmp_path,
                "raw_text": None,
                "patient_profile": None,
                "fetched_trials": None,
                "chunks_stored": None,
                "retrieved_chunks": None,
                "scored_trials": None,
                "error": None,
                "retry_count": 0
            })
            status.update(label="✅ Analysis Complete!", state="complete")

        os.unlink(tmp_path)

        if result.get("error"):
            st.error(f"❌ Pipeline failed: {result['error']}")
        else:
            profile = result["patient_profile"]
            trials = result.get("scored_trials", [])

            # Sidebar
            with st.sidebar:
                st.markdown('<div class="sidebar-title">👤 Patient Profile</div>', unsafe_allow_html=True)

                fields = [
                    ("🏷️ Diagnosis", profile.diagnosis),
                    ("🎂 Age", str(profile.age)),
                    ("⚧ Gender", profile.gender.value),
                ]
                if profile.stage:
                    fields.append(("📊 Stage", profile.stage))
                if profile.biomarkers:
                    fields.append(("🧬 Biomarkers", " • ".join(profile.biomarkers)))
                if profile.prior_treatments:
                    fields.append(("💊 Prior Treatments", " • ".join(profile.prior_treatments)))
                if profile.current_status:
                    fields.append(("📋 Status", profile.current_status))
                if profile.ecog_status is not None:
                    fields.append(("🏃 ECOG Score", str(profile.ecog_status)))

                for label, value in fields:
                    st.markdown(f"""
                    <div class="profile-item">
                        <div class="label">{label}</div>
                        <div class="value">{value}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Stats
            n_match = sum(1 for t in trials if t["match_type"] == "MATCH")
            n_partial = sum(1 for t in trials if t["match_type"] == "PARTIAL")
            n_no = sum(1 for t in trials if t["match_type"] == "NO")

            st.markdown(f"""
            <div class="stats-row">
                <div class="stat-card match">
                    <div class="icon">✅</div>
                    <div class="number">{n_match}</div>
                    <div class="label">Matches</div>
                </div>
                <div class="stat-card partial">
                    <div class="icon">⚠️</div>
                    <div class="number">{n_partial}</div>
                    <div class="label">Partial</div>
                </div>
                <div class="stat-card no">
                    <div class="icon">❌</div>
                    <div class="number">{n_no}</div>
                    <div class="label">No Match</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Results
            st.markdown('<div class="section-title">🎯 Trial Results</div>', unsafe_allow_html=True)

            for i, t in enumerate(trials):
                match_type = t["match_type"]
                score = t["score"]
                css = match_type.lower()

                if match_type == "MATCH":
                    pill_class, chip_class = "pill-match", "chip-match"
                elif match_type == "PARTIAL":
                    pill_class, chip_class = "pill-partial", "chip-partial"
                else:
                    pill_class, chip_class = "pill-no", "chip-no"

                with st.expander(f"{'✅' if match_type=='MATCH' else '⚠️' if match_type=='PARTIAL' else '❌'} **{t['nct_id']}** — {t['title'][:65]}", expanded=(match_type == "MATCH")):
                    st.markdown(f"""
                    <span class="verdict-pill {pill_class}">{match_type}</span>
                    <span class="score-chip {chip_class}">Score: {score:.2f}</span>
                    """, unsafe_allow_html=True)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Verdict", match_type)
                    col2.metric("Confidence", f"{score:.0%}")
                    col3.metric("Phase", t.get("phase") or "N/A")

                    st.markdown(f'<div class="trial-reason"><strong>💡 Reasoning:</strong> {t["reason"]}</div>', unsafe_allow_html=True)
                    st.link_button("🔗 View on ClinicalTrials.gov", f"https://clinicaltrials.gov/study/{t['nct_id']}")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
