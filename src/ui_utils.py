import streamlit as st

def apply_custom_css():
    """Inject premium CSS styling and keyframe animations to the app."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@700&display=swap');
        
        /* Base typography */
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        
        /* Animations */
        @keyframes gradient-shift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        @keyframes float {
            0% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-12px) rotate(3deg); }
            100% { transform: translateY(0px) rotate(0deg); }
        }
        
        @keyframes pulse-glow {
            0% { box-shadow: 0 0 10px rgba(99, 102, 241, 0.2); }
            50% { box-shadow: 0 0 20px rgba(99, 102, 241, 0.6); }
            100% { box-shadow: 0 0 10px rgba(99, 102, 241, 0.2); }
        }
        
        @keyframes slide-up {
            from { transform: translateY(20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        
        .icon-3d {
            filter: drop-shadow(0 8px 12px rgba(0,0,0,0.4));
            width: 48px;
            height: 48px;
            display: inline-block;
        }
        .float-anim {
            animation: float 4s ease-in-out infinite;
        }
        .float-anim-delayed {
            animation: float 4s ease-in-out infinite 2s;
        }
        
        /* Page Load Animation */
        .block-container {
            animation: slide-up 0.8s cubic-bezier(0.16, 1, 0.3, 1);
        }

        /* Typography & Headers */
        h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif !important;
            letter-spacing: -0.5px;
        }
        
        /* Main Title Gradient */
        .title-glow {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(270deg, #3b82f6, #8b5cf6, #ec4899);
            background-size: 200% 200%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradient-shift 5s ease infinite;
            margin-bottom: 0.5rem;
        }
        
        .subtitle-glow {
            font-size: 2rem;
            font-weight: 800;
            background: linear-gradient(270deg, #10b981, #3b82f6);
            background-size: 200% 200%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradient-shift 5s ease infinite;
            margin-bottom: 1rem;
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.95) !important;
            border-right: 1px solid rgba(255,255,255,0.05);
            backdrop-filter: blur(20px);
        }
        
        .sidebar-logo {
            font-size: 3.5rem;
            text-align: center;
            animation: float 4s ease-in-out infinite;
            margin-bottom: -1rem;
            text-shadow: 0 10px 20px rgba(0,0,0,0.3);
        }
        
        /* Feature Cards (Columns on homepage) */
        .feature-card {
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 1.8rem;
            height: 100%;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .feature-card::before {
            content: '';
            position: absolute;
            top: 0; left: -100%;
            width: 50%; height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.05), transparent);
            transition: 0.5s;
        }
        
        .feature-card:hover::before {
            left: 100%;
        }
        
        .feature-card:hover {
            transform: translateY(-5px) scale(1.02);
            border-color: rgba(139, 92, 246, 0.5);
            box-shadow: 0 10px 30px -10px rgba(139, 92, 246, 0.3);
        }
        
        .feature-icon {
            font-size: 2.5rem;
            margin-bottom: 1rem;
            display: inline-block;
            filter: drop-shadow(0 4px 6px rgba(0,0,0,0.3));
        }
        
        /* Sleek Badge for Kaggle Link */
        .kaggle-badge {
            display: inline-flex;
            align-items: center;
            background: linear-gradient(135deg, #20BEFF 0%, #177199 100%);
            color: white !important;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
            text-decoration: none;
            transition: all 0.2s ease;
            box-shadow: 0 4px 15px rgba(32, 190, 255, 0.3);
            margin-top: 10px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .kaggle-badge:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(32, 190, 255, 0.5);
            background: linear-gradient(135deg, #37c5ff 0%, #1a81b0 100%);
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            transition: all 0.3s;
            animation: pulse-glow 3s infinite;
        }
        
        .stButton > button:hover {
            opacity: 1;
            transform: scale(1.05);
            border: none;
            color: white;
        }
        
        /* Horizontal Rule */
        hr {
            border: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            margin: 2.5rem 0;
        }
        
        /* Metric overriding */
        [data-testid="stMetricValue"] {
            font-size: 2.5rem !important;
            font-weight: 800 !important;
            background: linear-gradient(135deg, #60a5fa, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Render sidebar elements universally so they appear on all pages
    st.sidebar.markdown('<div class="sidebar-logo"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="url(#blue-grad)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="icon-3d float-anim"><defs><linearGradient id="blue-grad" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#3b82f6" /><stop offset="100%" stop-color="#8b5cf6" /></linearGradient></defs><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><path d="M3.27 6.96L12 12.01L20.73 6.96"></path><path d="M12 22.08V12"></path></svg></div>', unsafe_allow_html=True)
    st.sidebar.title("Skill-Gap Analyzer")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <p style='color: #94a3b8; font-size: 0.95rem;'>
        <b>Powered by global market data</b> from 5,700+ Tech Job Postings (2021-2026).
    </div>
    <div style="margin-top: 10px;">
        <a href="https://www.kaggle.com/datasets/kapturovalexander/ai-job-market-insights" target="_blank" class="kaggle-badge">
            View Kaggle Dataset
        </a>
        <br><br>
        """,
        unsafe_allow_html=True
    )
