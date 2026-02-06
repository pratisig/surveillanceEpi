"""
============================================================
APPLICATION PRINCIPALE - PLATEFORME SURVEILLANCE ÉPIDÉMIOLOGIQUE
Point d'entrée avec menu de navigation
============================================================
"""

import streamlit as st

# Configuration de la page (DOIT être la première commande Streamlit)
st.set_page_config(
    page_title="Plateforme Surveillance Épidémiologique",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé SIMPLE (sans branding complexe)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
    }
    
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .info-card h3 {
        margin-top: 0;
        font-size: 2rem;
    }
    
    .info-card ul {
        list-style: none;
        padding-left: 0;
        line-height: 1.8;
    }
    
    .stButton > button {
        width: 100%;
        background: white;
        color: #667eea;
        border: 2px solid #667eea;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 10px;
    }
    
    .stButton > button:hover {
        background: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# En-tête
st.markdown('<h1 class="main-header">🏥 Plateforme de Surveillance Épidémiologique</h1>', unsafe_allow_html=True)
st.markdown("---")

# Navigation dans la sidebar
with st.sidebar:
    st.header("🧭 Navigation")
    page = st.selectbox(
        "Choisir une application",
        ["Accueil", "Paludisme", "Rougeole", "Manuel"]
    )

# Routage selon la page sélectionnée
if page == "Paludisme":
    import app_paludisme
    
elif page == "Rougeole":
    import app_rougeole
    
elif page == "Manuel":
    import app_manuel
    
else:  # Page d'accueil
    st.markdown("## Choisissez votre module d'analyse")
    st.info("Utilisez le menu dans la barre latérale pour accéder aux applications")
    
    # Cartes des applications
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🦟 Paludisme</h3>
            <h4>Outil d'analyse et de prédiction avancée</h4>
            <p>
                Cette application combine cartographie interactive, données environnementales et climatiques 
                pour identifier les zones à risque de transmission du paludisme.
            </p>
            <p><strong>Fonctionnalités clés :</strong></p>
            <ul>
                <li>• Cartographie dynamique</li>
                <li>• Données démographiques (WorldPop)</li>
                <li>• Analyse climatique (NASA POWER API)</li>
                <li>• Environnement (inondations, altitude, rivières)</li>
                <li>• Prédiction ML (Gradient Boosting, Random Forest)</li>
                <li>• Clustering géographique</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🦠 Rougeole</h3>
            <h4>Surveillance et prédiction par semaines épidémiologiques</h4>
            <p>
                Application spécialisée dans l'analyse des épidémies de rougeole avec suivi temporel précis 
                et évaluation des couvertures vaccinales.
            </p>
            <p><strong>Fonctionnalités clés :</strong></p>
            <ul>
                <li>• Suivi hebdomadaire</li>
                <li>• Couverture vaccinale</li>
                <li>• Données démographiques (WorldPop)</li>
                <li>• Prédiction avancée</li>
                <li>• Alertes précoces</li>
                <li>• Multi-pays (Niger, Burkina Faso, Mali, Mauritanie)</li>
                <li>• Pyramide des âges</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Documentation
    st.markdown("## 📚 Documentation")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📖 Manuel d'utilisation")
        st.write("Guide détaillé pas-à-pas")
    
    with col2:
        st.markdown("### 🔬 Méthodologie")
        st.write("Algorithmes et validation")
    
    with col3:
        st.markdown("### 💡 Glossaire")
        st.write("Définitions des variables")
    
    st.markdown("---")
    
    # Caractéristiques techniques
    st.markdown("## ⚙️ Caractéristiques Techniques")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🗺️ Cartographie")
        st.write("- Folium interactif")
        st.write("- Popups enrichis")
        st.write("- Couches multiples")
        st.write("- Export GeoJSON")
    
    with col2:
        st.markdown("### 🤖 Machine Learning")
        st.write("- Gradient Boosting")
        st.write("- Random Forest")
        st.write("- Validation temporelle")
        st.write("- R² > 0.80 typique")
    
    with col3:
        st.markdown("### 📊 Sources Données")
        st.write("- NASA POWER API")
        st.write("- WorldPop (GEE)")
        st.write("- Rasters environnement")
        st.write("- Linelists épidémio")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#7f8c8d; padding:2rem;">
        <p style="font-size:1.1rem;"><strong>Développé par Youssoupha MBODJI</strong></p>
        <p>📧 Email : youssoupha.mbodji@example.com</p>
        <p style="margin-top:1rem; font-size:0.9rem;">Version 3.0 | © 2026</p>
        <p style="font-size:0.9rem;">Plateforme de surveillance épidémiologique pour l'Afrique de l'Ouest</p>
    </div>
    """, unsafe_allow_html=True)
