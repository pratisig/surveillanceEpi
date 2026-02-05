"""
============================================================
APPLICATION PRINCIPALE - SURVEILLANCE ÉPIDÉMIOLOGIQUE
Réunit Paludisme et Rougeole en une seule plateforme
============================================================
"""

import streamlit as st

# Configuration de la page principale
st.set_page_config(
    page_title="Surveillance Épidémiologique",
    layout="wide",
    page_icon="🏥",
    initial_sidebar_state="collapsed"
)

# CSS personnalisé pour la page d'accueil
st.markdown("""
<style>
    /* Fond général */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Container principal */
    .main-container {
        background: white;
        border-radius: 20px;
        padding: 3rem;
        margin: 2rem auto;
        max-width: 1200px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    
    /* Titre principal */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    /* Sous-titre */
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 3rem;
    }
    
    /* Cards de choix */
    .disease-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 3px solid transparent;
    }
    
    .disease-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.2);
        border-color: #667eea;
    }
    
    /* Icônes */
    .disease-icon {
        font-size: 5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .disease-title {
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        color: #333;
        margin-bottom: 1rem;
    }
    
    .disease-description {
        text-align: center;
        color: #666;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Boutons */
    .stButton > button {
        width: 100%;
        height: 60px;
        font-size: 1.2rem;
        font-weight: 600;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Initialiser session state pour navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Fonction pour changer de page
def go_to_page(page_name):
    st.session_state.page = page_name
    st.rerun()

# ============================================================
# PAGE D'ACCUEIL
# ============================================================

if st.session_state.page == 'home':
    
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Titre principal
    st.markdown('<h1 class="main-title">🏥 Plateforme de Surveillance Épidémiologique</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Outils d\'analyse et de prédiction pour la préparation des interventions</p>', unsafe_allow_html=True)
    
    # Ligne horizontale
    st.markdown("---")
    
    # Deux colonnes pour les choix
    col1, col2 = st.columns(2, gap="large")
    
    # COLONNE PALUDISME
    with col1:
        st.markdown("""
        <div class="disease-card">
            <div class="disease-icon">🦟</div>
            <div class="disease-title">Paludisme</div>
            <div class="disease-description">
                <strong>Analyse spatiotemporelle avancée</strong><br>
                • Cartographie interactive des cas<br>
                • Données climatiques (NASA POWER API)<br>
                • Modélisation prédictive ML<br>
                • Analyse environnementale (inondations, altitude)<br>
                • Clustering géographique
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🦟 ANALYSER LE PALUDISME", key="btn_palu"):
            go_to_page('paludisme')
    
    # COLONNE ROUGEOLE
    with col2:
        st.markdown("""
        <div class="disease-card">
            <div class="disease-icon">🦠</div>
            <div class="disease-title">Rougeole</div>
            <div class="disease-description">
                <strong>Surveillance et prédiction multi-pays</strong><br>
                • Analyse par semaines épidémiologiques<br>
                • Données démographiques (WorldPop)<br>
                • Gradient Boosting & Random Forest<br>
                • Couverture vaccinale<br>
                • Alertes précoces
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🦠 ANALYSER LA ROUGEOLE", key="btn_rougeole"):
            go_to_page('rougeole')
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        st.markdown("**📧 Contact**")
        st.markdown("youssoupha.mbodji@example.com")
    
    with col_f2:
        st.markdown("**📖 Documentation**")
        st.markdown("[Manuel utilisateur](#)")
    
    with col_f3:
        st.markdown("**⚙️ Version**")
        st.markdown("v3.0 - 2026")

# ============================================================
# PAGE PALUDISME
# ============================================================

elif st.session_state.page == 'paludisme':
    
    # Bouton retour en sidebar
    with st.sidebar:
        if st.button("⬅️ Retour à l'accueil", key="back_palu"):
            go_to_page('home')
        
        st.markdown("---")
        st.markdown("### 🦟 Module Paludisme")
        st.info("Vous êtes dans l'application d'analyse du paludisme")
    
    # Importer et exécuter l'app paludisme
    try:
        import app_paludisme
        # Le code de app_paludisme.py s'exécute automatiquement
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement de l'application Paludisme : {e}")
        st.info("Assurez-vous que le fichier `app_paludisme.py` existe dans le même dossier.")
        if st.button("Retour à l'accueil"):
            go_to_page('home')

# ============================================================
# PAGE ROUGEOLE
# ============================================================

elif st.session_state.page == 'rougeole':
    
    # Bouton retour en sidebar
    with st.sidebar:
        if st.button("⬅️ Retour à l'accueil", key="back_rougeole"):
            go_to_page('home')
        
        st.markdown("---")
        st.markdown("### 🦠 Module Rougeole")
        st.info("Vous êtes dans l'application d'analyse de la rougeole")
    
    # Importer et exécuter l'app rougeole
    try:
        import app_rougeole
        # Le code de app_rougeole.py s'exécute automatiquement
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement de l'application Rougeole : {e}")
        st.info("Assurez-vous que le fichier `app_rougeole.py` existe dans le même dossier.")
        if st.button("Retour à l'accueil"):
            go_to_page('home')
