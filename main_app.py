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

# CSS personnalisé pour la page d'accueil (VERSION SOBRE)
st.markdown("""
<style>
    /* Fond général sobre */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Container principal */
    .main-container {
        background: white;
        border-radius: 20px;
        padding: 3rem;
        margin: 2rem auto;
        max-width: 1200px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }
    
    /* Titre principal */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    /* Sous-titre */
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Cards de choix */
    .disease-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        border: 2px solid #e9ecef;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .disease-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        border-color: #3498db;
    }
    
    /* Icônes */
    .disease-icon {
        font-size: 4rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .disease-title {
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    
    .disease-description {
        color: #555;
        font-size: 0.95rem;
        line-height: 1.7;
        text-align: justify;
    }
    
    /* Boutons */
    .stButton > button {
        width: 100%;
        height: 55px;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        background: #3498db;
        border: none;
        color: white;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: #2980b9;
        transform: scale(1.02);
    }
    
    /* Footer */
    .footer-section {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 2rem;
        text-align: center;
    }
    
    .footer-col {
        color: #555;
    }
    
    .footer-col strong {
        color: #2c3e50;
        display: block;
        margin-bottom: 0.5rem;
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
    st.markdown('<p class="subtitle">Outils d\'analyse spatiotemporelle et de prédiction pour la préparation des interventions sanitaires</p>', unsafe_allow_html=True)
    
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
                <strong>Outil d'analyse et de prédiction avancée</strong><br><br>
                
                Cette application combine cartographie interactive, données environnementales et climatiques 
                pour identifier les zones à risque de transmission du paludisme.<br><br>
                
                <strong>Fonctionnalités clés :</strong><br>
                • <strong>Cartographie dynamique</strong> : Visualisez la répartition spatiale des cas avec popups enrichis 
                (cas, décès, population, densité, climat, environnement)<br>
                • <strong>Données démographiques</strong> : Intégration WorldPop (population totale, enfants 0-14 ans, 
                densité) pour calculer des taux d'incidence précis<br>
                • <strong>Analyse climatique</strong> : NASA POWER API pour température, précipitations et humidité 
                (facteurs clés de transmission vectorielle)<br>
                • <strong>Environnement</strong> : Zones inondables, altitude, distance aux cours d'eau<br>
                • <strong>Prédiction ML</strong> : Modèles de machine learning (Gradient Boosting, Random Forest) 
                avec validation croisée temporelle pour anticiper les épidémies 2-12 mois à l'avance<br>
                • <strong>Clustering géographique</strong> : Identification automatique de zones homogènes pour cibler les interventions<br><br>
                
                <em>Idéal pour planifier les campagnes de distribution de moustiquaires et les pulvérisations.</em>
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
                <strong>Surveillance et prédiction par semaines épidémiologiques</strong><br><br>
                
                Application spécialisée dans l'analyse des épidémies de rougeole avec suivi temporel précis 
                et évaluation des couvertures vaccinales.<br><br>
                
                <strong>Fonctionnalités clés :</strong><br>
                • <strong>Suivi hebdomadaire</strong> : Analyse par semaines épidémiologiques pour détecter rapidement 
                les flambées<br>
                • <strong>Couverture vaccinale</strong> : Intégration des taux de vaccination pour identifier les poches 
                de susceptibilité<br>
                • <strong>Données démographiques</strong> : Population par tranches d'âge (focus 0-35 ans) via WorldPop 
                pour calculer les taux d'attaque et le risque par groupe d'âge<br>
                • <strong>Prédiction avancée</strong> : Algorithmes Gradient Boosting et Random Forest optimisés 
                pour séries temporelles épidémiques (lags, moyennes mobiles, saisonnalité)<br>
                • <strong>Alertes précoces</strong> : Seuils épidémiques automatiques basés sur les moyennes historiques<br>
                • <strong>Multi-pays</strong> : Support Niger, Burkina Faso, Mali, Mauritanie avec données géographiques intégrées<br>
                • <strong>Pyramide des âges</strong> : Visualisation détaillée de la structure démographique (0-4, 5-9, 10-14... ans)<br><br>
                
                <em>Essentiel pour préparer les campagnes de vaccination réactive et estimer les besoins en vaccins.</em>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🦠 ANALYSER LA ROUGEOLE", key="btn_rougeole"):
            go_to_page('rougeole')
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown('<div class="footer-section">', unsafe_allow_html=True)
    
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        st.markdown('<div class="footer-col"><strong>📧 Contact</strong>youssoupha.mbodji@example.com</div>', unsafe_allow_html=True)
    
    with col_f2:
        st.markdown('<div class="footer-col"><strong>📖 Documentation</strong></div>', unsafe_allow_html=True)
        if st.button("📚 Manuel d'utilisation", key="btn_manuel_home"):
            go_to_page('manuel')
    
    with col_f3:
        st.markdown('<div class="footer-col"><strong>⚙️ Version</strong>v3.0 - Février 2026</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PAGE PALUDISME
# ============================================================

elif st.session_state.page == 'paludisme':
    
    # Bouton retour en sidebar
    with st.sidebar:
        if st.button("⬅️ Retour à l'accueil", key="back_palu"):
            go_to_page('home')
        
        if st.button("📚 Manuel d'utilisation", key="manuel_palu"):
            go_to_page('manuel')
        
        st.markdown("---")
        st.markdown("### 🦟 Module Paludisme")
        st.info("Vous êtes dans l'application d'analyse du paludisme")
    
    # Importer et exécuter l'app paludisme
    try:
        import app_paludisme
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
        
        if st.button("📚 Manuel d'utilisation", key="manuel_rougeole"):
            go_to_page('manuel')
        
        st.markdown("---")
        st.markdown("### 🦠 Module Rougeole")
        st.info("Vous êtes dans l'application d'analyse de la rougeole")
    
    # Importer et exécuter l'app rougeole
    try:
        import app_rougeole
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement de l'application Rougeole : {e}")
        st.info("Assurez-vous que le fichier `app_rougeole.py` existe dans le même dossier.")
        if st.button("Retour à l'accueil"):
            go_to_page('home')

# ============================================================
# PAGE MANUEL
# ============================================================

elif st.session_state.page == 'manuel':
    
    # Bouton retour en sidebar
    with st.sidebar:
        if st.button("⬅️ Retour à l'accueil", key="back_manuel"):
            go_to_page('home')
        
        st.markdown("---")
        st.markdown("### 📚 Manuel d'utilisation")
    
    # Importer et exécuter le manuel
    try:
        import app_manuel
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du manuel : {e}")
        st.info("Assurez-vous que le fichier `app_manuel.py` existe dans le même dossier.")
        if st.button("Retour à l'accueil"):
            go_to_page('home')
