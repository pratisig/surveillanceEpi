"""
============================================================
APPLICATION PRINCIPALE - PLATEFORME SURVEILLANCE ÉPIDÉMIOLOGIQUE
Développée pour Médecins Sans Frontières (MSF)
============================================================
"""

import streamlit as st

# Configuration de la page (DOIT être la première commande Streamlit)
st.set_page_config(
    page_title="MSF - Surveillance Épidémiologique",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé avec branding MSF
st.markdown("""
<style>
    /* Couleurs MSF */
    :root {
        --msf-red: #E4032E;
        --msf-dark-red: #B30024;
        --msf-light-red: #FF1744;
        --msf-grey: #58595B;
        --msf-light-grey: #F5F5F5;
    }
    
    /* Conteneur principal */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    /* Cartes application */
    .app-card {
        background: linear-gradient(135deg, #E4032E 0%, #B30024 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 10px 30px rgba(228, 3, 46, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-left: 5px solid white;
    }
    
    .app-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(228, 3, 46, 0.4);
    }
    
    .app-card-rougeole {
        background: linear-gradient(135deg, #58595B 0%, #3a3b3d 100%);
        border-left: 5px solid #E4032E;
    }
    
    .app-card h3 {
        margin-top: 0;
        font-size: 2rem;
        font-weight: bold;
    }
    
    .app-card h4 {
        margin-top: 0.5rem;
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: normal;
    }
    
    .app-card ul {
        list-style: none;
        padding-left: 0;
        line-height: 1.8;
    }
    
    .app-card li {
        margin: 0.5rem 0;
    }
    
    .app-card strong {
        font-weight: 600;
    }
    
    .app-card em {
        display: block;
        margin-top: 1rem;
        font-style: italic;
        opacity: 0.9;
        border-top: 1px solid rgba(255,255,255,0.3);
        padding-top: 1rem;
    }
    
    /* Bannière en-tête MSF */
    .header-banner {
        text-align: center;
        padding: 3rem 1rem;
        background: linear-gradient(135deg, #E4032E 0%, #B30024 100%);
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        border-bottom: 5px solid white;
    }
    
    .header-banner h1 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: bold;
        text-transform: uppercase;
    }
    
    .header-banner p {
        font-size: 1.2rem;
        margin: 1rem 0 0 0;
        opacity: 0.95;
    }
    
    .msf-logo-text {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        letter-spacing: 2px;
    }
    
    /* Boutons de lancement en haut */
    .launch-button-container {
        text-align: center;
        margin: 1.5rem 0;
    }
    
    /* Surcharge style boutons Streamlit avec couleurs MSF */
    div[data-testid="column"] > div > div > div > button {
        width: 100% !important;
        background: white !important;
        color: #E4032E !important;
        border: 3px solid #E4032E !important;
        padding: 1rem 2rem !important;
        font-size: 1.2rem !important;
        font-weight: bold !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(228, 3, 46, 0.3) !important;
        text-transform: uppercase;
    }
    
    div[data-testid="column"] > div > div > div > button:hover {
        background: #E4032E !important;
        color: white !important;
        transform: scale(1.05) !important;
        box-shadow: 0 6px 20px rgba(228, 3, 46, 0.5) !important;
    }
    
    /* Boutons documentation */
    .stButton > button {
        background: white !important;
        color: #E4032E !important;
        border: 2px solid #E4032E !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background: #E4032E !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# En-tête principal MSF
st.markdown("""
<div class="header-banner">
    <div class="msf-logo-text">MÉDECINS SANS FRONTIÈRES</div>
    <h1>🏥 Plateforme de Surveillance Épidémiologique</h1>
    <p>Outils d'analyse, cartographie et prédiction pour le paludisme et la rougeole</p>
    <p style="font-size:0.9rem; margin-top:0.5rem; opacity:0.8;">Afrique de l'Ouest | Operational Research</p>
</div>
""", unsafe_allow_html=True)

# Introduction
st.markdown("""
<div style="text-align:center; margin:2rem 0;">
    <h2 style="color:#E4032E;">Choisissez votre module d'analyse</h2>
    <p style="font-size:1.1rem; color:#58595B;">
        Deux applications spécialisées pour une surveillance épidémiologique optimale
    </p>
</div>
""", unsafe_allow_html=True)

# Cartes des applications
col1, col2 = st.columns(2)

with col1:
    # BOUTON EN HAUT
    st.markdown('<div class="launch-button-container">', unsafe_allow_html=True)
    if st.button("🦟 LANCER L'APPLICATION PALUDISME", key="btn_palu", use_container_width=True):
        # CORRECTION ICI - Utilisez le nom exact du fichier
        st.switch_page("pages/app_paludisme.py")  # OU le nom exact de votre fichier
    st.markdown('</div>', unsafe_allow_html=True)
    
    # CARTE DESCRIPTIVE
    st.markdown("""
    <div class="app-card">
        <h3>🦟 Paludisme</h3>
        <h4>Outil d'analyse et de prédiction avancée</h4>
        <p>
            Cette application combine cartographie interactive, données environnementales et climatiques 
            pour identifier les zones à risque de transmission du paludisme.
        </p>
        <p><strong>Fonctionnalités clés :</strong></p>
        <ul>
            <li>• <strong>Cartographie dynamique</strong> : Visualisez la répartition spatiale des cas avec popups enrichis 
            (cas, décès, population, densité, climat, environnement)</li>
            <li>• <strong>Données démographiques</strong> : Intégration WorldPop (population totale, enfants 0-14 ans, 
            densité) pour calculer des taux d'incidence précis</li>
            <li>• <strong>Analyse climatique</strong> : NASA POWER API pour température, précipitations et humidité 
            (facteurs clés de transmission vectorielle)</li>
            <li>• <strong>Environnement</strong> : Zones inondables, altitude, distance aux cours d'eau</li>
            <li>• <strong>Prédiction ML</strong> : Modèles de machine learning (Gradient Boosting, Random Forest) 
            avec validation croisée temporelle pour anticiper les épidémies 2-12 mois à l'avance</li>
            <li>• <strong>Clustering géographique</strong> : Identification automatique de zones homogènes pour cibler les interventions</li>
        </ul>
        <em>Idéal pour planifier les campagnes de distribution de moustiquaires et les pulvérisations.</em>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # BOUTON EN HAUT
    st.markdown('<div class="launch-button-container">', unsafe_allow_html=True)
    if st.button("🦠 LANCER L'APPLICATION ROUGEOLE", key="btn_rougeole", use_container_width=True):
        # CORRECTION ICI - Utilisez le nom exact du fichier
        st.switch_page("pages/app_rougeole.py")  # OU le nom exact de votre fichier
    st.markdown('</div>', unsafe_allow_html=True)
    
    # CARTE DESCRIPTIVE
    st.markdown("""
    <div class="app-card app-card-rougeole">
        <h3>🦠 Rougeole</h3>
        <h4>Surveillance et prédiction par semaines épidémiologiques</h4>
        <p>
            Application spécialisée dans l'analyse des épidémies de rougeole avec suivi temporel précis 
            et évaluation des couvertures vaccinales.
        </p>
        <p><strong>Fonctionnalités clés :</strong></p>
        <ul>
            <li>• <strong>Suivi hebdomadaire</strong> : Analyse par semaines épidémiologiques pour détecter rapidement 
            les flambées</li>
            <li>• <strong>Couverture vaccinale</strong> : Intégration des taux de vaccination pour identifier les poches 
            de susceptibilité</li>
            <li>• <strong>Données démographiques</strong> : Population par tranches d'âge (focus 0-35 ans) via WorldPop 
            pour calculer les taux d'attaque et le risque par groupe d'âge</li>
            <li>• <strong>Prédiction avancée</strong> : Algorithmes Gradient Boosting et Random Forest optimisés 
            pour séries temporelles épidémiques (lags, moyennes mobiles, saisonnalité)</li>
            <li>• <strong>Alertes précoces</strong> : Seuils épidémiques automatiques basés sur les moyennes historiques</li>
            <li>• <strong>Multi-pays</strong> : Support Niger, Burkina Faso, Mali, Mauritanie avec données géographiques intégrées</li>
            <li>• <strong>Pyramide des âges</strong> : Visualisation détaillée de la structure démographique (0-4, 5-9, 10-14... ans)</li>
        </ul>
        <em>Essentiel pour préparer les campagnes de vaccination réactive et estimer les besoins en vaccins.</em>
    </div>
    """, unsafe_allow_html=True)

# Séparateur
st.markdown("---")

# Section Documentation
st.markdown("""
<div style="background:#F5F5F5; padding:2rem; border-radius:15px; margin:2rem 0; border-left:5px solid #E4032E;">
    <h2 style="text-align:center; color:#E4032E;">📚 Documentation et Ressources</h2>
    <p style="text-align:center; font-size:1.1rem; color:#58595B;">
        Guides complets, méthodologies et bonnes pratiques
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="background:white; padding:1.5rem; border-radius:10px; box-shadow:0 2px 10px rgba(228,3,46,0.1); border-top:3px solid #E4032E;">
        <h3 style="color:#E4032E;">📖 Manuel d'utilisation</h3>
        <p style="color:#58595B;">Guide détaillé pas-à-pas pour utiliser chaque module, interpréter les résultats et optimiser vos analyses.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("📖 Consulter le manuel", key="btn_manuel", use_container_width=True):
        st.switch_page("pages/app_manuel.py")  # CORRECTION ICI

with col2:
    st.markdown("""
    <div style="background:white; padding:1.5rem; border-radius:10px; box-shadow:0 2px 10px rgba(228,3,46,0.1); border-top:3px solid #E4032E;">
        <h3 style="color:#E4032E;">🔬 Méthodologie</h3>
        <p style="color:#58595B;">Explication des algorithmes de machine learning, validation croisée temporelle et feature engineering.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🔬 Voir la méthodologie", key="btn_methodo", use_container_width=True):
        st.switch_page("pages/app_manuel.py")  # CORRECTION ICI

with col3:
    st.markdown("""
    <div style="background:white; padding:1.5rem; border-radius:10px; box-shadow:0 2px 10px rgba(228,3,46,0.1); border-top:3px solid #E4032E;">
        <h3 style="color:#E4032E;">💡 Glossaire</h3>
        <p style="color:#58595B;">Définitions des variables (lags, moyennes mobiles, ACP, clustering spatial, etc.) et concepts clés.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("💡 Accéder au glossaire", key="btn_glossaire", use_container_width=True):
        st.switch_page("pages/app_manuel.py")  # CORRECTION ICI

# Séparateur
st.markdown("---")

# Section Caractéristiques techniques
st.markdown("""
<div style="text-align:center; margin:2rem 0;">
    <h2 style="color:#E4032E;">⚙️ Caractéristiques Techniques</h2>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="text-align:center; padding:1rem; background:white; border-radius:10px; box-shadow:0 2px 10px rgba(228,3,46,0.1);">
        <h3 style="color:#E4032E;">🗺️ Cartographie</h3>
        <ul style="list-style:none; padding:0; color:#58595B;">
            <li>• Folium interactif</li>
            <li>• Popups enrichis</li>
            <li>• Couches multiples</li>
            <li>• Export GeoJSON</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align:center; padding:1rem; background:white; border-radius:10px; box-shadow:0 2px 10px rgba(228,3,46,0.1);">
        <h3 style="color:#E4032E;">🤖 Machine Learning</h3>
        <ul style="list-style:none; padding:0; color:#58595B;">
            <li>• Gradient Boosting</li>
            <li>• Random Forest</li>
            <li>• Validation temporelle</li>
            <li>• R² > 0.80 typique</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="text-align:center; padding:1rem; background:white; border-radius:10px; box-shadow:0 2px 10px rgba(228,3,46,0.1);">
        <h3 style="color:#E4032E;">📊 Sources Données</h3>
        <ul style="list-style:none; padding:0; color:#58595B;">
            <li>• NASA POWER API</li>
            <li>• WorldPop (GEE)</li>
            <li>• Rasters environnement</li>
            <li>• Linelists épidémio</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer MSF
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#58595B; padding:2rem; background:#F5F5F5; border-radius:10px; border-top:3px solid #E4032E;">
    <p style="font-size:1.3rem; font-weight:bold; color:#E4032E; margin-bottom:1rem;">MÉDECINS SANS FRONTIÈRES</p>
    <p style="font-size:1.1rem;"><strong>Développé par Youssoupha MBODJI</strong></p>
    <p>📧 Email : youssoupha.mbodji@example.com</p>
    <p style="margin-top:1rem; font-size:0.9rem;">Version 3.0 | © 2026 MSF</p>
    <p style="font-size:0.9rem;">Plateforme de surveillance épidémiologique pour l'Afrique de l'Ouest</p>
    <p style="font-size:0.85rem; margin-top:1rem; font-style:italic;">
        "Bringing medical assistance to people affected by conflict, epidemics, disasters, or exclusion from healthcare"
    </p>
</div>
""", unsafe_allow_html=True)
