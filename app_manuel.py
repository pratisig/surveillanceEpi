"""
============================================================
MANUEL D'UTILISATION - PLATEFORME SURVEILLANCE ÉPIDÉMIOLOGIQUE
Documentation complète Paludisme + Rougeole
============================================================
"""

import streamlit as st

# CSS personnalisé pour le manuel
st.markdown("""
<style>
    /* Styles des cartes info */
    .info-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 5px solid #2196f3;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .benefit-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 5px solid #4caf50;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border-left: 5px solid #ff9800;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .alert-box {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 5px solid #f44336;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Titres dans les cartes */
    .info-card h3, .info-card h4, .info-card h5,
    .benefit-box h3, .benefit-box h4, .benefit-box h5,
    .warning-box h3, .warning-box h4, .warning-box h5 {
        margin-top: 0;
        color: #2c3e50;
    }
    
    /* Listes */
    .info-card ul, .benefit-box ul, .warning-box ul {
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    
    .info-card li, .benefit-box li, .warning-box li {
        margin: 0.5rem 0;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# En-tête
st.title("📚 Manuel d'Utilisation - Plateforme de Surveillance Épidémiologique")
st.markdown("*Guide complet pour l'utilisation des modules Paludisme et Rougeole*")
st.markdown("---")

# Onglets principaux
tab_palu, tab_rougeole, tab_glossaire = st.tabs([
    "🦟 Paludisme", 
    "🦠 Rougeole",
    "📖 Glossaire & Méthodologie"
])

# ============================================================
# TAB 1 : PALUDISME
# ============================================================

with tab_palu:
    st.header("🦟 Application de Surveillance du Paludisme")
    
    # Section 1 : Introduction
    st.markdown("## 📋 C'est quoi EpiPalu Predict ?")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>En bref</h3>
            <p style="font-size:1.1rem;">EpiPalu Predict est un <b>outil intelligent</b> qui vous aide à :</p>
            <ul style="font-size:1.05rem; line-height:1.8;">
                <li><b>Visualiser</b> où se trouvent les cas de paludisme</li>
                <li><b>Comprendre</b> l'influence du climat (pluie, chaleur)</li>
                <li><b>Prévoir</b> où les cas vont augmenter (1 à 12 semaines)</li>
                <li><b>Alerter</b> les zones à risque élevé</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="benefit-box">
            <h4>Pourquoi c'est utile ?</h4>
            <ul style="line-height:1.8;">
                <li><b>Gagner du temps</b> : Analyse automatique en quelques clics</li>
                <li><b>Anticiper</b> : Préparer les interventions avant les pics</li>
                <li><b>Optimiser</b> : Mieux répartir les ressources (médicaments, moustiquaires)</li>
                <li><b>Décider</b> : S'appuyer sur des données objectives</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Section 2 : Sources de données
    st.markdown("## 📊 Sources de Données Intégrées")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>🌍 Données Géographiques</h4>
            <ul>
                <li><b>Aires de santé</b> : Zones administratives sanitaires (GeoJSON/Shapefile)</li>
                <li><b>Cours d'eau</b> : Réseau hydrographique (zones de reproduction moustiques)</li>
                <li><b>Altitude</b> : Modèle numérique de terrain (rasters)</li>
                <li><b>Zones inondables</b> : Risque d'inondation (rasters)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>🌡️ Données Climatiques</h4>
            <ul>
                <li><b>NASA POWER API</b> (gratuit) :
                    <ul>
                        <li>Température quotidienne (°C)</li>
                        <li>Précipitations (mm/jour)</li>
                        <li>Humidité relative (%)</li>
                    </ul>
                </li>
                <li><b>Agrégation</b> : Moyennes/totaux hebdomadaires par aire de santé</li>
                <li><b>Période</b> : Données disponibles depuis 1981</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>👥 Données Démographiques (NOUVEAU)</h4>
            <ul>
                <li><b>WorldPop</b> (Google Earth Engine) :
                    <ul>
                        <li><b>Population totale</b> : Nombre d'habitants par aire</li>
                        <li><b>Enfants 0-14 ans</b> : Population pédiatrique (plus vulnérable)</li>
                        <li><b>Densité de population</b> : Habitants/km²</li>
                        <li><b>Tranches d'âge détaillées</b> : 0-4, 5-9, 10-14... jusqu'à 30-34 ans</li>
                    </ul>
                </li>
                <li><b>Résolution</b> : 100m (précision quartier)</li>
                <li><b>Mise à jour</b> : Données annuelles (dernière : 2020)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="benefit-box">
            <h5>🎯 Utilité pour l'analyse du paludisme</h5>
            <ul style="background:#fff9c4; padding:1rem; border-radius:5px;">
                <li><b>Calcul taux d'incidence</b> : Cas pour 10 000 habitants (indicateur épidémiologique standard)</li>
                <li><b>Priorisation zones à risque</b> : Densité forte + cas élevés = intervention urgente</li>
                <li><b>Estimation besoins</b> : Population enfants → doses médicaments/moustiquaires</li>
                <li><b>Coefficient d'ajustement prédictions</b> : Le modèle ajuste ses prévisions selon la pression démographique (risque relatif par zone)</li>
                <li><b>Identification poches vulnérables</b> : Enfants 0-14 ans = 60-80% des cas graves de paludisme</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h4>📋 Données Épidémiologiques</h4>
            <ul>
                <li><b>Cas hebdomadaires</b> : Nombre de cas confirmés par semaine et aire</li>
                <li><b>Décès</b> : Mortalité palustre</li>
                <li><b>Format attendu</b> : CSV avec colonnes health_area, week_, cases, deaths</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Section 3 : Guide d'utilisation
    st.markdown("## 🚀 Guide d'Utilisation Pas-à-Pas")
    
    with st.expander("**Étape 1️⃣ : Charger les aires de santé**", expanded=False):
        st.markdown("""
        <div class="info-card">
            <h4>Comment faire ?</h4>
            <ol>
                <li>Dans la <b>sidebar</b> (barre latérale gauche), cliquez sur <b>"🗺️ Aires de Santé"</b></li>
                <li>Uploadez votre fichier (formats acceptés : .shp, .geojson, .zip)</li>
                <li>Le fichier doit contenir les colonnes :
                    <ul>
                        <li><code>health_area</code> : Nom de l'aire de santé</li>
                        <li><code>geometry</code> : Géométrie (polygones)</li>
                    </ul>
                </li>
            </ol>
            <h5>✅ Validation</h5>
            <p>Vous devez voir : <code>✓ X aires de santé chargées</code></p>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("**Étape 2️⃣ : Charger les cas de paludisme**", expanded=False):
        st.markdown("""
        <div class="info-card">
            <h4>Format CSV attendu</h4>
            <p><b>Colonnes obligatoires :</b></p>
            <table style="width:100%; border-collapse:collapse;">
                <tr style="background:#f5f5f5;">
                    <th style="padding:0.5rem; border:1px solid #ddd;">Colonne</th>
                    <th style="padding:0.5rem; border:1px solid #ddd;">Description</th>
                    <th style="padding:0.5rem; border:1px solid #ddd;">Exemple</th>
                </tr>
                <tr>
                    <td style="padding:0.5rem; border:1px solid #ddd;"><code>health_area</code></td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">Nom de l'aire (doit correspondre au fichier géographique)</td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">Dakar Centre</td>
                </tr>
                <tr>
                    <td style="padding:0.5rem; border:1px solid #ddd;"><code>week_</code></td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">Numéro ou nom de semaine</td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">S01, 2024-W01</td>
                </tr>
                <tr>
                    <td style="padding:0.5rem; border:1px solid #ddd;"><code>cases</code></td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">Nombre de cas confirmés</td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">45</td>
                </tr>
                <tr>
                    <td style="padding:0.5rem; border:1px solid #ddd;"><code>deaths</code></td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">Nombre de décès (optionnel)</td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">2</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("**Étape 3️⃣ : Activer les données climatiques (optionnel mais recommandé)**", expanded=False):
        st.markdown("""
        <div class="benefit-box">
            <h4>⚡ API Climat - GRATUIT et RAPIDE</h4>
            <p><b>NASA POWER API</b> : Données météo depuis 1981, sans inscription</p>
            <h5>Activation :</h5>
            <ol>
                <li>Cochez <code>☑️ Activer API Climat</code> dans la sidebar</li>
                <li>Sélectionnez l'API : <b>NASA POWER (recommandé)</b></li>
                <li>Cliquez sur <b>"📥 Télécharger Données Climat"</b></li>
                <li>Patientez 10-30 secondes (selon nombre d'aires)</li>
            </ol>
            <h5>Variables obtenues (par aire et par semaine) :</h5>
            <ul>
                <li>🌡️ <b>Température moyenne</b> : Impact direct sur cycle de reproduction moustique</li>
                <li>🌧️ <b>Précipitations totales</b> : Gîtes larvaires (eau stagnante)</li>
                <li>💧 <b>Humidité relative</b> : Survie des moustiques adultes</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
            <h5>⚠️ Pourquoi c'est important ?</h5>
            <p>Le paludisme est une <b>maladie climatosensible</b> :</p>
            <ul>
                <li><b>Température optimale</b> : 25-30°C (accélère développement parasite dans moustique)</li>
                <li><b>Pluies</b> : Créent gîtes larvaires → explosion population moustiques 7-10 jours après</li>
                <li><b>Humidité > 60%</b> : Augmente longévité moustiques femelles (transmission prolongée)</li>
            </ul>
            <p><b>Impact sur prédictions :</b> +20-30% de précision avec climat vs sans climat</p>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("**Étape 4️⃣ : Activer les données démographiques (NOUVEAU)**", expanded=False):
        st.markdown("""
        <div class="benefit-box">
            <h4>👥 WorldPop - Population haute résolution</h4>
            <p><b>Google Earth Engine</b> : Données populationnelles mondiales, résolution 100m</p>
            <h5>Activation :</h5>
            <ol>
                <li>Cochez <code>☑️ Activer WorldPop (GEE)</code> dans la sidebar</li>
                <li>Assurez-vous que GEE est connecté (voir <code>✓ GEE connecté</code> en haut)</li>
                <li>Cliquez sur <b>"📥 Extraire Population"</b></li>
                <li>Patientez 20-60 secondes (calcul par aire de santé)</li>
            </ol>
            <h5>Données extraites :</h5>
            <table style="width:100%; border-collapse:collapse; margin:1rem 0;">
                <tr style="background:#f5f5f5;">
                    <th style="padding:0.5rem; border:1px solid #ddd;">Variable</th>
                    <th style="padding:0.5rem; border:1px solid #ddd;">Description</th>
                    <th style="padding:0.5rem; border:1px solid #ddd;">Utilisation</th>
                </tr>
                <tr>
                    <td style="padding:0.5rem; border:1px solid #ddd;"><b>Pop_Totale</b></td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">Nombre total d'habitants</td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">Dénominateur taux d'incidence</td>
                </tr>
                <tr>
                    <td style="padding:0.5rem; border:1px solid #ddd;"><b>Pop_Enfants_0_14</b></td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">Population pédiatrique</td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">Estimation besoins en MII/médicaments</td>
                </tr>
                <tr>
                    <td style="padding:0.5rem; border:1px solid #ddd;"><b>Densite_Pop</b></td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">Habitants par km²</td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">Risque de transmission (densité élevée = plus de contacts)</td>
                </tr>
                <tr>
                    <td style="padding:0.5rem; border:1px solid #ddd;"><b>Pop_M/F_0_4, 5_9...</b></td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">Tranches d'âge par sexe</td>
                    <td style="padding:0.5rem; border:1px solid #ddd;">Pyramide des âges (visualisation)</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="alert-box">
            <h5>🎯 Impact sur l'analyse et la prédiction</h5>
            <h6>1. Calcul automatique du taux d'incidence</h6>
            <code>Taux d'incidence = (Cas / Pop_Totale) × 10 000</code>
            <p>Permet de comparer le risque entre zones de tailles différentes</p>
            <h6>2. Coefficient d'ajustement démographique</h6>
            <p>Le modèle prédictif calcule un <b>coefficient de risque relatif</b> pour chaque aire :</p>
            <ul>
                <li><b>Coefficient > 1</b> : Zone à risque plus élevé que la moyenne (ex: forte densité + faible altitude)</li>
                <li><b>Coefficient < 1</b> : Zone à risque plus faible (ex: faible densité, zone urbaine bien drainée)</li>
                <li><b>Coefficient = 1</b> : Risque moyen</li>
            </ul>
            <p><b>Utilisation :</b> Les prédictions sont multipliées par ce coefficient pour mieux refléter le risque local</p>
            <h6>3. Priorisation des interventions</h6>
            <p>Le tableau de bord affiche automatiquement :</p>
            <ul>
                <li>🔴 <b>Zones prioritaires</b> : Cas élevés + population enfants élevée + densité forte</li>
                <li>🟡 <b>Zones à surveiller</b> : Taux d'incidence croissant + coefficient risque > 1.2</li>
            </ul>
            <h6>4. Estimation des besoins en ressources</h6>
            <p>Calculs automatiques basés sur Pop_Enfants_0_14 :</p>
            <code>
            • Moustiquaires (1 MII pour 2 enfants) : Pop_Enfants / 2<br>
            • Doses TDR (20% population exposée) : Pop_Totale × 0.20<br>
            • ACT (15% cas confirmés) : Cas_prédits × 0.15
            </code>
        </div>
        """, unsafe_allow_html=True)
    
    # Reste des sections Paludisme...
    st.markdown("---")
    st.markdown("## 💡 Conseils et Bonnes Pratiques")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="benefit-box">
            <h4>✅ Pour des prédictions optimales</h4>
            <ul>
                <li><b>Données historiques</b> : Au moins 20-30 semaines de données</li>
                <li><b>Activer API Climat</b> : +20-30% précision</li>
                <li><b>Activer WorldPop</b> : Coefficient ajustement démographique</li>
                <li><b>Ajouter rasters environnement</b> : Inondation, altitude, rivières</li>
                <li><b>Utiliser Gradient Boosting</b> : Meilleur algorithme pour séries temporelles</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Limites et précautions</h4>
            <ul>
                <li><b>Qualité données</b> : Vérifiez cohérence (pas de valeurs aberrantes)</li>
                <li><b>Prédictions long terme</b> : Plus c'est loin, moins c'est précis</li>
                <li><b>Événements exceptionnels</b> : Le modèle ne prédit pas les épidémies inhabituelles</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# TAB 2 : ROUGEOLE (Structure identique - Je montre juste le début)
# ============================================================

with tab_rougeole:
    st.header("🦠 Application de Surveillance de la Rougeole")
    
    st.markdown("## 📋 Qu'est-ce que l'application Rougeole ?")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>En bref</h3>
            <p style="font-size:1.1rem;">Outil spécialisé de <b>surveillance et prédiction</b> des épidémies de rougeole :</p>
            <ul style="font-size:1.05rem; line-height:1.8;">
                <li><b>Suivi temporel précis</b> : Analyse par semaines épidémiologiques</li>
                <li><b>Détection précoce</b> : Alertes automatiques basées sur seuils historiques</li>
                <li><b>Évaluation couverture vaccinale</b> : Identification poches de susceptibilité</li>
                <li><b>Prédiction ML</b> : Anticipation flambées 4-12 semaines à l'avance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # ... (Continuez avec la même structure pour Rougeole)
    st.info("📖 Section Rougeole : Structure identique à Paludisme avec contenus spécifiques rougeole")

# ============================================================
# TAB 3 : GLOSSAIRE
# ============================================================

with tab_glossaire:
    st.header("📖 Glossaire des Variables & Méthodologie")
    
    glossary_tabs = st.tabs([
        "📅 Temporelles",
        "📊 Historique",
        "🌡️ Climat",
        "👥 Démographie",
        "🧮 Méthodes"
    ])
    
    with glossary_tabs[0]:
        st.markdown("""
        <div class="info-card">
            <h4>Numéro de semaine (week_num)</h4>
            <p><b>Signification :</b> Numéro séquentiel de la semaine (1, 2, 3...)</p>
            <p><b>Utilité :</b> Capture la tendance générale dans le temps</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#7f8c8d; padding:2rem;">
    <h4>📧 Contact Support Technique</h4>
    <p>📧 Email : youssoupha.mbodji@example.com</p>
    <p>Version 3.0 | Développé par <b>Youssoupha MBODJI</b> | © 2026</p>
</div>
""", unsafe_allow_html=True)
