# ============================================================
# APP SURVEILLANCE & PRÉDICTION ROUGEOLE - VERSION 3.0
# PARTIE 1/6 - IMPORTS ET CONFIGURATION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime, timedelta
import requests
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
import ee
import json
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
import zipfile
import tempfile
import os
from shapely.geometry import shape
import warnings
warnings.filterwarnings('ignore')

# Configuration Streamlit
#st.set_page_config(
#    page_title="Surveillance Rougeole Multi-pays",
#    layout="wide",
#    page_icon="🦠",
#    initial_sidebar_state="expanded"
#)

# CSS personnalisé
st.markdown("""
<style>
.metric-card{background-color:#f0f2f6;padding:15px;border-radius:10px;box-shadow:2px 2px 5px rgba(0,0,0,0.1)}
.high-risk{background-color:#ffebee;color:#c62828;font-weight:bold;padding:5px;border-radius:3px}
.medium-risk{background-color:#fff3e0;color:#ef6c00;padding:5px;border-radius:3px}
.low-risk{background-color:#e8f5e9;color:#2e7d32;padding:5px;border-radius:3px}
.stButton>button{width:100%}
h1{color:#d32f2f}
.info-box{background-color:#e3f2fd;padding:10px;border-left:4px solid #2196f3;margin:10px 0}
.model-hint{background-color:#fff9c4;padding:8px;border-radius:5px;font-size:0.9em;margin:5px 0}
.weight-box{background-color:#e8f5e9;padding:10px;border-radius:5px;margin:10px 0;border-left:4px solid #4caf50}
</style>
""", unsafe_allow_html=True)

st.title("🦠 Dashboard de Surveillance et Prédiction - Rougeole")
st.markdown("### Analyse épidémiologique et modélisation prédictive par semaines épidémiologiques")

# Mapping pays ISO3
PAYS_ISO3_MAP = {
    "Niger": "ner",
    "Burkina Faso": "bfa",
    "Mali": "mli",
    "Mauritanie": "mrt"
}

# Initialisation Google Earth Engine
@st.cache_resource
def init_gee():
    try:
        key_dict = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
        credentials = ee.ServiceAccountCredentials(
            key_dict["client_email"],
            key_data=json.dumps(key_dict)
        )
        ee.Initialize(credentials)
        return True
    except:
        try:
            ee.Initialize()
            return True
        except:
            return False

gee_ok = init_gee()
if gee_ok:
    st.sidebar.success("✓ GEE connecté")

# Session state
if 'pays_precedent' not in st.session_state:
    st.session_state.pays_precedent = None
if 'sa_gdf_cache' not in st.session_state:
    st.session_state.sa_gdf_cache = None

# Configuration Sidebar
st.sidebar.header("📂 Configuration de l'Analyse")

# Mode démo
st.sidebar.subheader("🎯 Mode d'utilisation")
mode_demo = st.sidebar.radio(
    "Choisissez votre mode",
    ["📊 Données réelles", "🧪 Mode démo (données simulées)"],
    help="Mode démo : génère automatiquement des données fictives pour tester l'application"
)

# Aires de santé
st.sidebar.subheader("🗺️ Aires de Santé")
option_aire = st.sidebar.radio(
    "Source des données géographiques",
    ["Fichier local (ao_hlthArea.zip)", "Upload personnalisé"],
    key='option_aire'
)

pays_selectionne = None
iso3_pays = None

if option_aire == "Fichier local (ao_hlthArea.zip)":
    pays_selectionne = st.sidebar.selectbox(
        "🌍 Sélectionner le pays",
        list(PAYS_ISO3_MAP.keys()),
        key='pays_select'
    )
    iso3_pays = PAYS_ISO3_MAP[pays_selectionne]
    
    pays_change = (st.session_state.pays_precedent != pays_selectionne)
    if pays_change:
        st.session_state.pays_precedent = pays_selectionne
        st.session_state.sa_gdf_cache = None
        st.rerun()

upload_file = None
if option_aire == "Upload personnalisé":
    upload_file = st.sidebar.file_uploader(
        "Charger un fichier géographique",
        type=["shp", "geojson", "zip"],
        help="Format : Shapefile ou GeoJSON avec colonnes 'iso3' et 'health_area'"
    )

# Données épidémiologiques
st.sidebar.subheader("📊 Données Épidémiologiques")

if mode_demo == "🧪 Mode démo (données simulées)":
    option_linelist = "Données fictives (test)"
    linelist_file = None
    vaccination_file = None
    st.sidebar.info("📊 Mode démo activé - Données simulées")
else:
    linelist_file = st.sidebar.file_uploader(
        "📋 Linelists rougeole (CSV)",
        type=["csv"],
        help="Format : health_area, Semaine_Epi, Cas_Total OU Date_Debut_Eruption, Aire_Sante..."
    )
    
    vaccination_file = st.sidebar.file_uploader(
        "💉 Couverture vaccinale (CSV - optionnel)",
        type=["csv"],
        help="Format : health_area, Taux_Vaccination (en %)"
    )

# Période d'analyse
st.sidebar.subheader("📅 Période d'Analyse")

# Sélection par semaines épidémiologiques
col1, col2 = st.sidebar.columns(2)

with col1:
    annee_debut = st.number_input(
        "Année début",
        min_value=2000,
        max_value=datetime.now().year,
        value=2024,
        step=1,
        key="annee_debut"
    )
    
    semaine_debut = st.number_input(
        "Semaine début",
        min_value=1,
        max_value=53,
        value=1,
        step=1,
        key="semaine_debut",
        help="Semaine épidémiologique (1-53)"
    )

with col2:
    annee_fin = st.number_input(
        "Année fin",
        min_value=2000,
        max_value=datetime.now().year,
        value=datetime.now().year,
        step=1,
        key="annee_fin"
    )
    
    semaine_fin = st.number_input(
        "Semaine fin",
        min_value=1,
        max_value=53,
        value=datetime.now().isocalendar().week,
        step=1,
        key="semaine_fin",
        help="Semaine épidémiologique (1-53)"
    )

# Validation de la période
if annee_debut > annee_fin:
    st.sidebar.error("⚠️ L'année de début doit être ≤ année de fin")
elif annee_debut == annee_fin and semaine_debut > semaine_fin:
    st.sidebar.error("⚠️ La semaine de début doit être ≤ semaine de fin")
else:
    # Calculer le nombre de semaines dans la période
    nb_annees = annee_fin - annee_debut
    nb_semaines = (nb_annees * 52) + (semaine_fin - semaine_debut) + 1
    st.sidebar.success(f"✅ Période : {nb_semaines} semaines")
    st.sidebar.info(f"📅 S{semaine_debut:02d}/{annee_debut} → S{semaine_fin:02d}/{annee_fin}")


# Paramètres de prédiction
st.sidebar.subheader("🔮 Paramètres de Prédiction")
pred_mois = st.sidebar.slider(
    "Période de prédiction (mois)",
    min_value=1,
    max_value=12,
    value=3,
    help="Nombre de mois à prédire après la dernière semaine de données"
)
n_weeks_pred = pred_mois * 4

st.sidebar.info(f"📆 Prédiction sur **{n_weeks_pred} semaines épidémiologiques** (~{pred_mois} mois)")

# Choix du modèle
st.sidebar.subheader("🤖 Modèle de Prédiction")

modele_choisi = st.sidebar.selectbox(
    "Choisissez votre algorithme",
    [
        "GradientBoosting (Recommandé)",
        "RandomForest",
        "Ridge Regression",
        "Lasso Regression",
        "Decision Tree"
    ],
    help="Sélectionnez l'algorithme de machine learning pour la prédiction"
)

# Hints pour chaque modèle
model_hints = {
    "GradientBoosting (Recommandé)": "🎯 **Gradient Boosting** : Très performant pour les séries temporelles. Combine plusieurs modèles faibles pour créer un modèle fort. Excellent pour capturer les relations non-linéaires. Recommandé pour la surveillance épidémiologique.",
    "RandomForest": "🌳 **Random Forest** : Ensemble d'arbres de décision. Robuste aux valeurs aberrantes et aux données manquantes. Bon pour les interactions complexes entre variables.",
    "Ridge Regression": "📊 **Ridge Regression** : Régression linéaire avec régularisation L2. Simple et rapide. Idéal pour relations linéaires. Moins performant sur données non-linéaires.",
    "Lasso Regression": "🎯 **Lasso Regression** : Régularisation L1 avec sélection automatique des variables. Utile quand beaucoup de variables peu importantes. Simplifie le modèle.",
    "Decision Tree": "🌲 **Decision Tree** : Arbre de décision unique. Simple à interpréter mais risque de sur-apprentissage. Moins robuste que les méthodes d'ensemble."
}

st.sidebar.markdown(f'<div class="model-hint">{model_hints[modele_choisi]}</div>', unsafe_allow_html=True)

# ========== SYSTÈME HYBRIDE D'IMPORTANCE DES VARIABLES ==========
st.sidebar.subheader("⚖️ Importance des Variables")

mode_importance = st.sidebar.radio(
    "Mode de pondération",
    ["🤖 Automatique (ML)", "👨‍⚕️ Manuel (Expert)"],
    help="Automatique : calculé par le modèle ML | Manuel : poids définis par expertise épidémiologique"
)

poids_manuels = {}
poids_normalises = {}

if mode_importance == "👨‍⚕️ Manuel (Expert)":
    with st.sidebar.expander("⚙️ Configurer les poids", expanded=True):
        st.markdown("**Définissez l'importance de chaque groupe de variables**")
        st.caption("Les poids seront automatiquement normalisés pour totaliser 100%")
        
        poids_manuels["Historique_Cas"] = st.slider(
            "📈 Historique des cas (lags)",
            min_value=0,
            max_value=100,
            value=40,
            step=5,
            help="Importance des cas passés (4 dernières semaines)"
        )
        
        poids_manuels["Vaccination"] = st.slider(
            "💉 Couverture vaccinale",
            min_value=0,
            max_value=100,
            value=35,
            step=5,
            help="Importance du taux de vaccination et non-vaccinés"
        )
        
        poids_manuels["Demographie"] = st.slider(
            "👥 Démographie",
            min_value=0,
            max_value=100,
            value=15,
            step=5,
            help="Importance de la population et densité"
        )
        
        poids_manuels["Urbanisation"] = st.slider(
            "🏙️ Urbanisation",
            min_value=0,
            max_value=100,
            value=8,
            step=2,
            help="Importance du type d'habitat (urbain/rural)"
        )
        
        poids_manuels["Climat"] = st.slider(
            "🌡️ Facteurs climatiques",
            min_value=0,
            max_value=100,
            value=2,
            step=1,
            help="Importance de la température, humidité, saison"
        )
        
        # Calculer le total et normaliser
        total_poids = sum(poids_manuels.values())
        
        if total_poids > 0:
            for key in poids_manuels:
                poids_normalises[key] = poids_manuels[key] / total_poids
        
        # Afficher le résumé
        st.markdown("---")
        st.markdown("**📊 Répartition normalisée :**")
        for key, value in poids_normalises.items():
            st.markdown(f"• {key} : **{value*100:.1f}%**")
        
        if abs(total_poids - 100) > 5:
            st.info(f"ℹ️ Total brut : {total_poids}% → Normalisé à 100%")
else:
    st.sidebar.info("Le modèle ML calculera automatiquement l'importance optimale de chaque variable")

# Seuils d'alerte
st.sidebar.subheader("⚙️ Seuils d'Alerte")
with st.sidebar.expander("Configurer les seuils", expanded=False):
    seuil_baisse = st.slider(
        "Seuil de baisse significative (%)",
        min_value=10,
        max_value=90,
        value=75,
        step=5,
        help="Afficher les aires avec baisse ≥ X% par rapport à la moyenne"
    )
    seuil_hausse = st.slider(
        "Seuil de hausse significative (%)",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        help="Afficher les aires avec hausse ≥ X% par rapport à la moyenne"
    )
    seuil_alerte_epidemique = st.number_input(
        "Seuil d'alerte épidémique (cas/semaine)",
        min_value=1,
        max_value=100,
        value=5,
        help="Nombre de cas par semaine déclenchant une alerte"
    )

# Fonctions de chargement géographique
@st.cache_data
def load_health_areas_from_zip(zip_path, iso3_filter):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(tmpdir)
            
            shp_files = [f for f in os.listdir(tmpdir) if f.endswith('.shp')]
            if not shp_files:
                raise ValueError("Aucun fichier .shp trouvé dans le ZIP")
            
            shp_path = os.path.join(tmpdir, shp_files[0])
            gdf_full = gpd.read_file(shp_path)
            
            iso3_col = None
            for col in ['iso3', 'ISO3', 'iso_code', 'ISO_CODE', 'country_iso', 'COUNTRY_ISO']:
                if col in gdf_full.columns:
                    iso3_col = col
                    break
            
            if iso3_col is None:
                st.warning(f"⚠️ Colonne ISO3 non trouvée. Colonnes : {list(gdf_full.columns)}")
                return gpd.GeoDataFrame()
            
            gdf = gdf_full[gdf_full[iso3_col] == iso3_filter].copy()
            
            if gdf.empty:
                st.warning(f"⚠️ Aucune aire de santé pour {iso3_filter}")
                return gpd.GeoDataFrame()
            
            name_col = None
            for col in ['health_area', 'HEALTH_AREA', 'name_fr', 'name', 'NAME', 'nom', 'NOM', 'aire_sante']:
                if col in gdf.columns:
                    name_col = col
                    break
            
            if name_col:
                gdf['health_area'] = gdf[name_col]
            else:
                gdf['health_area'] = [f"Aire_{i+1}" for i in range(len(gdf))]
            
            gdf = gdf[gdf.geometry.is_valid]
            
            if gdf.crs is None:
                gdf.set_crs("EPSG:4326", inplace=True)
            elif gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs("EPSG:4326")
            
            return gdf
            
    except Exception as e:
        st.error(f"❌ Erreur ZIP : {e}")
        return gpd.GeoDataFrame()

def load_shapefile_from_upload(upload_file):
    try:
        if upload_file.name.endswith('.zip'):
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, 'upload.zip')
                with open(zip_path, 'wb') as f:
                    f.write(upload_file.getvalue())
                
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(tmpdir)
                    shp_files = [f for f in os.listdir(tmpdir) if f.endswith('.shp')]
                    if shp_files:
                        gdf = gpd.read_file(os.path.join(tmpdir, shp_files[0]))
                    else:
                        raise ValueError("Aucun .shp trouvé")
        else:
            gdf = gpd.read_file(upload_file)
        
        if "health_area" not in gdf.columns:
            for col in ["health_area", "HEALTH_AREA", "name_fr", "name", "NAME", "nom", "NOM"]:
                if col in gdf.columns:
                    gdf["health_area"] = gdf[col]
                    break
            else:
                gdf["health_area"] = [f"Aire_{i}" for i in range(len(gdf))]
        
        gdf = gdf[gdf.geometry.is_valid]
        
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")
        
        return gdf
        
    except Exception as e:
        st.error(f"❌ Erreur lecture : {e}")
        return gpd.GeoDataFrame()

# ============================================================
# PARTIE 2/6 - CHARGEMENT AIRES DE SANTÉ ET DONNÉES DE CAS
# ============================================================

# Chargement des aires de santé
if st.session_state.sa_gdf_cache is not None and option_aire == "Fichier local (ao_hlthArea.zip)":
    sa_gdf = st.session_state.sa_gdf_cache
    st.sidebar.success(f"✓ {len(sa_gdf)} aires chargées (cache)")
else:
    with st.spinner(f"🔄 Chargement des aires de santé..."):
        if option_aire == "Fichier local (ao_hlthArea.zip)":
            zip_path = os.path.join("data", "ao_hlthArea.zip")
            if not os.path.exists(zip_path):
                st.error(f"❌ Fichier non trouvé : {zip_path}")
                st.info("📁 Placez 'ao_hlthArea.zip' dans le dossier 'data/'")
                st.stop()
            
            sa_gdf = load_health_areas_from_zip(zip_path, iso3_pays)
            
            if sa_gdf.empty:
                st.error(f"❌ Impossible de charger {pays_selectionne} ({iso3_pays})")
                st.stop()
            else:
                st.sidebar.success(f"✓ {len(sa_gdf)} aires chargées ({iso3_pays})")
                st.session_state.sa_gdf_cache = sa_gdf
                
        elif option_aire == "Upload personnalisé":
            if upload_file is None:
                st.warning("⚠️ Veuillez uploader un fichier")
                st.stop()
            else:
                sa_gdf = load_shapefile_from_upload(upload_file)
                if sa_gdf.empty:
                    st.error("❌ Fichier invalide")
                    st.stop()
                else:
                    st.sidebar.success(f"✓ {len(sa_gdf)} aires chargées")
                    st.session_state.sa_gdf_cache = sa_gdf

if sa_gdf.empty or sa_gdf is None:
    st.error("❌ Aucune aire chargée")
    st.stop()

# Génération de données fictives
@st.cache_data
def generate_dummy_linelists(_sa_gdf, n=500, start=None, end=None):
    np.random.seed(42)
    
    if start is None:
        start = datetime(2024, 1, 1)
    if end is None:
        end = datetime.today()
    
    delta_days = (end - start).days
    dates = pd.to_datetime(start) + pd.to_timedelta(
        np.random.exponential(scale=delta_days/3, size=n).clip(0, delta_days).astype(int),
        unit="D"
    )
    
    df = pd.DataFrame({
        "ID_Cas": range(1, n+1),
        "Date_Debut_Eruption": dates,
        "Date_Notification": dates + pd.to_timedelta(np.random.poisson(3, n), unit="D"),
        "Aire_Sante": np.random.choice(_sa_gdf["health_area"].unique(), n),
        "Age_Mois": np.random.gamma(shape=2, scale=30, size=n).clip(6, 180).astype(int),
        "Statut_Vaccinal": np.random.choice(["Oui", "Non"], n, p=[0.55, 0.45]),
        "Sexe": np.random.choice(["M", "F"], n),
        "Issue": np.random.choice(["Guéri", "Décédé", "Inconnu"], n, p=[0.92, 0.03, 0.05])
    })
    
    return df

@st.cache_data
def generate_dummy_vaccination(_sa_gdf):
    np.random.seed(42)
    
    return pd.DataFrame({
        "health_area": _sa_gdf["health_area"],
        "Taux_Vaccination": np.random.beta(a=8, b=2, size=len(_sa_gdf)) * 100
    })
# ============================================================
# CHARGEMENT LINELIST - DÉTECTION AUTOMATIQUE DU SÉPARATEUR
# ============================================================

with st.spinner('Chargement données de cas...'):
    if mode_demo == "🧪 Mode démo (données simulées)":
        df = generate_dummy_linelists(sa_gdf, start=start_date, end=end_date)
        vaccination_df = generate_dummy_vaccination(sa_gdf)
        st.sidebar.info(f"{len(df)} cas simulés générés")
    else:
        if linelist_file is None:
            st.error("Veuillez uploader un fichier CSV de lineliste")
            st.stop()
        
        try:
            # ====== DÉTECTION AUTOMATIQUE DU SÉPARATEUR ======
            sample = linelist_file.read(1024).decode('utf-8', errors='ignore')
            linelist_file.seek(0)
            
            semicolon_count = sample.count(';')
            comma_count = sample.count(',')
            tab_count = sample.count('\t')
            
            if semicolon_count > comma_count and semicolon_count > tab_count:
                separator = ';'
            elif tab_count > comma_count:
                separator = '\t'
            else:
                separator = ','
            
            st.sidebar.info(f"🔍 Séparateur détecté : `{repr(separator)}`")
            
            # Lire avec le bon séparateur
            try:
                df_raw = pd.read_csv(linelist_file, sep=separator, encoding='utf-8')
            except UnicodeDecodeError:
                linelist_file.seek(0)
                df_raw = pd.read_csv(linelist_file, sep=separator, encoding='latin1')
            
            st.sidebar.success(f"✅ {len(df_raw)} lignes chargées")
            
            # ====== MAPPING INTELLIGENT DES COLONNES ======
            COLUMNS_MAPPING_EXTENDED = {
                'health_area': [
                    'health_area', 'healtharea', 'HEALTH_AREA',
                    'aire_sante', 'Aire_Sante', 'airesante',
                    'district', 'District', 'zone', 'Zone',
                    'name_fr', 'NAME', 'nom', 'Nom'
                ],
                'Semaine_Epi': [
                    'Semaine_Epi', 'SemaineEpi', 'semaine_epi',
                    'semaine', 'Semaine', 'week', 'Week',
                    'epi_week', 'SE', 'se'
                ],
                'Annee': [
                    'Annee', 'Année', 'annee', 'année',
                    'year', 'Year', 'an', 'An'
                ],
                'Cas_Total': [
                    'Cas_Total', 'CasTotal', 'cas_total',
                    'cas', 'Cas', 'cases', 'Cases',
                    'nb_cas', 'nombre_cas'
                ],
                'Deces': [
                    'Deces', 'Décès', 'deces', 'décès',
                    'deaths', 'Deaths', 'nb_deces'
                ],
                'Region': [
                    'regions', 'Regions', 'region', 'Region'
                ]
            }
            
            # Appliquer le mapping
            rename_dict = {}
            for standard_col, possible_cols in COLUMNS_MAPPING_EXTENDED.items():
                for col in possible_cols:
                    if col in df_raw.columns:
                        if col != standard_col:
                            rename_dict[col] = standard_col
                        break
            
            if rename_dict:
                df_raw = df_raw.rename(columns=rename_dict)
                st.sidebar.success(f"🔄 Colonnes renommées : {len(rename_dict)}")
            
           # ====== VÉRIFIER FORMAT AGRÉGÉ OU LINELIST ======
            if 'Semaine_Epi' in df_raw.columns and ('Cas_Total' in df_raw.columns or 'cas' in df_raw.columns):
                # FORMAT AGRÉGÉ - Expansion en linelist
                st.sidebar.info("📊 Format agrégé détecté - Expansion en linelist...")
                
                # Normaliser le nom de la colonne cas
                if 'Cas_Total' not in df_raw.columns:
                    for col in ['cas', 'Cas', 'cases', 'Cases', 'nb_cas']:
                        if col in df_raw.columns:
                            df_raw['Cas_Total'] = df_raw[col]
                            break
                
                expanded_rows = []
                lignes_ignorees = 0
                
                for _, row in df_raw.iterrows():
                    try:
                        aire = row.get('health_area') or row.get('Aire_Sante', 'Inconnu')
                        
                        # Vérifier semaine
                        semaine_val = row.get('Semaine_Epi')
                        if pd.isna(semaine_val):
                            lignes_ignorees += 1
                            continue
                        semaine = int(semaine_val)
                        
                        # Valider que la semaine est entre 1 et 53
                        if semaine < 1 or semaine > 53:
                            lignes_ignorees += 1
                            continue
                        
                        # Vérifier cas total
                        cas_total_val = row.get('Cas_Total')
                        if pd.isna(cas_total_val) or cas_total_val <= 0:
                            lignes_ignorees += 1
                            continue
                        cas_total = int(cas_total_val)
                        
                        # Vérifier année
                        annee_val = row.get('Annee')
                        if pd.isna(annee_val):
                            annee = datetime.now().year  # Année courante par défaut
                        else:
                            annee = int(annee_val)
                        
                        # Valider l'année (entre 2000 et année courante + 1)
                        if annee < 2000 or annee > datetime.now().year + 1:
                            st.warning(f"⚠️ Année invalide détectée : {annee} pour aire {aire}, semaine {semaine}")
                            lignes_ignorees += 1
                            continue
                        
                        # Créer une date fictive pour la semaine ISO
                        try:
                            # Méthode ISO : année-Semaine-Jour (lundi = 1)
                            base_date = datetime.strptime(f"{annee}-W{semaine:02d}-1", "%Y-W%W-%w")
                        except:
                            try:
                                # Méthode alternative : 1er jour de l'année + (semaine-1) * 7 jours
                                base_date = datetime(int(annee), 1, 1) + timedelta(weeks=semaine-1)
                            except:
                                lignes_ignorees += 1
                                continue
                        
                        # Créer cas_total lignes individuelles
                        for i in range(cas_total):
                            # Répartir les cas aléatoirement sur les 7 jours de la semaine
                            jour_aleatoire = np.random.randint(0, 7)
                            date_cas = base_date + timedelta(days=jour_aleatoire)
                            
                            expanded_rows.append({
                                'ID_Cas': len(expanded_rows) + 1,
                                'Date_Debut_Eruption': date_cas,
                                'Date_Notification': date_cas + timedelta(days=np.random.randint(0, 10)),
                                'Aire_Sante': aire,
                                'Annee': annee,
                                'Semaine_Epi': semaine,
                                'Age_Mois': np.random.randint(6, 180),  # Âge aléatoire entre 6 mois et 15 ans
                                'Statut_Vaccinal': 'Inconnu',
                                'Sexe': 'Inconnu',
                                'Issue': 'Inconnu'
                            })
                    
                    except (ValueError, TypeError) as e:
                        lignes_ignorees += 1
                        continue
                
                if lignes_ignorees > 0:
                    st.sidebar.warning(f"⚠️ {lignes_ignorees} lignes ignorées (valeurs invalides ou manquantes)")
                
                if len(expanded_rows) == 0:
                    st.error("❌ Aucune donnée valide trouvée dans le CSV")
                    
                    with st.expander("🔍 Aperçu et diagnostic"):
                        st.write("**Premières lignes du fichier :**")
                        st.dataframe(df_raw.head(10))
                        
                        st.write("**Statistiques des colonnes :**")
                        col_info = pd.DataFrame({
                            'Colonne': df_raw.columns,
                            'Type': df_raw.dtypes.values,
                            'Valeurs manquantes': df_raw.isnull().sum().values,
                            'Valeurs uniques': [df_raw[col].nunique() for col in df_raw.columns]
                        })
                        st.dataframe(col_info)
                        
                        st.info("""
                        **Format attendu (agrégé) :**
                        - `health_area` ou `Aire_Sante` : nom de l'aire de santé
                        - `Semaine_Epi` : numéro de semaine (1 à 52)
                        - `Annee` : année (ex: 2024)
                        - `Cas_Total` ou `cas` : nombre de cas (> 0)
                        
                        **Exemple de ligne valide :**
                        | health_area | Semaine_Epi | Annee | Cas_Total |
                        |-------------|-------------|-------|-----------|
                        | Dakar       | 15          | 2024  | 12        |
                        """)
                    
                    st.stop()
                
                df = pd.DataFrame(expanded_rows)
                st.sidebar.success(f"✅ Expansion : {len(df)} cas individuels créés ({len(df['Aire_Sante'].unique())} aires, {len(df.groupby(['Annee', 'Semaine_Epi']))} semaines)")
                
                # Afficher un résumé de la période couverte
                periode_debut = df.groupby(['Annee', 'Semaine_Epi']).size().reset_index().iloc[0]
                periode_fin = df.groupby(['Annee', 'Semaine_Epi']).size().reset_index().iloc[-1]
                st.sidebar.info(f"📅 Période : S{periode_debut['Semaine_Epi']:02d}/{periode_debut['Annee']} → S{periode_fin['Semaine_Epi']:02d}/{periode_fin['Annee']}")
            
            elif 'Date_Debut_Eruption' in df_raw.columns:
                # FORMAT LINELIST STANDARD
                df = df_raw.copy()
                for col in ['Date_Debut_Eruption', 'Date_Notification']:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                
                # Créer les colonnes Semaine_Epi et Annee si elles n'existent pas
                if 'Semaine_Epi' not in df.columns:
                    df['Semaine_Epi'] = df['Date_Debut_Eruption'].dt.isocalendar().week
                if 'Annee' not in df.columns:
                    df['Annee'] = df['Date_Debut_Eruption'].dt.isocalendar().year

                st.sidebar.success(f"✅ Expansion : {len(df)} cas individuels créés")
            
            elif 'Date_Debut_Eruption' in df_raw.columns:
                # FORMAT LINELIST STANDARD
                df = df_raw.copy()
                for col in ['Date_Debut_Eruption', 'Date_Notification']:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
            
            else:
                st.error("❌ Format CSV non reconnu")
                st.info("""
                **Formats acceptés :**
                
                **Format 1 (Agrégé) :**
                - `health_area` ou `regions`
                - `Semaine_Epi` (numérique, sans valeurs vides)
                - `Cas_Total` (numérique > 0, sans valeurs vides)
                - `Annee` (numérique)
                
                **Format 2 (Linelist) :**
                - `Date_Debut_Eruption`
                - `Aire_Sante`
                - Autres colonnes optionnelles...
                """)
                
                with st.expander("🔍 Aperçu de vos données"):
                    st.write("**Colonnes détectées :**")
                    st.write(list(df_raw.columns))
                    st.write("**Premières lignes :**")
                    st.dataframe(df_raw.head(10))
                
                st.stop()
            
            st.sidebar.success(f"✅ {len(df)} cas chargés")
            
        except Exception as e:
            st.error(f"❌ Erreur CSV : {e}")
            st.code(f"Type d'erreur : {type(e).__name__}")
            
            # Aide contextuelle
            if "cannot convert float NaN to integer" in str(e):
                st.warning("""
                **Problème : Valeurs manquantes (NaN)**
                
                Votre fichier contient des cellules vides. Veuillez :
                1. Ouvrir le fichier CSV dans Excel/LibreOffice
                2. Supprimer les lignes avec des cellules vides dans `Semaine_Epi`, `Cas_Total` ou `Annee`
                3. Vérifier que toutes les valeurs sont numériques
                4. Sauvegarder et réessayer
                """)
            
            st.stop()
        
        if vaccination_file is not None:
            try:
                vaccination_df = pd.read_csv(vaccination_file)
                st.sidebar.success(f"✓ Couverture vaccinale chargée ({len(vaccination_df)} aires)")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Erreur vaccination CSV : {e}")
                vaccination_df = None
        else:
            if "Statut_Vaccinal" in df.columns:
                vacc_by_area = df.groupby("Aire_Sante").agg({
                    "Statut_Vaccinal": lambda x: ((x == "Oui").sum() / len(x) * 100) if len(x) > 0 else 0
                }).reset_index()
                vacc_by_area.columns = ["health_area", "Taux_Vaccination"]
                vaccination_df = vacc_by_area
                st.sidebar.info("ℹ️ Taux vaccination extrait de la linelist")
            else:
                vaccination_df = None
                st.sidebar.info("ℹ️ Pas de données de vaccination")

# Filtrer par période

# Normaliser les noms de colonnes du DataFrame df
COLONNES_MAPPING = {
    # Colonne aire de santé
    "Aire_Sante": ["Aire_Sante", "aire_sante", "health_area", "HEALTH_AREA", "name_fr", "NAME", "nom", "NOM"],
    # Colonne date de début
    "Date_Debut_Eruption": ["Date_Debut_Eruption", "date_debut_eruption", "Date_Debut", "date_onset", "Date_Onset", "symptom_onset"],
    # Colonne date notification
    "Date_Notification": ["Date_Notification", "date_notification", "Date_Notif", "date_notif", "notification_date"],
    # Colonne ID cas
    "ID_Cas": ["ID_Cas", "id_cas", "ID", "id", "Case_ID", "case_id", "ID_cas"],
    # Colonne âge
    "Age_Mois": ["Age_Mois", "age_mois", "Age", "age", "AGE", "Age_Months", "age_months"],
    # Colonne statut vaccinal
    "Statut_Vaccinal": ["Statut_Vaccinal", "statut_vaccinal", "Vaccin", "vaccin", "Vaccination_Status", "vaccination_status", "Vacc_Statut"],
    # Colonne sexe
    "Sexe": ["Sexe", "sexe", "Sex", "sex", "Gender", "gender"],
    # Colonne issue
    "Issue": ["Issue", "issue", "Outcome", "outcome", "OUTCOME"]
}

def normaliser_colonnes(dataframe, mapping):
    """Renommer les colonnes du dataframe selon le mapping standardisé"""
    rename_dict = {}
    for col_standard, col_possibles in mapping.items():
        for col_possible in col_possibles:
            if col_possible in dataframe.columns and col_possible != col_standard:
                rename_dict[col_possible] = col_standard
                break
    if rename_dict:
        dataframe = dataframe.rename(columns=rename_dict)
    return dataframe

# Appliquer la normalisation
df = normaliser_colonnes(df, COLONNES_MAPPING)

# Si "ID_Cas" n'existe pas, en créer une
if "ID_Cas" not in df.columns:
    df["ID_Cas"] = range(1, len(df) + 1)
# Normaliser les colonnes
if 'Date_Debut_Eruption' in df.columns:
    df['Date_Debut_Eruption'] = pd.to_datetime(df['Date_Debut_Eruption'], errors='coerce')
    dates_valides = df['Date_Debut_Eruption'].dropna()
    
    if len(dates_valides) > 0:
        date_min_data = dates_valides.min().date()
        date_max_data = dates_valides.max().date()
        
        # Afficher la plage disponible
        st.info(f"📅 **Plage de données disponibles :** {date_min_data} → {date_max_data}")
        
        # Vérifier que les dates sélectionnées sont valides
        if start_date < date_min_data:
            st.warning(f"⚠️ Date de début ajustée : {start_date} → {date_min_data} (données disponibles)")
            start_date = date_min_data
        
        if end_date > date_max_data:
            st.warning(f"⚠️ Date de fin ajustée : {end_date} → {date_max_data} (dernières données)")
            end_date = date_max_data
        
        if start_date > end_date:
            st.error("❌ La date de début doit être antérieure à la date de fin")
            st.stop()
        
        # ============================================================
        # FILTRAGE PAR SEMAINES ÉPIDÉMIOLOGIQUES
        # ============================================================
        
        # Créer les colonnes Annee et Semaine_Epi si elles n'existent pas encore
        if 'Annee' not in df.columns and 'Date_Debut_Eruption' in df.columns:
            df['Annee'] = df['Date_Debut_Eruption'].dt.isocalendar().year
        if 'Semaine_Epi' not in df.columns and 'Date_Debut_Eruption' in df.columns:
            df['Semaine_Epi'] = df['Date_Debut_Eruption'].dt.isocalendar().week
        
        # Vérifier que les colonnes existent
        if 'Annee' in df.columns and 'Semaine_Epi' in df.columns:
            
            # Afficher la plage disponible
            annee_min_data = df['Annee'].min()
            annee_max_data = df['Annee'].max()
            semaine_min_data = df[df['Annee'] == annee_min_data]['Semaine_Epi'].min()
            semaine_max_data = df[df['Annee'] == annee_max_data]['Semaine_Epi'].max()
            
            st.info(f"📅 **Plage de données disponibles :** S{semaine_min_data:02d}/{annee_min_data} → S{semaine_max_data:02d}/{annee_max_data}")
            
            # Filtrer par période sélectionnée
            df_before_filter = len(df)
            
            # Créer un identifiant unique pour trier chronologiquement
            df['Periode_ID'] = df['Annee'] * 100 + df['Semaine_Epi']
            periode_debut_id = annee_debut * 100 + semaine_debut
            periode_fin_id = annee_fin * 100 + semaine_fin
            
            df = df[(df['Periode_ID'] >= periode_debut_id) & (df['Periode_ID'] <= periode_fin_id)]
            df = df.drop(columns=['Periode_ID'])  # Supprimer la colonne temporaire
            
            df_after_filter = len(df)
            
            if df_after_filter == 0:
                st.error(f"❌ Aucune donnée disponible pour la période S{semaine_debut:02d}/{annee_debut} → S{semaine_fin:02d}/{annee_fin}")
                st.info(f"Plage disponible : S{semaine_min_data:02d}/{annee_min_data} → S{semaine_max_data:02d}/{annee_max_data}")
                st.stop()
            
            st.success(f"✅ **{df_after_filter:,} cas** sur la période sélectionnée ({df_before_filter - df_after_filter} cas exclus)")
            
        else:
            st.error("❌ Colonnes 'Annee' et 'Semaine_Epi' manquantes")
            st.stop()
        
        # Vérifier qu'il reste des données après filtrage
        if len(df) == 0:
            st.error("❌ Aucune donnée disponible après filtrage")
            st.stop()
        
        st.sidebar.success(f"✅ {len(df)} cas analysés")


# ====== CRÉER LES COLONNES TEMPORELLES ======
if 'Date_Debut_Eruption' in df.columns:
    df['Annee'] = df['Date_Debut_Eruption'].dt.isocalendar().year
    df['Semaine_Epi'] = df['Date_Debut_Eruption'].dt.isocalendar().week
else:
    st.error("❌ Colonne 'Date_Debut_Eruption' manquante")
    st.stop()

# Vérifier qu'il reste des données après filtrage
if len(df) == 0:
    st.error("❌ Aucune donnée disponible après filtrage")
    st.stop()
# Si "Aire_Sante" n'existe pas, essayer de la créer depuis sa_gdf
if "Aire_Sante" not in df.columns:
    # Chercher n'importe quelle colonne qui pourrait contenir un nom d'aire
    for col in df.columns:
        if df[col].dtype == object:
            # Vérifier si les valeurs matchent avec les aires de santé
            sample_values = set(df[col].dropna().unique())
            sa_values = set(sa_gdf["health_area"].unique())
            if len(sample_values.intersection(sa_values)) > 0:
                df["Aire_Sante"] = df[col]
                st.sidebar.info(f"ℹ️ Colonne 'Aire_Sante' créée depuis '{col}'")
                break
    else:
        # Si rien ne match, assigner une aire par défaut
        df["Aire_Sante"] = sa_gdf["health_area"].iloc[0]
        st.sidebar.warning("⚠️ Aucune colonne aire trouvée, valeur par défaut assignée")

# Vérifier et convertir les dates
if "Date_Debut_Eruption" in df.columns:
    df["Date_Debut_Eruption"] = pd.to_datetime(df["Date_Debut_Eruption"], errors='coerce')
else:
    # Chercher une colonne date
    for col in df.columns:
        try:
            test_dates = pd.to_datetime(df[col], errors='coerce')
            if test_dates.notna().sum() > len(df) * 0.5:  # Plus de 50% de dates valides
                df["Date_Debut_Eruption"] = test_dates
                st.sidebar.info(f"ℹ️ 'Date_Debut_Eruption' créée depuis '{col}'")
                break
        except:
            continue
    else:
        # Créer une date par défaut
        df["Date_Debut_Eruption"] = pd.to_datetime(start_date)
        st.sidebar.warning("⚠️ Aucune colonne date trouvée, date de début assignée par défaut")

if "Date_Notification" not in df.columns:
    # Créer Date_Notification = Date_Debut_Eruption + 3 jours par défaut
    df["Date_Notification"] = df["Date_Debut_Eruption"] + pd.to_timedelta(3, unit="D")

# Ajouter des colonnes optionnelles par défaut si absentes
if "Age_Mois" not in df.columns:
    df["Age_Mois"] = np.nan

if "Statut_Vaccinal" not in df.columns:
    df["Statut_Vaccinal"] = "Inconnu"

if "Sexe" not in df.columns:
    df["Sexe"] = "Inconnu"

if "Issue" not in df.columns:
    df["Issue"] = "Inconnu"

# Filtrer par période
df = df[
    (df["Date_Debut_Eruption"] >= pd.to_datetime(start_date)) &
    (df["Date_Debut_Eruption"] <= pd.to_datetime(end_date))
].copy()

if len(df) == 0:
    st.warning("⚠️ Aucun cas dans la période")
    st.stop()

# Calculer semaine épidémiologique
def calculer_semaine_epidemio(date):
    return date.isocalendar()[1]

df['Semaine_Epi'] = df['Date_Debut_Eruption'].apply(calculer_semaine_epidemio)
df['Annee'] = df['Date_Debut_Eruption'].dt.year
df['Semaine_Annee'] = df['Annee'].astype(str) + '-S' + df['Semaine_Epi'].astype(str).str.zfill(2)

derniere_semaine_epi = df['Semaine_Epi'].max()
derniere_annee = df['Annee'].max()

st.sidebar.info(f"📅 Dernière semaine : **S{derniere_semaine_epi}** ({derniere_annee})")
# ============================================================
# PARTIE 3/6 - ENRICHISSEMENT AVEC DONNÉES EXTERNES
# WorldPop, NASA POWER, GHSL
# ============================================================

# WorldPop - Données démographiques
@st.cache_data
def worldpop_children_stats(_sa_gdf, use_gee):
    if not use_gee:
        st.sidebar.warning("⚠️ WorldPop : GEE indisponible")
        return pd.DataFrame({
            "health_area": _sa_gdf["health_area"],
            "Pop_Totale": [np.nan] * len(_sa_gdf),
            "Pop_Garcons": [np.nan] * len(_sa_gdf),
            "Pop_Filles": [np.nan] * len(_sa_gdf),
            "Pop_Enfants": [np.nan] * len(_sa_gdf),
            # Nouvelles colonnes pour pyramide
            "Pop_M_0": [np.nan] * len(_sa_gdf),
            "Pop_M_1": [np.nan] * len(_sa_gdf),
            "Pop_M_5": [np.nan] * len(_sa_gdf),
            "Pop_M_10": [np.nan] * len(_sa_gdf),
            "Pop_F_0": [np.nan] * len(_sa_gdf),
            "Pop_F_1": [np.nan] * len(_sa_gdf),
            "Pop_F_5": [np.nan] * len(_sa_gdf),
            "Pop_F_10": [np.nan] * len(_sa_gdf)
        })
    
    try:
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        status_text.text("📥 Chargement WorldPop...")
        dataset = ee.ImageCollection("WorldPop/GP/100m/pop_age_sex")
        pop_img = dataset.mosaic()
        
        male_bands = ["M_0", "M_1", "M_5", "M_10"]
        female_bands = ["F_0", "F_1", "F_5", "F_10"]
        
        selected_males = pop_img.select(male_bands)
        selected_females = pop_img.select(female_bands)
        total_pop = pop_img.select(['population'])
        
        # Sommes par sexe
        males_sum = selected_males.reduce(ee.Reducer.sum()).rename('garcons')
        females_sum = selected_females.reduce(ee.Reducer.sum()).rename('filles')
        enfants = males_sum.add(females_sum).rename('enfants')
        
        # ========== MOSAÏQUE AVEC TOUTES LES BANDES ==========
        final_mosaic = (total_pop
                       .addBands(selected_males)      # Bandes M_0, M_1, M_5, M_10
                       .addBands(selected_females)    # Bandes F_0, F_1, F_5, F_10
                       .addBands(males_sum)
                       .addBands(females_sum)
                       .addBands(enfants))
        # ====================================================
        
        # Conversion densité → compte absolu
        pixel_area = ee.Image.pixelArea().divide(10000)
        final_mosaic_count = final_mosaic.multiply(pixel_area)
        
        status_text.text("🗺️ Conversion géométries...")
        features = []
        for idx, row in _sa_gdf.iterrows():
            geom = row['geometry']
            props = {"health_area": row["health_area"]}
            
            if geom.geom_type == 'Polygon':
                coords = [[[x, y] for x, y in geom.exterior.coords]]
                ee_geom = ee.Geometry.Polygon(coords)
            elif geom.geom_type == 'MultiPolygon':
                coords = []
                for poly in geom.geoms:
                    coords.append([[[x, y] for x, y in poly.exterior.coords]])
                ee_geom = ee.Geometry.MultiPolygon(coords)
            else:
                continue
            
            features.append(ee.Feature(ee_geom, props))
        
        fc = ee.FeatureCollection(features)
        
        status_text.text("🔢 Calcul statistiques zonales...")
        stats = final_mosaic_count.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.sum(),
            scale=100,
            crs='EPSG:4326'
        )
        
        status_text.text("📊 Extraction résultats...")
        stats_info = stats.getInfo()
        
        data_list = []
        total_aires = len(stats_info['features'])
        
        for i, feat in enumerate(stats_info['features']):
            props = feat['properties']
            
            # ========== EXTRACTION DÉTAILLÉE ==========
            pop_totale = props.get("population", 0)
            garcons = props.get("garcons", 0)
            filles = props.get("filles", 0)
            enfants_total = props.get("enfants", 0)
            
            # Extraire chaque tranche d'âge individuellement
            m_0 = props.get("M_0", 0)
            m_1 = props.get("M_1", 0)
            m_5 = props.get("M_5", 0)
            m_10 = props.get("M_10", 0)
            
            f_0 = props.get("F_0", 0)
            f_1 = props.get("F_1", 0)
            f_5 = props.get("F_5", 0)
            f_10 = props.get("F_10", 0)
            # ==========================================
            
            data_list.append({
                "health_area": props.get("health_area", ""),
                "Pop_Totale": int(pop_totale) if pop_totale > 0 else np.nan,
                "Pop_Garcons": int(garcons),
                "Pop_Filles": int(filles),
                "Pop_Enfants": int(enfants_total),
                # Nouvelles colonnes pour pyramide
                "Pop_M_0": int(m_0),
                "Pop_M_1": int(m_1),
                "Pop_M_5": int(m_5),
                "Pop_M_10": int(m_10),
                "Pop_F_0": int(f_0),
                "Pop_F_1": int(f_1),
                "Pop_F_5": int(f_5),
                "Pop_F_10": int(f_10)
            })
            
            progress_value = min((i + 1) / total_aires, 1.0)
            progress_bar.progress(progress_value)
        
        progress_bar.empty()
        status_text.text("✅ WorldPop terminé")
        
        return pd.DataFrame(data_list)
        
    except Exception as e:
        st.sidebar.error(f"❌ WorldPop : {str(e)}")
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()
        return pd.DataFrame({
            "health_area": _sa_gdf["health_area"],
            "Pop_Totale": [np.nan] * len(_sa_gdf),
            "Pop_Garcons": [np.nan] * len(_sa_gdf),
            "Pop_Filles": [np.nan] * len(_sa_gdf),
            "Pop_Enfants": [np.nan] * len(_sa_gdf),
            "Pop_M_0": [np.nan] * len(_sa_gdf),
            "Pop_M_1": [np.nan] * len(_sa_gdf),
            "Pop_M_5": [np.nan] * len(_sa_gdf),
            "Pop_M_10": [np.nan] * len(_sa_gdf),
            "Pop_F_0": [np.nan] * len(_sa_gdf),
            "Pop_F_1": [np.nan] * len(_sa_gdf),
            "Pop_F_5": [np.nan] * len(_sa_gdf),
            "Pop_F_10": [np.nan] * len(_sa_gdf)
        })


# GHSL - Classification urbaine
@st.cache_data
def urban_classification(_sa_gdf, use_gee):
    if not use_gee:
        st.sidebar.warning("⚠️ GHSL : GEE indisponible")
        return pd.DataFrame({
            "health_area": _sa_gdf["health_area"],
            "Urbanisation": [np.nan] * len(_sa_gdf)
        })
    
    try:
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        status_text.text("🏙️ Classification urbaine...")
        
        features = []
        for idx, row in _sa_gdf.iterrows():
            geom = row['geometry']
            props = {"health_area": row["health_area"]}
            
            if geom.geom_type == 'Polygon':
                coords = [[[x, y] for x, y in geom.exterior.coords]]
                ee_geom = ee.Geometry.Polygon(coords)
            elif geom.geom_type == 'MultiPolygon':
                coords = []
                for poly in geom.geoms:
                    coords.append([[[x, y] for x, y in poly.exterior.coords]])
                ee_geom = ee.Geometry.MultiPolygon(coords)
            else:
                continue
            
            features.append(ee.Feature(ee_geom, props))
        
        fc = ee.FeatureCollection(features)
        smod = ee.Image("JRC/GHSL/P2023A/GHS_SMOD/2020")
        
        def classify(feature):
            stats = smod.reduceRegion(
                ee.Reducer.mode(),
                feature.geometry(),
                scale=1000,
                maxPixels=1e9
            )
            smod_value = ee.Number(stats.get("smod_code")).toInt()
            urbanisation = ee.Algorithms.If(
                smod_value.gte(30),
                "Urbain",
                ee.Algorithms.If(smod_value.eq(23), "Semi-urbain", "Rural")
            )
            return feature.set({"Urbanisation": urbanisation})
        
        urban_fc = fc.map(classify)
        urban_info = urban_fc.getInfo()
        
        data_list = []
        total_aires = len(urban_info['features'])
        
        for i, feat in enumerate(urban_info['features']):
            props = feat['properties']
            data_list.append({
                "health_area": props.get("health_area", ""),
                "Urbanisation": props.get("Urbanisation", "Rural")
            })
            progress_value = min((i + 1) / total_aires, 1.0)
            progress_bar.progress(progress_value)
        
        progress_bar.empty()
        status_text.text("✅ GHSL terminé")
        
        return pd.DataFrame(data_list)
        
    except Exception as e:
        st.sidebar.error(f"❌ GHSL : {str(e)}")
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()
        return pd.DataFrame({
            "health_area": _sa_gdf["health_area"],
            "Urbanisation": [np.nan] * len(_sa_gdf)
        })

# NASA POWER - Données climatiques
@st.cache_data(ttl=86400)
def fetch_climate_nasa_power(_sa_gdf, start_date, end_date):
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    data_list = []
    total_aires = len(_sa_gdf)
    
    for idx, row in _sa_gdf.iterrows():
        status_text.text(f"🌡️ Climat {idx+1}/{total_aires}...")
        
        lat, lon = row.geometry.centroid.y, row.geometry.centroid.x
        
        url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            "parameters": "T2M,PRECTOTCORR,RH2M",
            "community": "AG",
            "longitude": lon,
            "latitude": lat,
            "start": start_date.strftime("%Y%m%d"),
            "end": end_date.strftime("%Y%m%d"),
            "format": "JSON"
        }
        
        try:
            r = requests.get(url, params=params, timeout=30)
            j = r.json()
            
            if "properties" in j and "parameter" in j["properties"]:
                p = j["properties"]["parameter"]
                
                temp_values = list(p.get("T2M", {}).values())
                rh_values = list(p.get("RH2M", {}).values())
                
                temp_mean = np.nanmean(temp_values) if temp_values else np.nan
                rh_mean = np.nanmean(rh_values) if rh_values else np.nan
                
                saison_seche_hum = rh_mean * 0.7 if not np.isnan(rh_mean) else np.nan
                
                data_list.append({
                    "health_area": row["health_area"],
                    "Temperature_Moy": temp_mean,
                    "Humidite_Moy": rh_mean,
                    "Saison_Seche_Humidite": saison_seche_hum
                })
            else:
                data_list.append({
                    "health_area": row["health_area"],
                    "Temperature_Moy": np.nan,
                    "Humidite_Moy": np.nan,
                    "Saison_Seche_Humidite": np.nan
                })
        except:
            data_list.append({
                "health_area": row["health_area"],
                "Temperature_Moy": np.nan,
                "Humidite_Moy": np.nan,
                "Saison_Seche_Humidite": np.nan
            })
        
        progress_value = min((idx + 1) / total_aires, 1.0)
        progress_bar.progress(progress_value)
    
    progress_bar.empty()
    status_text.text("✅ Climat terminé")
    
    return pd.DataFrame(data_list)

# Enrichissement du GeoDataFrame
with st.spinner("🔄 Enrichissement des données..."):
    pop_df = worldpop_children_stats(sa_gdf, gee_ok)
    urban_df = urban_classification(sa_gdf, gee_ok)
    climate_df = fetch_climate_nasa_power(sa_gdf, start_date, end_date)

sa_gdf_enrichi = sa_gdf.copy()
sa_gdf_enrichi = sa_gdf_enrichi.merge(pop_df, on="health_area", how="left")
sa_gdf_enrichi = sa_gdf_enrichi.merge(urban_df, on="health_area", how="left")
sa_gdf_enrichi = sa_gdf_enrichi.merge(climate_df, on="health_area", how="left")

if vaccination_df is not None:
    sa_gdf_enrichi = sa_gdf_enrichi.merge(vaccination_df, on="health_area", how="left")
else:
    sa_gdf_enrichi["Taux_Vaccination"] = np.nan

# reprojection en métrique égale-aire
sa_gdf_m = sa_gdf_enrichi.to_crs("ESRI:54009")  # Mollweide (mètres)

sa_gdf_enrichi["Superficie_km2"] = sa_gdf_m.geometry.area / 1e6

sa_gdf_enrichi["Densite_Pop"] = (
    sa_gdf_enrichi["Pop_Totale"] / sa_gdf_enrichi["Superficie_km2"].replace(0, np.nan)
)

sa_gdf_enrichi["Densite_Enfants"] = (
    sa_gdf_enrichi["Pop_Enfants"] / sa_gdf_enrichi["Superficie_km2"].replace(0, np.nan)
)

sa_gdf_enrichi = sa_gdf_enrichi.replace([np.inf, -np.inf], np.nan)

st.sidebar.success("✓ Enrichissement terminé")

st.sidebar.markdown("---")
st.sidebar.subheader("📋 Données disponibles")

donnees_dispo = {
    "Population": not sa_gdf_enrichi["Pop_Totale"].isna().all(),
    "Urbanisation": not sa_gdf_enrichi["Urbanisation"].isna().all(),
    "Climat": not sa_gdf_enrichi["Humidite_Moy"].isna().all(),
    "Vaccination": not sa_gdf_enrichi["Taux_Vaccination"].isna().all()
}

for nom, dispo in donnees_dispo.items():
    icone = "✅" if dispo else "❌"
    st.sidebar.text(f"{icone} {nom}")

# ============================================================
# PARTIE 4/6 - KPIS, CARTE ET ANALYSES
# ============================================================

# KPIs
st.header("📊 Indicateurs Clés de Performance")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("📈 Cas totaux", f"{len(df):,}")

with col2:
    if "Statut_Vaccinal" in df.columns and df["Statut_Vaccinal"].notna().sum() > 0 and (df["Statut_Vaccinal"] != "Inconnu").sum() > 0:
        taux_non_vac = (df["Statut_Vaccinal"] == "Non").mean() * 100
        delta_vac = taux_non_vac - 45
        st.metric("💉 Non vaccinés", f"{taux_non_vac:.1f}%", delta=f"{delta_vac:+.1f}%")
    else:
        st.metric("💉 Non vaccinés", "N/A")

with col3:
    if "Age_Mois" in df.columns and df["Age_Mois"].notna().sum() > 0:
        age_median = df["Age_Mois"].median()
        st.metric("👶 Âge médian", f"{int(age_median)} mois")
    else:
        st.metric("👶 Âge médian", "N/A")

with col4:
    if "Issue" in df.columns and df["Issue"].notna().sum() > 0 and (df["Issue"] == "Décédé").sum() > 0:
        taux_deces = (df["Issue"] == "Décédé").mean() * 100
        st.metric("☠️ Létalité", f"{taux_deces:.2f}%")
    else:
        st.metric("☠️ Létalité", "N/A")

with col5:
    n_aires_touchees = df["Aire_Sante"].nunique()
    pct_aires = (n_aires_touchees / len(sa_gdf)) * 100
    st.metric("🗺️ Aires touchées", f"{n_aires_touchees}/{len(sa_gdf)}", delta=f"{pct_aires:.0f}%")


# ============================================================
# TOP 10 AIRES DE SANTÉ - PAR TAUX D'ATTAQUE ET PAR CAS
# ============================================================

st.subheader("🏆 Top 10 Aires de Santé")

# Calculer les statistiques par aire
aggdict = {'ID_Cas': 'count'}
if 'Age_Mois' in df.columns:
    aggdict['Age_Mois'] = 'mean'
if 'Statut_Vaccinal' in df.columns:
    aggdict['Statut_Vaccinal'] = lambda x: ((x == 'Non').sum() / len(x) * 100)

cases_by_area = df.groupby('Aire_Sante').agg(aggdict).reset_index()

rename_map = {'ID_Cas': 'Cas_Observes'}
if 'Age_Mois' in cases_by_area.columns:
    rename_map['Age_Mois'] = 'Age_Moyen'
if 'Statut_Vaccinal' in cases_by_area.columns:
    rename_map['Statut_Vaccinal'] = 'Taux_Non_Vaccines'

cases_by_area = cases_by_area.rename(columns=rename_map)

if 'Taux_Non_Vaccines' not in cases_by_area.columns:
    cases_by_area['Taux_Non_Vaccines'] = 0
if 'Age_Moyen' not in cases_by_area.columns:
    cases_by_area['Age_Moyen'] = 0

# 🔧 CORRECTION : Fusionner avec les données géographiques
cases_by_area = cases_by_area.merge(
    sa_gdf_enrichi[['health_area', 'Pop_Totale', 'Pop_Enfants']],
    left_on='Aire_Sante',
    right_on='health_area',
    how='left'
)

# 🔧 CORRECTION : Gérer les aires sans correspondance
cases_by_area['Pop_Totale'] = cases_by_area['Pop_Totale'].fillna(0)
cases_by_area['Pop_Enfants'] = cases_by_area['Pop_Enfants'].fillna(0)

# Calculer le taux d'attaque pour 10 000 habitants
cases_by_area['Taux_Attaque_10K'] = (
    (cases_by_area['Cas_Observes'] / cases_by_area['Pop_Totale'].replace(0, np.nan)) * 10000
).fillna(0)

# Calculer le taux d'attaque pour 10 000 enfants
cases_by_area['Taux_Attaque_Enfants_10K'] = (
    (cases_by_area['Cas_Observes'] / cases_by_area['Pop_Enfants'].replace(0, np.nan)) * 10000
).fillna(0)

# Créer deux onglets
tab1, tab2 = st.tabs(["📈 Par Taux d'Attaque", "🔢 Par Nombre de Cas"])

with tab1:
    st.markdown("**Top 10 - Aires avec le plus haut taux d'attaque (pour 10 000 habitants)**")
    
    top10_taux = cases_by_area.nlargest(10, 'Taux_Attaque_10K')
    
    fig_taux = px.bar(
        top10_taux,
        x='Taux_Attaque_10K',
        y='Aire_Sante',
        orientation='h',
        title='Top 10 - Taux d\'attaque le plus élevé',
        labels={'Taux_Attaque_10K': 'Taux pour 10K hab.', 'Aire_Sante': 'Aire de santé'},
        color='Taux_Attaque_10K',
        color_continuous_scale='Reds',
        text='Taux_Attaque_10K'
    )
    fig_taux.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig_taux.update_layout(height=500)
    st.plotly_chart(fig_taux, use_container_width=True)
    
    # Tableau détaillé
    st.dataframe(
        top10_taux[['Aire_Sante', 'Cas_Observes', 'Pop_Totale', 'Taux_Attaque_10K']]
        .style.format({
            'Cas_Observes': '{:.0f}',
            'Pop_Totale': '{:,.0f}',
            'Taux_Attaque_10K': '{:.2f}'
        })
        .background_gradient(subset=['Taux_Attaque_10K'], cmap='Reds'),
        use_container_width=True
    )

with tab2:
    st.markdown("**Top 10 - Aires avec le plus grand nombre de cas**")
    
    top10_cas = cases_by_area.nlargest(10, 'Cas_Observes')
    
    fig_cas = px.bar(
        top10_cas,
        x='Cas_Observes',
        y='Aire_Sante',
        orientation='h',
        title='Top 10 - Nombre de cas le plus élevé',
        labels={'Cas_Observes': 'Nombre de cas', 'Aire_Sante': 'Aire de santé'},
        color='Cas_Observes',
        color_continuous_scale='Oranges',
        text='Cas_Observes'
    )
    fig_cas.update_traces(textposition='outside')
    fig_cas.update_layout(height=500)
    st.plotly_chart(fig_cas, use_container_width=True)
    
    # Tableau détaillé
    st.dataframe(
        top10_cas[['Aire_Sante', 'Cas_Observes', 'Pop_Totale', 'Taux_Attaque_10K']]
        .style.format({
            'Cas_Observes': '{:.0f}',
            'Pop_Totale': '{:,.0f}',
            'Taux_Attaque_10K': '{:.2f}'
        })
        .background_gradient(subset=['Cas_Observes'], cmap='Oranges'),
        use_container_width=True
    )

# Métriques récapitulatives
col1, col2, col3 = st.columns(3)

with col1:
    if len(cases_by_area[cases_by_area['Taux_Attaque_10K'] > 0]) > 0:
        taux_max = cases_by_area['Taux_Attaque_10K'].max()
        aire_taux_max = cases_by_area.loc[cases_by_area['Taux_Attaque_10K'].idxmax(), 'Aire_Sante']
        st.metric("Taux d'attaque max", f"{taux_max:.1f}/10K", aire_taux_max)
    else:
        st.metric("Taux d'attaque max", "N/A")

with col2:
    taux_moyen = cases_by_area['Taux_Attaque_10K'].mean()
    st.metric("Taux d'attaque moyen", f"{taux_moyen:.1f}/10K")

with col3:
    aires_alerte = len(cases_by_area[cases_by_area['Taux_Attaque_10K'] > 10])  # Seuil OMS
    st.metric("Aires en alerte (>10/10K)", aires_alerte, delta_color="inverse")

# ============================================================
# FUSION AVEC LE GEODATAFRAME (IMPORTANT - NE PAS SUPPRIMER)
# ============================================================

# 🔧 CORRECTION : Utiliser Aire_Sante au lieu de health_area
sa_gdf_with_cases = sa_gdf_enrichi.merge(
    cases_by_area[['Aire_Sante', 'Cas_Observes', 'Taux_Non_Vaccines', 'Age_Moyen', 'Taux_Attaque_10K']],
    left_on='health_area',
    right_on='Aire_Sante',
    how='left'
)

# Remplir les valeurs manquantes
sa_gdf_with_cases['Cas_Observes'] = sa_gdf_with_cases['Cas_Observes'].fillna(0).astype(int)
sa_gdf_with_cases['Taux_Non_Vaccines'] = sa_gdf_with_cases['Taux_Non_Vaccines'].fillna(0)
sa_gdf_with_cases['Age_Moyen'] = sa_gdf_with_cases['Age_Moyen'].fillna(0)
sa_gdf_with_cases['Taux_Attaque_10K'] = sa_gdf_with_cases['Taux_Attaque_10K'].fillna(0)

# 🔧 CORRECTION : Créer Taux_Attaque_10000 à partir de Taux_Attaque_10K
sa_gdf_with_cases['Taux_Attaque_10000'] = sa_gdf_with_cases['Taux_Attaque_10K']


# ============================================================
# LA SECTION "CARTOGRAPHIE" COMMENCE ICI
# ============================================================

st.header("Cartographie de la Situation Actuelle")

center_lat = sa_gdf_with_cases.geometry.centroid.y.mean()
center_lon = sa_gdf_with_cases.geometry.centroid.x.mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=6,
    tiles="CartoDB positron",
    control_scale=True
)

import branca.colormap as cm

max_cases = sa_gdf_with_cases["Cas_Observes"].max()

if max_cases > 0:
    colormap = cm.LinearColormap(
        colors=['#e8f5e9', '#81c784', '#ffeb3b', '#ff9800', '#f44336', '#b71c1c'],
        vmin=0,
        vmax=max_cases,
        caption="Nombre de cas observés"
    )
    colormap.add_to(m)

for idx, row in sa_gdf_with_cases.iterrows():
    aire_name = row['health_area']
    cas_obs = int(row.get('Cas_Observes', 0))
    pop_enfants = row.get('Pop_Enfants', np.nan)
    taux_attaque = row.get('Taux_Attaque_10000', np.nan)
    urbanisation = row.get('Urbanisation', 'N/A')
    densite = row.get('Densite_Pop', np.nan)
    
    popup_html = f"""
    <div style="font-family: Arial; width: 350px;">
        <h3 style="margin-bottom: 10px; color: #1976d2; border-bottom: 2px solid #1976d2;">
            {aire_name}
        </h3>
        <div style="background-color: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px;">
            <h4 style="margin: 0; color: #d32f2f;">📊 Situation Épidémiologique</h4>
            <table style="width: 100%; margin-top: 5px;">
                <tr><td><b>Cas observés :</b></td><td style="text-align: right;">
                    <b style="font-size: 18px; color: #d32f2f;">{cas_obs}</b>
                </td></tr>
                <tr><td>Population enfants :</td><td style="text-align: right;">
                    {f"{int(pop_enfants):,}" if not np.isnan(pop_enfants) else "N/A"}
                </td></tr>
                <tr><td>Taux d'attaque :</td><td style="text-align: right;">
                    {f"{taux_attaque:.1f}/10K" if not np.isnan(taux_attaque) else "N/A"}
                </td></tr>
                <tr><td>Type habitat :</td><td style="text-align: right;">
                    <b>{urbanisation if pd.notna(urbanisation) else "N/A"}</b>
                </td></tr>
                <tr><td>Densité pop :</td><td style="text-align: right;">
                    {f"{densite:.1f} hab/km²" if not np.isnan(densite) else "N/A"}
                </td></tr>
            </table>
        </div>
    </div>
    """
    
    fill_color = colormap(row['Cas_Observes']) if max_cases > 0 else '#e0e0e0'
    
    if row['Cas_Observes'] >= seuil_alerte_epidemique:
        line_color = '#b71c1c'
        line_weight = 2
    else:
        line_color = 'black'
        line_weight = 0.5
    
    folium.GeoJson(
        row['geometry'],
        style_function=lambda x, color=fill_color, weight=line_weight, border=line_color: {
            'fillColor': color,
            'color': border,
            'weight': weight,
            'fillOpacity': 0.7
        },
        tooltip=folium.Tooltip(
            f"<b>{aire_name}</b><br>{cas_obs} cas",
            sticky=True
        ),
        popup=folium.Popup(popup_html, max_width=400)
    ).add_to(m)
    
    if cas_obs > 0:
        folium.Marker(
            location=[row.geometry.centroid.y, row.geometry.centroid.x],
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    font-size: 7pt;
                    color: black;
                    font-weight: normal;
                    background: none;
                    padding: 0;
                    border: none;
                    box-shadow: none;
                    white-space: nowrap;
                ">{aire_name}</div>
                """
            ),
        ).add_to(m)


heat_data = [
    [row.geometry.centroid.y, row.geometry.centroid.x, row['Cas_Observes']]
    for idx, row in sa_gdf_with_cases.iterrows()
    if row['Cas_Observes'] > 0
]

if heat_data:
    HeatMap(
        heat_data,
        radius=20,
        blur=25,
        max_zoom=13,
        gradient={0.0: 'blue', 0.5: 'yellow', 1.0: 'red'}
    ).add_to(m)

legend_html = f'''
<div style="
    position: fixed;
    bottom: 50px;
    left: 50px;
    width: 250px;
    background-color: white;
    border: 2px solid grey;
    z-index:9999;
    font-size:14px;
    padding: 10px;
    border-radius: 5px;">
    <p style="margin: 0; font-weight: bold;">📊 Légende</p>
    <p style="margin: 5px 0;">
        <span style="background-color: #e8f5e9; padding: 2px 8px;">Faible</span>
        0-{max_cases//3:.0f} cas
    </p>
    <p style="margin: 5px 0;">
        <span style="background-color: #ffeb3b; padding: 2px 8px;">Moyen</span>
        {max_cases//3:.0f}-{2*max_cases//3:.0f} cas
    </p>
    <p style="margin: 5px 0;">
        <span style="background-color: #f44336; padding: 2px 8px; color: white;">Élevé</span>
        >{2*max_cases//3:.0f} cas
    </p>
    <p style="margin: 5px 0; padding-top: 5px; border-top: 1px solid #ccc;">
        <b>Seuil alerte :</b> {seuil_alerte_epidemique} cas/sem
    </p>
</div>
'''

m.get_root().html.add_child(folium.Element(legend_html))

st_folium(m, width=1400, height=650)

col1, col2, col3 = st.columns(3)

with col1:
    aires_alerte = len(sa_gdf_with_cases[sa_gdf_with_cases['Cas_Observes'] >= seuil_alerte_epidemique])
    st.metric("🚨 Aires en alerte", aires_alerte, f"{aires_alerte/len(sa_gdf)*100:.1f}%")

with col2:
    aires_sans_cas = len(sa_gdf_with_cases[sa_gdf_with_cases['Cas_Observes'] == 0])
    st.metric("✅ Aires sans cas", aires_sans_cas, f"{aires_sans_cas/len(sa_gdf)*100:.1f}%")

with col3:
    densite_pop_moy = sa_gdf_with_cases['Densite_Pop'].mean()
    st.metric("📍 Densité pop. moy.", f"{densite_pop_moy:.1f} hab/km²")
# Analyse temporelle
st.header("📈 Analyse Temporelle par Semaines Épidémiologiques")

weekly_cases = df.groupby(['Annee', 'Semaine_Epi']).size().reset_index(name='Cas')
weekly_cases['Semaine_Label'] = weekly_cases['Annee'].astype(str) + '-S' + weekly_cases['Semaine_Epi'].astype(str).str.zfill(2)

fig_epi = go.Figure()

fig_epi.add_trace(go.Scatter(
    x=weekly_cases['Semaine_Label'],
    y=weekly_cases['Cas'],
    mode='lines+markers',
    name='Cas observés',
    line=dict(color='#d32f2f', width=3),
    marker=dict(size=6),
    hovertemplate='<b>%{x}</b><br>Cas : %{y}<extra></extra>'
))

from scipy.signal import savgol_filter

if len(weekly_cases) > 5:
    tendance = savgol_filter(
        weekly_cases['Cas'],
        window_length=min(7, len(weekly_cases) if len(weekly_cases) % 2 == 1 else len(weekly_cases)-1),
        polyorder=2
    )
    fig_epi.add_trace(go.Scatter(
        x=weekly_cases['Semaine_Label'],
        y=tendance,
        mode='lines',
        name='Tendance',
        line=dict(color='#1976d2', width=2, dash='dash'),
        hovertemplate='<b>%{x}</b><br>Tendance : %{y:.1f}<extra></extra>'
    ))

fig_epi.add_hline(
    y=seuil_alerte_epidemique,
    line_dash="dot",
    line_color="orange",
    annotation_text=f"Seuil d'alerte ({seuil_alerte_epidemique} cas/sem)",
    annotation_position="right"
)

fig_epi.update_layout(
    title="Courbe épidémique par semaines épidémiologiques",
    xaxis_title="Semaine épidémiologique",
    yaxis_title="Nombre de cas",
    hovermode='x unified',
    height=400
)

st.plotly_chart(fig_epi, use_container_width=True)

col1, col2, col3 = st.columns(3)

with col1:
    semaine_max = weekly_cases.loc[weekly_cases['Cas'].idxmax()]
    st.metric(
        "🔴 Semaine avec pic maximal",
        semaine_max['Semaine_Label'],
        f"{int(semaine_max['Cas'])} cas"
    )

with col2:
    cas_moyen_semaine = weekly_cases['Cas'].mean()
    st.metric("📊 Moyenne hebdomadaire", f"{cas_moyen_semaine:.1f} cas")

with col3:
    if len(weekly_cases) >= 2:
        variation = weekly_cases.iloc[-1]['Cas'] - weekly_cases.iloc[-2]['Cas']
        cas_precedent = weekly_cases.iloc[-2]['Cas']
        pct_variation = (variation / cas_precedent * 100) if cas_precedent > 0 else 0
        st.metric("📉 Variation dernière semaine", f"{int(variation):+d} cas", f"{pct_variation:+.1f}%")
    else:
        st.metric("📉 Variation dernière semaine", "N/A")

# ============================================================
# DISTRIBUTION PAR TRANCHES D'ÂGE (SI DISPONIBLE)
# ============================================================

# Vérifier si les données d'âge sont disponibles et valides
age_disponible = 'Age_Mois' in df.columns and df['Age_Mois'].notna().sum() > 0 and (df['Age_Mois'] > 0).sum() > 0

if age_disponible:
    st.subheader("📊 Distribution par Tranches d'Âge")
    
    df['Tranche_Age'] = pd.cut(
        df['Age_Mois'],
        bins=[0, 12, 60, 120, 180],
        labels=['0-1 an', '1-5 ans', '5-10 ans', '10-15 ans']
    )
    
    agg_dict_age = {'ID_Cas': 'count'}
    
    # Ajouter vaccination seulement si disponible
    if 'Statut_Vaccinal' in df.columns and df['Statut_Vaccinal'].notna().sum() > 0:
        agg_dict_age['Statut_Vaccinal'] = lambda x: ((x == 'Non').sum() / len(x) * 100) if len(x) > 0 else 0
    
    age_stats = df.groupby('Tranche_Age').agg(agg_dict_age).reset_index()
    
    rename_map_age = {'ID_Cas': 'Nombre_Cas'}
    if 'Statut_Vaccinal' in age_stats.columns:
        rename_map_age['Statut_Vaccinal'] = 'Pct_Non_Vaccines'
    
    age_stats = age_stats.rename(columns=rename_map_age)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_age = px.bar(
            age_stats,
            x='Tranche_Age',
            y='Nombre_Cas',
            title='Cas par tranche d\'âge',
            color='Nombre_Cas',
            color_continuous_scale='Reds',
            text='Nombre_Cas'
        )
        fig_age.update_traces(textposition='outside')
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        if 'Pct_Non_Vaccines' in age_stats.columns and age_stats['Pct_Non_Vaccines'].sum() > 0:
            fig_vacc_age = px.bar(
                age_stats,
                x='Tranche_Age',
                y='Pct_Non_Vaccines',
                title='% non vaccinés par âge',
                color='Pct_Non_Vaccines',
                color_continuous_scale='Oranges',
                text='Pct_Non_Vaccines'
            )
            fig_vacc_age.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig_vacc_age, use_container_width=True)
        else:
            st.info("ℹ️ Données de vaccination par âge non disponibles")
else:
    st.info("ℹ️ Données d'âge non disponibles - Section âge masquée")

# ============================================================
# PYRAMIDE DES ÂGES POPULATION (WORLDPOP)
# ============================================================

st.header("📊 Pyramide des Âges - Population Enfantine")

if donnees_dispo["Population"]:
    # Préparer les données de population par tranche d'âge
    pyramid_data = []
    
    for idx, row in sa_gdf_enrichi.iterrows():
        aire = row['health_area']
        
        # Récupérer les valeurs de population par tranche d'âge depuis WorldPop
        # Si vous avez déjà extrait ces données individuellement
        pop_0_1_m = row.get('Pop_M_0', 0) + row.get('Pop_M_1', 0)  # 0-4 ans garçons
        pop_5_9_m = row.get('Pop_M_5', 0)  # 5-9 ans garçons
        pop_10_14_m = row.get('Pop_M_10', 0)  # 10-14 ans garçons
        
        pop_0_1_f = row.get('Pop_F_0', 0) + row.get('Pop_F_1', 0)  # 0-4 ans filles
        pop_5_9_f = row.get('Pop_F_5', 0)  # 5-9 ans filles
        pop_10_14_f = row.get('Pop_F_10', 0)  # 10-14 ans filles
        
        pyramid_data.append({
            'Aire': aire,
            'Garçons_0-4': pop_0_1_m,
            'Garçons_5-9': pop_5_9_m,
            'Garçons_10-14': pop_10_14_m,
            'Filles_0-4': pop_0_1_f,
            'Filles_5-9': pop_5_9_f,
            'Filles_10-14': pop_10_14_f
        })
    
    pyramid_df = pd.DataFrame(pyramid_data)
    
    # Agréger pour tout le territoire
    total_garcons_0_4 = pyramid_df['Garçons_0-4'].sum()
    total_garcons_5_9 = pyramid_df['Garçons_5-9'].sum()
    total_garcons_10_14 = pyramid_df['Garçons_10-14'].sum()
    
    total_filles_0_4 = pyramid_df['Filles_0-4'].sum()
    total_filles_5_9 = pyramid_df['Filles_5-9'].sum()
    total_filles_10_14 = pyramid_df['Filles_10-14'].sum()
    
    # Créer le dataframe pour la pyramide
    age_groups = ['0-4', '5-9', '10-14']
    
    pyramid_plot_df = pd.DataFrame({
        'Age': age_groups,
        'Garçons': [-total_garcons_0_4, -total_garcons_5_9, -total_garcons_10_14],  # Négatif pour gauche
        'Filles': [total_filles_0_4, total_filles_5_9, total_filles_10_14]  # Positif pour droite
    })
    
    # Créer la pyramide avec Plotly
    fig_pyramid = go.Figure()
    
    # Barres garçons (gauche - négatif)
    fig_pyramid.add_trace(go.Bar(
        y=pyramid_plot_df['Age'],
        x=pyramid_plot_df['Garçons'],
        name='Garçons',
        orientation='h',
        marker=dict(color='#42a5f5'),
        text=[f"{abs(x):,.0f}" for x in pyramid_plot_df['Garçons']],
        textposition='inside',
        hovertemplate='<b>%{y} ans</b><br>Garçons: %{text}<extra></extra>'
    ))
    
    # Barres filles (droite - positif)
    fig_pyramid.add_trace(go.Bar(
        y=pyramid_plot_df['Age'],
        x=pyramid_plot_df['Filles'],
        name='Filles',
        orientation='h',
        marker=dict(color='#ec407a'),
        text=[f"{x:,.0f}" for x in pyramid_plot_df['Filles']],
        textposition='inside',
        hovertemplate='<b>%{y} ans</b><br>Filles: %{text}<extra></extra>'
    ))
    
    # Calculer les limites symétriques
    max_val = max(
        abs(pyramid_plot_df['Garçons'].min()),
        pyramid_plot_df['Filles'].max()
    )
    
    fig_pyramid.update_layout(
        title='Pyramide des Âges - Population Enfantine (0-14 ans)',
        xaxis=dict(
            title='Population',
            tickvals=[-max_val, -max_val/2, 0, max_val/2, max_val],
            ticktext=[f"{int(max_val):,}", f"{int(max_val/2):,}", "0", 
                     f"{int(max_val/2):,}", f"{int(max_val):,}"],
            range=[-max_val * 1.1, max_val * 1.1]
        ),
        yaxis=dict(title='Tranche d\'âge'),
        barmode='overlay',
        height=400,
        bargap=0.1,
        showlegend=True,
        legend=dict(x=0.85, y=0.95),
        hovermode='y unified'
    )
    
    st.plotly_chart(fig_pyramid, use_container_width=True)
    
    # Statistiques complémentaires
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_garcons = total_garcons_0_4 + total_garcons_5_9 + total_garcons_10_14
        st.metric("👦 Garçons (0-14 ans)", f"{int(total_garcons):,}")
    
    with col2:
        total_filles = total_filles_0_4 + total_filles_5_9 + total_filles_10_14
        st.metric("👧 Filles (0-14 ans)", f"{int(total_filles):,}")
    
    with col3:
        ratio = (total_garcons / total_filles * 100) if total_filles > 0 else 0
        st.metric("⚖️ Ratio G/F", f"{ratio:.1f}%")

else:
    st.info("📊 Données de population non disponibles. Pyramide des âges non affichable.")

# Nowcasting
st.subheader("⏱️ Nowcasting - Correction des Délais de Notification")

st.info("""
**Nowcasting (Prévision immédiate) :** Technique d'ajustement permettant d'estimer le nombre réel de cas en tenant compte des délais de notification.
""")

if "Date_Notification" in df.columns and "Date_Debut_Eruption" in df.columns:
    df["Delai_Notification"] = (df["Date_Notification"] - df["Date_Debut_Eruption"]).dt.days
    delai_available = True
else:
    df["Delai_Notification"] = 3  # Valeur par défaut
    delai_available = False

delai_moyen = df["Delai_Notification"].mean()
delai_median = df["Delai_Notification"].median()
delai_std = df["Delai_Notification"].std()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Délai moyen de notification", f"{delai_moyen:.1f} jours" if delai_available and not np.isnan(delai_moyen) else "N/A")

with col2:
    st.metric("Délai médian", f"{delai_median:.0f} jours" if delai_available and not np.isnan(delai_median) else "N/A")

with col3:
    st.metric("Écart-type", f"{delai_std:.1f} jours" if delai_available and not np.isnan(delai_std) else "N/A")

with col4:
    derniere_semaine_label = weekly_cases.iloc[-1]['Semaine_Label']
    cas_derniere_semaine = int(weekly_cases.iloc[-1]['Cas'])
    if delai_available and not np.isnan(delai_moyen):
        facteur_correction = 1 + (delai_moyen / 7)
        cas_corriges = int(cas_derniere_semaine * facteur_correction)
        st.metric(
            f"Cas corrigés ({derniere_semaine_label})",
            cas_corriges,
            delta=f"+{cas_corriges - cas_derniere_semaine}"
        )
    else:
        st.metric(
            f"Cas corrigés ({derniere_semaine_label})",
            cas_derniere_semaine,
            delta="N/A"
        )

if delai_available:
    fig_delai = px.histogram(
        df,
        x="Delai_Notification",
        nbins=20,
        title="Distribution des délais de notification",
        labels={"Delai_Notification": "Délai (jours)", "count": "Nombre de cas"},
        color_discrete_sequence=['#d32f2f']
    )

    fig_delai.add_vline(x=delai_moyen, line_dash="dash", line_color="blue", annotation_text=f"Moyenne : {delai_moyen:.1f}j")
    fig_delai.add_vline(x=delai_median, line_dash="dash", line_color="green", annotation_text=f"Médiane : {delai_median:.0f}j")

    st.plotly_chart(fig_delai, use_container_width=True)
else:
    st.info("ℹ️ Données de délai de notification non disponibles")

# ============================================================
# MODÉLISATION PRÉDICTIVE - VERSION FINALE UNIQUE
# ============================================================

st.header("🔮 Modélisation Prédictive par Semaines Épidémiologiques")

st.markdown(f"""
<div class="info-box">
<b>Configuration de la prédiction :</b><br>
- Dernière semaine de données : <b>S{derniere_semaine_epi} ({derniere_annee})</b><br>
- Période de prédiction : <b>{pred_mois} mois ({n_weeks_pred} semaines)</b><br>
- Semaines prédites : <b>S{derniere_semaine_epi+1} à S{min(derniere_semaine_epi+n_weeks_pred, 52)}</b><br>
- Modèle sélectionné : <b>{modele_choisi}</b><br>
- Mode importance : <b>{mode_importance}</b><br>
- Seuils configurés : Baisse ≥{seuil_baisse}%, Hausse ≥{seuil_hausse}%
</div>
""", unsafe_allow_html=True)

if 'prediction_rougeole_lancee' not in st.session_state:
    st.session_state.prediction_rougeole_lancee = False

col1, col2 = st.columns([3, 1])

with col1:
    if st.button("🚀 Lancer la Modélisation Prédictive", type="primary", use_container_width=True, key="btn_model_rougeole"):
        st.session_state.prediction_rougeole_lancee = True

with col2:
    if st.button("🔄 Réinitialiser", use_container_width=True, key="btn_reset_rougeole"):
        st.session_state.prediction_rougeole_lancee = False

if not st.session_state.prediction_rougeole_lancee:
    st.info("👆 Cliquez sur le bouton ci-dessus pour lancer la modélisation")
    st.stop()

with st.spinner("🤖 Préparation des données et entraînement..."):
    
    weekly_features = df.groupby(["Aire_Sante", "Annee", "Semaine_Epi"]).agg(
        Cas_Observes=("ID_Cas", "count"),
        Non_Vaccines=("Statut_Vaccinal", lambda x: (x == "Non").mean() * 100),
        Age_Moyen=("Age_Mois", "mean")
    ).reset_index()
    
    weekly_features['Semaine_Label'] = (
        weekly_features['Annee'].astype(str) + '-S' +
        weekly_features['Semaine_Epi'].astype(str).str.zfill(2)
    )
    
    weekly_features = weekly_features.merge(
        sa_gdf_enrichi[[
            "health_area", "Pop_Totale", "Pop_Enfants",
            "Densite_Pop", "Densite_Enfants", "Urbanisation",
            "Temperature_Moy", "Humidite_Moy", "Saison_Seche_Humidite",
            "Taux_Vaccination"
        ]],
        left_on="Aire_Sante",
        right_on="health_area",
        how="left"
    )
    
    weekly_features['Age_Moyen'] = weekly_features['Age_Moyen'].fillna(weekly_features['Age_Moyen'].median())
    weekly_features['Non_Vaccines'] = weekly_features['Non_Vaccines'].fillna(
        weekly_features['Non_Vaccines'].mean() if weekly_features['Non_Vaccines'].notna().any() else 50.0
    )
    
    le_urban = LabelEncoder()
    weekly_features["Urban_Encoded"] = le_urban.fit_transform(
        weekly_features["Urbanisation"].fillna("Rural")
    )
    
    if donnees_dispo["Climat"]:
        scaler_climat = MinMaxScaler()
        climate_cols = ["Temperature_Moy", "Humidite_Moy", "Saison_Seche_Humidite"]
        
        for col in climate_cols:
            if col in weekly_features.columns:
                col_mean = weekly_features[col].mean()
                if pd.isna(col_mean):
                    col_mean = 0
                weekly_features[col] = weekly_features[col].fillna(col_mean)
        
        climate_data_to_scale = weekly_features[climate_cols].values
        climate_scaled = scaler_climat.fit_transform(climate_data_to_scale)
        
        for idx, col in enumerate(climate_cols):
            weekly_features[f"{col}_Norm"] = climate_scaled[:, idx]
        
        weekly_features["Coef_Climatique"] = (
            weekly_features.get("Temperature_Moy_Norm", 0) * 0.4 +
            weekly_features.get("Humidite_Moy_Norm", 0) * 0.4 +
            weekly_features.get("Saison_Seche_Humidite_Norm", 0) * 0.2
        )
    
    weekly_features = weekly_features.sort_values(['Aire_Sante', 'Annee', 'Semaine_Epi'])
    
    for lag in [1, 2, 3, 4]:
        weekly_features[f'Cas_Lag_{lag}'] = (
            weekly_features.groupby('Aire_Sante')['Cas_Observes'].shift(lag)
        )
    
    numeric_cols = weekly_features.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        weekly_features[col] = weekly_features[col].replace([np.inf, -np.inf], np.nan)
        col_mean = weekly_features[col].mean()
        if pd.isna(col_mean):
            col_mean = 0
        weekly_features[col] = weekly_features[col].fillna(col_mean)
    
    st.subheader("📚 Entraînement du Modèle")
    
    feature_cols = [
        "Cas_Observes", "Age_Moyen", "Semaine_Epi",
        "Cas_Lag_1", "Cas_Lag_2", "Cas_Lag_3", "Cas_Lag_4"
    ]
    
    feature_groups = {
        "Historique_Cas": ["Cas_Lag_1", "Cas_Lag_2", "Cas_Lag_3", "Cas_Lag_4"],
        "Vaccination": [],
        "Demographie": [],
        "Urbanisation": [],
        "Climat": []
    }
    
    if donnees_dispo["Population"]:
        feature_cols.extend(["Pop_Totale", "Pop_Enfants", "Densite_Pop", "Densite_Enfants"])
        feature_groups["Demographie"] = ["Pop_Totale", "Pop_Enfants", "Densite_Pop", "Densite_Enfants"]
        st.info("✅ Données démographiques intégrées au modèle")
    
    if donnees_dispo["Urbanisation"]:
        feature_cols.append("Urban_Encoded")
        feature_groups["Urbanisation"] = ["Urban_Encoded"]
        st.info("✅ Classification urbaine intégrée au modèle")
    
    if donnees_dispo["Climat"]:
        feature_cols.append("Coef_Climatique")
        feature_groups["Climat"] = ["Coef_Climatique"]
        st.info("✅ Coefficient climatique composite intégré au modèle")
    
    if donnees_dispo["Vaccination"]:
        feature_cols.extend(["Taux_Vaccination", "Non_Vaccines"])
        feature_groups["Vaccination"] = ["Taux_Vaccination", "Non_Vaccines"]
        st.info("✅ Données vaccinales intégrées au modèle")
    elif "Non_Vaccines" in weekly_features.columns:
        feature_cols.append("Non_Vaccines")
        feature_groups["Vaccination"] = ["Non_Vaccines"]
    
    st.markdown(f"**Variables utilisées :** {len(feature_cols)} features")
    
    nan_before = weekly_features[feature_cols].isna().sum()
    if nan_before.any():
        for col in feature_cols:
            weekly_features[col] = weekly_features[col].fillna(0)
    
    weekly_features_clean = weekly_features.dropna(subset=feature_cols)
    
    if len(weekly_features_clean) < 20:
        st.warning("⚠️ Données insuffisantes (minimum 20 observations requises)")
        st.stop()
    
    X = weekly_features_clean[feature_cols].copy()
    y = weekly_features_clean["Cas_Observes"].copy()
    
    if X.isna().any().any():
        X = X.fillna(0)
    
    if y.isna().any():
        y = y.fillna(0)
    
    if mode_importance == "👨‍⚕️ Manuel (Expert)":
        st.markdown('<div class="weight-box">', unsafe_allow_html=True)
        st.markdown("**⚖️ Application des poids manuels aux variables**")
        
        column_weights = {}
        
        for group_name, weight in poids_normalises.items():
            if group_name in feature_groups:
                cols_in_group = feature_groups[group_name]
                if len(cols_in_group) > 0:
                    weight_per_col = weight / len(cols_in_group)
                    for col in cols_in_group:
                        if col in feature_cols:
                            column_weights[col] = weight_per_col
        
        for col in feature_cols:
            if col not in column_weights:
                column_weights[col] = 0.01
        
        X_weighted = X.copy()
        for col in feature_cols:
            if col in column_weights:
                X_weighted[col] = X_weighted[col] * column_weights[col]
        
        weights_df = pd.DataFrame({
            "Variable": list(column_weights.keys()),
            "Poids": [f"{v*100:.2f}%" for v in column_weights.values()]
        })
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(weights_df, use_container_width=True, hide_index=True)
        with col2:
            st.metric("Total des poids", "100.00%")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_weighted)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    
    if np.isnan(X_scaled).any():
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    if modele_choisi == "GradientBoosting (Recommandé)":
        model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=4,
            random_state=42
        )
    elif modele_choisi == "RandomForest":
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=4,
            random_state=42
        )
    elif modele_choisi == "Ridge Regression":
        model = Ridge(alpha=1.0, random_state=42)
    elif modele_choisi == "Lasso Regression":
        model = Lasso(alpha=0.1, random_state=42)
    elif modele_choisi == "Decision Tree":
        model = DecisionTreeRegressor(
            max_depth=8,
            min_samples_split=4,
            random_state=42
        )
    
    model.fit(X_train, y_train)
    
    score_test = model.score(X_test, y_test)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 R² Test", f"{score_test:.3f}")
    with col2:
        st.metric("🎯 R² CV (5-fold)", f"{cv_mean:.3f}")
    with col3:
        st.metric("📏 Écart-type CV", f"±{cv_std:.3f}")
    
    if cv_mean > 0.7:
        st.success(f"✅ Modèle performant ({modele_choisi})")
    elif cv_mean > 0.5:
        st.warning(f"⚠️ Modèle acceptable ({modele_choisi})")
    else:
        st.error(f"❌ Modèle peu performant - envisagez un autre algorithme")
    
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            "Variable": feature_cols,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
        
        with st.expander("📊 Importance des variables", expanded=True):
            if mode_importance == "👨‍⚕️ Manuel (Expert)":
                st.info("ℹ️ Ces importances reflètent l'influence des variables **après application des poids manuels**")
            else:
                st.info("ℹ️ Ces importances sont **calculées automatiquement** par le modèle ML")
            
            fig_imp = px.bar(
                feature_importance.head(10),
                x="Importance",
                y="Variable",
                orientation="h",
                title="Top 10 variables les plus importantes",
                color="Importance",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_imp, use_container_width=True)
    
    # ============================================================
# GÉNÉRATION DES PRÉDICTIONS FUTURES
# ============================================================

st.subheader(f"📅 Génération des Prédictions - {n_weeks_pred} Semaines")

# Calcul de la dernière semaine réelle dans les données
derniere_semaine_reelle_epi = df['Semaine_Epi'].max()
derniere_annee_reelle = df['Annee'].max()

try:
    derniere_date_reelle = datetime.strptime(
        f"{derniere_annee_reelle}-W{derniere_semaine_reelle_epi:02d}-1", 
        "%Y-W%W-%w"
    )
except:
    derniere_date_reelle = end_date

st.info(f"📆 Dernière semaine de données : S{derniere_semaine_reelle_epi:02d} ({derniere_annee_reelle})")
st.info(f"🔮 Prédiction : S{derniere_semaine_reelle_epi+1:02d} à S{(derniere_semaine_reelle_epi+n_weeks_pred-1) % 52 + 1:02d}")

future_climate = None
if donnees_dispo["Climat"]:
    future_start = end_date + timedelta(days=1)
    future_end = end_date + timedelta(days=n_weeks_pred * 7)
    
    with st.spinner("🌡️ Chargement prévisions climatiques..."):
        try:
            future_climate = fetch_climate_nasa_power(sa_gdf, future_start, future_end)
            st.info("✅ Prévisions climatiques intégrées aux prédictions")
        except:
            st.warning("⚠️ Prévisions climatiques indisponibles - utilisation valeurs moyennes")

future_predictions = []

for aire in weekly_features["Aire_Sante"].unique():
    aire_data = weekly_features[weekly_features["Aire_Sante"] == aire].sort_values(['Annee', 'Semaine_Epi'])
    
    if aire_data.empty:
        continue
    
    last_obs = aire_data.iloc[-1]
    
    last_4_weeks = aire_data.tail(4)['Cas_Observes'].values
    if len(last_4_weeks) < 4:
        last_4_weeks = np.pad(last_4_weeks, (4-len(last_4_weeks), 0), 'edge')
    
    for i in range(1, n_weeks_pred + 1):
        nouvelle_semaine_epi = (derniere_semaine_reelle_epi + i - 1) % 52 + 1
        nouvelle_annee = derniere_annee_reelle + ((derniere_semaine_reelle_epi + i - 1) // 52)
        
        future_row = {
            "Aire_Sante": aire,
            "Annee": nouvelle_annee,
            "Semaine_Epi": nouvelle_semaine_epi,
            "Semaine_Label": f"{nouvelle_annee}-S{str(nouvelle_semaine_epi).zfill(2)}",
            "Age_Moyen": last_obs["Age_Moyen"] if pd.notna(last_obs["Age_Moyen"]) else 0
        }
        
        if donnees_dispo["Population"]:
            future_row.update({
                "Pop_Totale": last_obs["Pop_Totale"] if pd.notna(last_obs["Pop_Totale"]) else 0,
                "Pop_Enfants": last_obs["Pop_Enfants"] if pd.notna(last_obs["Pop_Enfants"]) else 0,
                "Densite_Pop": last_obs["Densite_Pop"] if pd.notna(last_obs["Densite_Pop"]) else 0,
                "Densite_Enfants": last_obs["Densite_Enfants"] if pd.notna(last_obs["Densite_Enfants"]) else 0
            })
        
        if donnees_dispo["Urbanisation"]:
            future_row["Urban_Encoded"] = last_obs["Urban_Encoded"] if pd.notna(last_obs["Urban_Encoded"]) else 0
        
        if donnees_dispo["Climat"]:
            if future_climate is not None:
                climate_aire = future_climate[future_climate["health_area"] == aire]
                if not climate_aire.empty:
                    temp_val = climate_aire.iloc[0].get("Temperature_Moy", 0)
                    hum_val = climate_aire.iloc[0].get("Humidite_Moy", 0)
                    saison_val = climate_aire.iloc[0].get("Saison_Seche_Humidite", 0)
                    
                    temp_val = 0 if pd.isna(temp_val) else temp_val
                    hum_val = 0 if pd.isna(hum_val) else hum_val
                    saison_val = 0 if pd.isna(saison_val) else saison_val
                    
                    temp_norm = scaler_climat.transform([[temp_val, 0, 0]])[0][0]
                    hum_norm = scaler_climat.transform([[0, hum_val, 0]])[0][1]
                    saison_norm = scaler_climat.transform([[0, 0, saison_val]])[0][2]
                    
                    future_row["Coef_Climatique"] = temp_norm * 0.4 + hum_norm * 0.4 + saison_norm * 0.2
                else:
                    future_row["Coef_Climatique"] = last_obs.get("Coef_Climatique", 0)
            else:
                future_row["Coef_Climatique"] = last_obs.get("Coef_Climatique", 0)
        
        if donnees_dispo["Vaccination"]:
            future_row["Taux_Vaccination"] = last_obs["Taux_Vaccination"] if pd.notna(last_obs["Taux_Vaccination"]) else 80
            future_row["Non_Vaccines"] = last_obs["Non_Vaccines"] if pd.notna(last_obs["Non_Vaccines"]) else 20
        elif "Non_Vaccines" in last_obs:
            future_row["Non_Vaccines"] = last_obs["Non_Vaccines"] if pd.notna(last_obs["Non_Vaccines"]) else 20
        
        if i == 1:
            future_row["Cas_Observes"] = last_obs["Cas_Observes"]
            future_row["Cas_Lag_1"] = last_4_weeks[-1]
            future_row["Cas_Lag_2"] = last_4_weeks[-2] if len(last_4_weeks) >= 2 else last_4_weeks[-1]
            future_row["Cas_Lag_3"] = last_4_weeks[-3] if len(last_4_weeks) >= 3 else last_4_weeks[-1]
            future_row["Cas_Lag_4"] = last_4_weeks[-4] if len(last_4_weeks) >= 4 else last_4_weeks[-1]
        else:
            prev_predictions_aire = [
                p["Predicted_Cases"] for p in future_predictions
                if p["Aire_Sante"] == aire
            ]
            
            if len(prev_predictions_aire) > 0:
                future_row["Cas_Observes"] = prev_predictions_aire[-1]
                future_row["Cas_Lag_1"] = prev_predictions_aire[-1] if len(prev_predictions_aire) >= 1 else last_4_weeks[-1]
                future_row["Cas_Lag_2"] = prev_predictions_aire[-2] if len(prev_predictions_aire) >= 2 else last_4_weeks[-2]
                future_row["Cas_Lag_3"] = prev_predictions_aire[-3] if len(prev_predictions_aire) >= 3 else last_4_weeks[-3]
                future_row["Cas_Lag_4"] = prev_predictions_aire[-4] if len(prev_predictions_aire) >= 4 else last_4_weeks[-4]
            else:
                future_row["Cas_Observes"] = last_obs["Cas_Observes"]
                future_row["Cas_Lag_1"] = last_4_weeks[-1]
                future_row["Cas_Lag_2"] = last_4_weeks[-2] if len(last_4_weeks) >= 2 else last_4_weeks[-1]
                future_row["Cas_Lag_3"] = last_4_weeks[-3] if len(last_4_weeks) >= 3 else last_4_weeks[-1]
                future_row["Cas_Lag_4"] = last_4_weeks[-4] if len(last_4_weeks) >= 4 else last_4_weeks[-1]
        
        X_future_values = []
        for col in feature_cols:
            val = future_row.get(col, 0)
            if pd.isna(val):
                val = 0
            X_future_values.append(val)
        
        X_future = np.array([X_future_values])
        
        if mode_importance == "👨‍⚕️ Manuel (Expert)":
            for idx, col in enumerate(feature_cols):
                if col in column_weights:
                    X_future[0, idx] = X_future[0, idx] * column_weights[col]
        
        if np.isnan(X_future).any():
            X_future = np.nan_to_num(X_future, nan=0.0)
        
        X_future_scaled = scaler.transform(X_future)
        
        if np.isnan(X_future_scaled).any():
            X_future_scaled = np.nan_to_num(X_future_scaled, nan=0.0)
        
        predicted_cases = max(0, model.predict(X_future_scaled)[0])
        
        if cv_std > 0:
            noise_seed = hash(aire) % 1000
            np.random.seed(noise_seed + i)
            noise = np.random.normal(0, predicted_cases * cv_std * 0.15)
            predicted_cases = max(0, predicted_cases + noise)
        
        future_row["Predicted_Cases"] = predicted_cases
        future_predictions.append(future_row)

future_df = pd.DataFrame(future_predictions)
future_df['Predicted_Cases'] = future_df['Predicted_Cases'].round(0).astype(int)

st.success(f"✓ {len(future_df)} prédictions générées ({len(future_df['Aire_Sante'].unique())} aires × {n_weeks_pred} semaines)")

# ============================================================
# ANALYSE DES PRÉDICTIONS
# ============================================================

moyenne_historique = weekly_features.groupby("Aire_Sante")["Cas_Observes"].mean().reset_index()
moyenne_historique.columns = ["Aire_Sante", "Moyenne_Historique"]

risk_df = future_df.groupby("Aire_Sante").agg(
    Cas_Predits_Total=("Predicted_Cases", "sum"),
    Cas_Predits_Max=("Predicted_Cases", "max"),
    Cas_Predits_Moyen=("Predicted_Cases", "mean"),
    Semaine_Pic=("Predicted_Cases", lambda x: future_df.loc[x.idxmax(), "Semaine_Label"] if len(x) > 0 else "N/A")
).reset_index()

risk_df['Cas_Predits_Total'] = risk_df['Cas_Predits_Total'].round(0).astype(int)
risk_df['Cas_Predits_Max'] = risk_df['Cas_Predits_Max'].round(0).astype(int)
risk_df['Cas_Predits_Moyen'] = risk_df['Cas_Predits_Moyen'].round(1)

risk_df = risk_df.merge(moyenne_historique, on="Aire_Sante", how="left")

risk_df["Variation_Pct"] = (
    (risk_df["Cas_Predits_Moyen"] - risk_df["Moyenne_Historique"]) /
    risk_df["Moyenne_Historique"].replace(0, 1)
) * 100

risk_df["Categorie_Variation"] = pd.cut(
    risk_df["Variation_Pct"],
    bins=[-np.inf, -seuil_baisse, -10, 10, seuil_hausse, np.inf],
    labels=["Forte baisse", "Baisse modérée", "Stable", "Hausse modérée", "Forte hausse"]
)

risk_df = risk_df.sort_values("Variation_Pct", ascending=False)

st.subheader("📊 Tableau de Synthèse des Prédictions")

st.dataframe(
    risk_df.style.format({
        'Cas_Predits_Total': '{:.0f}',
        'Cas_Predits_Max': '{:.0f}',
        'Cas_Predits_Moyen': '{:.1f}',
        'Moyenne_Historique': '{:.1f}',
        'Variation_Pct': '{:.1f}%'
    }),
    use_container_width=True
)

st.subheader("📈 Visualisations")

top_risk = risk_df.head(10)

fig_top = px.bar(
    top_risk,
    x='Cas_Predits_Total',
    y='Aire_Sante',
    orientation='h',
    title='Top 10 Aires à Risque (Cas prédits totaux)',
    labels={'Cas_Predits_Total': 'Cas prédits', 'Aire_Sante': 'Aire de santé'},
    color='Variation_Pct',
    color_continuous_scale='RdYlGn_r'
)

st.plotly_chart(fig_top, use_container_width=True)

# ============================================================
# APRÈS LES VISUALISATIONS (APRÈS LA HEATMAP)
# ============================================================

st.subheader("🗓️ Heatmap Hebdomadaire des Prédictions")

heatmap_data = future_df.pivot_table(
    values='Predicted_Cases',
    index='Aire_Sante',
    columns='Semaine_Label',
    aggfunc='sum',
    fill_value=0
)

heatmap_data = heatmap_data.round(0).astype(int)

# ============================================================
# CORRECTION : Heatmap style damier avec cellules lisibles
# ============================================================

fig_heatmap = go.Figure(data=go.Heatmap(
    z=heatmap_data.values,
    x=heatmap_data.columns,
    y=heatmap_data.index,
    colorscale='Reds',
    showscale=True,
    colorbar=dict(
        title=dict(
            text="Cas<br>prédits",
            side="right"
        ),
        tickmode="linear",
        tick0=0,
        dtick=max(1, heatmap_data.values.max() // 10)
    ),
    hovertemplate='<b>%{y}</b><br>Semaine: %{x}<br>Cas prédits: %{z}<extra></extra>',
    text=heatmap_data.values,
    texttemplate='%{text}',
    textfont=dict(
        size=10,
        color='white'
    ),
    xgap=2,  # Espacement horizontal entre cellules
    ygap=2   # Espacement vertical entre cellules
))

fig_heatmap.update_layout(
    title=f"Prédictions par Aire et par Semaine ({n_weeks_pred} semaines)",
    xaxis=dict(
        title="Semaine épidémiologique",
        tickangle=-45,
        tickfont=dict(size=10),
        side='bottom',
        showgrid=False
    ),
    yaxis=dict(
        title="Aire de santé",
        tickfont=dict(size=9),
        showgrid=False
    ),
    height=max(400, len(heatmap_data) * 25),  # Hauteur adaptative
    width=None,
    plot_bgcolor='#f0f0f0',
    paper_bgcolor='white',
    margin=dict(l=150, r=50, t=80, b=100)
)

st.plotly_chart(fig_heatmap, use_container_width=True)

# Ajouter des statistiques sous la heatmap
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_cas = heatmap_data.values.sum()
    st.metric("Total cas prédits", f"{int(total_cas):,}")

with col2:
    semaine_max = heatmap_data.sum(axis=0).idxmax()
    cas_semaine_max = int(heatmap_data.sum(axis=0).max())
    st.metric("Semaine pic", semaine_max, f"{cas_semaine_max} cas")

with col3:
    aire_max = heatmap_data.sum(axis=1).idxmax()
    cas_aire_max = int(heatmap_data.sum(axis=1).max())
    st.metric("Aire la plus touchée", aire_max, f"{cas_aire_max} cas", delta_color="inverse")

with col4:
    moyenne_hebdo = heatmap_data.values.mean()
    st.metric("Moyenne par cellule", f"{moyenne_hebdo:.1f}")



# ============================================================
# CARTES INTERACTIVES DES PRÉDICTIONS
# ============================================================

st.subheader("🗺️ Cartographie des Prédictions")

# Vérifier la correspondance des aires
aires_gdf = set(sa_gdf_enrichi['health_area'].unique())
aires_risk = set(risk_df['Aire_Sante'].unique())
aires_communes = aires_gdf.intersection(aires_risk)

if len(aires_communes) == 0:
    st.error("❌ Aucune correspondance entre les aires géographiques et les prédictions")
    
    with st.expander("🔍 Diagnostic des aires"):
        st.write("**Aires dans le shapefile :**")
        st.write(list(aires_gdf)[:10])
        st.write("**Aires dans les prédictions :**")
        st.write(list(aires_risk)[:10])
        
        # Tentative de fuzzy matching
        from difflib import get_close_matches
        st.write("**Correspondances possibles (fuzzy) :**")
        for aire_risk in list(aires_risk)[:5]:
            matches = get_close_matches(aire_risk, list(aires_gdf), n=3, cutoff=0.6)
            st.write(f"- `{aire_risk}` → {matches}")
    
    st.stop()
else:
    st.info(f"✅ {len(aires_communes)} aires correspondent entre géographie et prédictions")

# Fusionner les prédictions avec la géométrie
gdf_predictions = sa_gdf_enrichi.merge(
    risk_df[['Aire_Sante', 'Cas_Predits_Total', 'Cas_Predits_Max', 'Variation_Pct', 'Categorie_Variation', 'Semaine_Pic']],
    left_on='health_area',
    right_on='Aire_Sante',
    how='left'
)

# Compter les aires sans prédictions
aires_sans_pred = gdf_predictions[gdf_predictions['Cas_Predits_Total'].isna()]
if len(aires_sans_pred) > 0:
    st.warning(f"⚠️ {len(aires_sans_pred)} aires sans prédictions (pas de données historiques)")

# ============================================================
# CORRECTION : Convertir Categorie_Variation en string AVANT fillna
# ============================================================

# Convertir la colonne catégorielle en string
gdf_predictions['Categorie_Variation'] = gdf_predictions['Categorie_Variation'].astype(str)

# Remplir les valeurs manquantes
gdf_predictions['Cas_Predits_Total'] = gdf_predictions['Cas_Predits_Total'].fillna(0).astype(int)
gdf_predictions['Cas_Predits_Max'] = gdf_predictions['Cas_Predits_Max'].fillna(0).astype(int)
gdf_predictions['Variation_Pct'] = gdf_predictions['Variation_Pct'].fillna(0)
gdf_predictions['Categorie_Variation'] = gdf_predictions['Categorie_Variation'].replace('nan', 'Aucune donnée')  # Remplacer 'nan' par le texte voulu
gdf_predictions['Semaine_Pic'] = gdf_predictions['Semaine_Pic'].fillna('N/A')

# Vérifier qu'il y a au moins une prédiction valide
if gdf_predictions['Cas_Predits_Total'].sum() == 0:
    st.error("❌ Aucune prédiction valide générée")
    st.info("Vérifiez que les noms d'aires dans votre CSV correspondent exactement aux noms dans le shapefile")
    st.stop()

# Créer la carte
center_lat = gdf_predictions.geometry.centroid.y.mean()
center_lon = gdf_predictions.geometry.centroid.x.mean()

m_predictions = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=6,
    tiles='CartoDB positron'
)

# Carte 1 : Cas prédits totaux (UNIQUEMENT aires avec prédictions)
gdf_avec_predictions = gdf_predictions[gdf_predictions['Cas_Predits_Total'] > 0]

if len(gdf_avec_predictions) > 0:
    folium.Choropleth(
        geo_data=gdf_avec_predictions,
        data=gdf_avec_predictions,
        columns=['health_area', 'Cas_Predits_Total'],
        key_on='feature.properties.health_area',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f'Cas prédits totaux ({n_weeks_pred} semaines)',
        name='Cas prédits totaux'
    ).add_to(m_predictions)
else:
    st.error("❌ Aucune aire avec des prédictions > 0")

# Carte 2 : Variation par rapport à la moyenne historique
folium.Choropleth(
    geo_data=gdf_predictions,
    data=gdf_predictions,
    columns=['health_area', 'Variation_Pct'],
    key_on='feature.properties.health_area',
    fill_color='RdYlGn_r',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Variation (%) vs moyenne historique',
    name='Variation (%)',
    show=False
).add_to(m_predictions)

# Ajouter des markers avec popups détaillés
for idx, row in gdf_predictions.iterrows():
    
    # IGNORER les aires sans prédictions
    if row['Cas_Predits_Total'] == 0:
        continue
    
    # Couleur selon catégorie
    categorie = row['Categorie_Variation']
    
    if categorie == 'Forte hausse':
        color = 'red'
        icon = 'arrow-up'
    elif categorie == 'Hausse modérée':
        color = 'orange'
        icon = 'arrow-up'
    elif categorie == 'Stable':
        color = 'blue'
        icon = 'minus'
    elif categorie == 'Baisse modérée':
        color = 'lightgreen'
        icon = 'arrow-down'
    elif categorie == 'Forte baisse':
        color = 'green'
        icon = 'arrow-down'
    else:  # Aucune donnée ou autre
        color = 'gray'
        icon = 'question'
    
    # HTML du popup
    popup_html = f"""
    <div style="width:350px; font-family:Arial; font-size:13px;">
        <h4 style="color:#E4032E; margin:0; padding-bottom:8px; border-bottom:2px solid #E4032E;">
            {row['health_area']}
        </h4>
        <table style="width:100%; margin-top:10px; border-collapse:collapse;">
            <tr style="background-color:#f9f9f9;">
                <td style="padding:6px; font-weight:bold;">🔮 Cas prédits (total)</td>
                <td style="padding:6px; text-align:right;"><b>{row['Cas_Predits_Total']}</b></td>
            </tr>
            <tr>
                <td style="padding:6px; font-weight:bold;">📈 Cas max (semaine)</td>
                <td style="padding:6px; text-align:right;">{row['Cas_Predits_Max']}</td>
            </tr>
            <tr style="background-color:#f9f9f9;">
                <td style="padding:6px; font-weight:bold;">📅 Semaine pic</td>
                <td style="padding:6px; text-align:right;">{row['Semaine_Pic']}</td>
            </tr>
            <tr>
                <td style="padding:6px; font-weight:bold;">📊 Variation</td>
                <td style="padding:6px; text-align:right; color:{'red' if row['Variation_Pct'] > 0 else 'green'};">
                    {row['Variation_Pct']:.1f}%
                </td>
            </tr>
            <tr style="background-color:#f0f0f0;">
                <td colspan="2" style="padding:6px; text-align:center; font-weight:bold;">
                    {categorie}
                </td>
            </tr>
        </table>
    </div>
    """
    
    # Taille du marker proportionnelle aux cas prédits
    radius = min(5 + row['Cas_Predits_Total'] / 10, 25)
    
    folium.CircleMarker(
        location=[row.geometry.centroid.y, row.geometry.centroid.x],
        radius=radius,
        popup=folium.Popup(popup_html, max_width=400),
        color=color,
        fill=True,
        fillColor=color,
        fillOpacity=0.7,
        weight=2
    ).add_to(m_predictions)

# Ajouter contrôle de couches
folium.LayerControl().add_to(m_predictions)

# Afficher la carte
st_folium(m_predictions, width=1200, height=600, key='carte_predictions_rougeole')

# Légende des catégories
st.markdown(f"""
<div style="background:#f0f2f6; padding:1rem; border-radius:8px; margin-top:1rem;">
    <b>🎨 Légende des catégories :</b><br>
    🔴 <b>Forte hausse</b> : Variation ≥{seuil_hausse}% (Action urgente requise)<br>
    🟠 <b>Hausse modérée</b> : Variation entre 10% et {seuil_hausse}%<br>
    🔵 <b>Stable</b> : Variation entre -10% et +10%<br>
    🟢 <b>Baisse modérée</b> : Variation entre -{seuil_baisse}% et -10%<br>
    🟢 <b>Forte baisse</b> : Variation ≤-{seuil_baisse}% (Amélioration significative)
</div>
""", unsafe_allow_html=True)

# Carte des clusters à risque
st.subheader("🎯 Carte des Zones à Risque Élevé")

# Filtrer les aires en forte hausse
aires_critiques = gdf_predictions[gdf_predictions['Categorie_Variation'] == 'Forte hausse']

if len(aires_critiques) > 0:
    
    m_risque = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='CartoDB positron'
    )
    
    # Ajouter toutes les aires en gris clair
    folium.GeoJson(
        gdf_predictions,
        style_function=lambda x: {
            'fillColor': '#e0e0e0',
            'color': '#999999',
            'weight': 1,
            'fillOpacity': 0.3
        },
        name='Toutes les aires'
    ).add_to(m_risque)
    
    # Mettre en évidence les aires critiques
    for idx, row in aires_critiques.iterrows():
        
        # Style rouge pour zones critiques
        folium.GeoJson(
            row.geometry,
            style_function=lambda x: {
                'fillColor': '#ff0000',
                'color': '#8B0000',
                'weight': 3,
                'fillOpacity': 0.6
            }
        ).add_to(m_risque)
        
        # Marker avec alerte
        folium.Marker(
            location=[row.geometry.centroid.y, row.geometry.centroid.x],
            popup=folium.Popup(f"""
            <div style="width:250px; font-family:Arial;">
                <h4 style="color:red; margin:0;">⚠️ ALERTE</h4>
                <p style="margin:5px 0;"><b>{row['health_area']}</b></p>
                <p style="margin:5px 0;">Cas prédits : <b>{row['Cas_Predits_Total']}</b></p>
                <p style="margin:5px 0;">Hausse : <b style="color:red;">+{row['Variation_Pct']:.1f}%</b></p>
                <p style="margin:5px 0;">Pic : <b>{row['Semaine_Pic']}</b></p>
            </div>
            """, max_width=300),
            icon=folium.Icon(color='red', icon='exclamation-triangle', prefix='fa')
        ).add_to(m_risque)
    
    st_folium(m_risque, width=1200, height=600, key='carte_risque_rougeole')
    
    st.error(f"🚨 **{len(aires_critiques)} aires identifiées à risque élevé** - Intervention prioritaire recommandée")
    
else:
    st.success("✅ Aucune zone à risque élevé identifiée dans les prédictions")

# Carte de chaleur (heatmap géographique) si beaucoup de cas
if gdf_predictions['Cas_Predits_Total'].sum() > 100:
    
    st.subheader("🔥 Carte de Chaleur des Cas Prédits")
    
    # Préparer les données pour heatmap
    heat_data = []
    for idx, row in gdf_predictions.iterrows():
        if row['Cas_Predits_Total'] > 0:
            lat = row.geometry.centroid.y
            lon = row.geometry.centroid.x
            weight = row['Cas_Predits_Total']
            heat_data.append([lat, lon, weight])
    
    if len(heat_data) > 0:
        m_heat = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles='CartoDB positron'
        )
        
        # Ajouter heatmap
        from folium.plugins import HeatMap
        HeatMap(
            heat_data,
            min_opacity=0.3,
            max_opacity=0.8,
            radius=25,
            blur=20,
            gradient={
                0.0: 'blue',
                0.3: 'lime',
                0.5: 'yellow',
                0.7: 'orange',
                1.0: 'red'
            }
        ).add_to(m_heat)
        
        st_folium(m_heat, width=1200, height=600, key='heatmap_rougeole')
        
        st.info("💡 Les zones rouges/oranges indiquent les concentrations de cas prédits les plus élevées")

# ============================================================
# SUITE DU CODE : ALERTES ET TÉLÉCHARGEMENTS
# ============================================================

st.subheader("🚨 Alertes et Recommandations")

forte_hausse = risk_df[risk_df['Categorie_Variation'] == 'Forte hausse']

if len(forte_hausse) > 0:
    st.error(f"⚠️ **{len(forte_hausse)} aires en FORTE HAUSSE** (≥{seuil_hausse}%)")
    
    with st.expander("📋 Détails des aires critiques", expanded=True):
        st.dataframe(
            forte_hausse[['Aire_Sante', 'Cas_Predits_Total', 'Variation_Pct', 'Semaine_Pic']]
            .style.format({
                'Cas_Predits_Total': '{:.0f}',
                'Variation_Pct': '{:.1f}%'
            }),
            use_container_width=True
        )
        
        st.markdown("**🎯 Actions recommandées :**")
        st.markdown("- Intensifier la surveillance épidémiologique")
        st.markdown("- Préparer campagne de vaccination réactive (CVR)")
        st.markdown("- Renforcer stocks de vaccins et intrants")
        st.markdown("- Communication précoce aux équipes terrain")
else:
    st.success("✅ Aucune aire en forte hausse détectée")

forte_baisse = risk_df[risk_df['Categorie_Variation'] == 'Forte baisse']

if len(forte_baisse) > 0:
    st.success(f"✅ **{len(forte_baisse)} aires en FORTE BAISSE** (≥{seuil_baisse}%)")
    
    with st.expander("📋 Aires en amélioration"):
        st.dataframe(
            forte_baisse[['Aire_Sante', 'Cas_Predits_Total', 'Variation_Pct']]
            .style.format({
                'Cas_Predits_Total': '{:.0f}',
                'Variation_Pct': '{:.1f}%'
            }),
            use_container_width=True
        )

st.subheader("💾 Téléchargements")

col1, col2, col3 = st.columns(3)

with col1:
    csv_predictions = future_df.to_csv(index=False)
    st.download_button(
        label="📥 Prédictions détaillées (CSV)",
        data=csv_predictions,
        file_name=f"predictions_rougeole_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True,
        key="dl_pred_csv"
    )

with col2:
    csv_synthese = risk_df.to_csv(index=False)
    st.download_button(
        label="📊 Synthèse par aire (CSV)",
        data=csv_synthese,
        file_name=f"synthese_risque_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True,
        key="dl_synth_csv"
    )

with col3:
    from io import BytesIO
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        risk_df.to_excel(writer, sheet_name='Synthèse', index=False)
        future_df.to_excel(writer, sheet_name='Prédictions détaillées', index=False)
        heatmap_data.to_excel(writer, sheet_name='Heatmap')
    
    st.download_button(
        label="📊 Rapport complet (Excel)",
        data=output.getvalue(),
        file_name=f"rapport_predictions_rougeole_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key="dl_rapport_excel"
    )

# Export GeoJSON des prédictions
col4, col5, col6 = st.columns(3)

with col4:
    geojson_predictions = gdf_predictions.to_json()
    st.download_button(
        label="🗺️ Carte prédictions (GeoJSON)",
        data=geojson_predictions,
        file_name=f"carte_predictions_rougeole_{datetime.now().strftime('%Y%m%d')}.geojson",
        mime="application/json",
        use_container_width=True,
        key="dl_geojson_pred"
    )

with col5:
    if len(aires_critiques) > 0:
        geojson_risque = aires_critiques.to_json()
        st.download_button(
            label="⚠️ Zones à risque (GeoJSON)",
            data=geojson_risque,
            file_name=f"zones_risque_rougeole_{datetime.now().strftime('%Y%m%d')}.geojson",
            mime="application/json",
            use_container_width=True,
            key="dl_geojson_risque"
        )

st.markdown("---")
st.success("✅ Modélisation terminée avec succès !")
st.info("💡 Ajustez les paramètres dans la sidebar pour relancer une nouvelle prédiction")

st.stop()
