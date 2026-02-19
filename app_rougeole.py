# ============================================================
# APP SURVEILLANCE & PRÉDICTION ROUGEOLE - VERSION CORRIGÉE
# PARTIE 1/5 : IMPORTS, CONFIGURATION, SIDEBAR
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

# ============================================================
# SIDEBAR CONFIGURATION
# ============================================================

st.sidebar.header("📂 Configuration de l'Analyse")

# Mode démo
st.sidebar.subheader("🎯 Mode d'utilisation")
mode_demo = st.sidebar.radio(
    "Choisissez votre mode",
    ["📊 Données réelles", "🧪 Mode démo (données simulées)"],
    help="Mode démo : génère automatiquement des données fictives"
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
        help="Format : Shapefile ou GeoJSON"
    )

# Données épidémiologiques
st.sidebar.subheader("📊 Données Épidémiologiques")

if mode_demo == "🧪 Mode démo (données simulées)":
    linelist_file = None
    vaccination_file = None
    st.sidebar.info("📊 Mode démo activé")
else:
    linelist_file = st.sidebar.file_uploader(
        "📋 Linelists rougeole (CSV)",
        type=["csv"],
        help="Format : health_area, Semaine_Epi, Cas_Total"
    )

    vaccination_file = st.sidebar.file_uploader(
        "💉 Couverture vaccinale (CSV - optionnel)",
        type=["csv"],
        help="Format : health_area, Taux_Vaccination"
    )

# Période d'analyse
st.sidebar.subheader("📅 Période d'Analyse")

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
    help="Nombre de mois à prédire"
)
n_weeks_pred = pred_mois * 4

st.sidebar.info(f"📆 Prédiction sur **{n_weeks_pred} semaines**")

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
    ]
)

# Mode importance
st.sidebar.subheader("⚖️ Importance des Variables")

mode_importance = st.sidebar.radio(
    "Mode de pondération",
    ["🤖 Automatique (ML)", "👨‍⚕️ Manuel (Expert)"]
)

poids_manuels = {}
poids_normalises = {}

if mode_importance == "👨‍⚕️ Manuel (Expert)":
    with st.sidebar.expander("⚙️ Configurer les poids", expanded=True):
        poids_manuels["Historique_Cas"] = st.slider("📈 Historique des cas", 0, 100, 40, 5)
        poids_manuels["Vaccination"] = st.slider("💉 Vaccination", 0, 100, 35, 5)
        poids_manuels["Demographie"] = st.slider("👥 Démographie", 0, 100, 15, 5)
        poids_manuels["Urbanisation"] = st.slider("🏙️ Urbanisation", 0, 100, 8, 2)
        poids_manuels["Climat"] = st.slider("🌡️ Climat", 0, 100, 2, 1)

        total_poids = sum(poids_manuels.values())
        if total_poids > 0:
            for key in poids_manuels:
                poids_normalises[key] = poids_manuels[key] / total_poids

# Seuils d'alerte
st.sidebar.subheader("⚙️ Seuils d'Alerte")
with st.sidebar.expander("Configurer les seuils", expanded=False):
    seuil_baisse = st.slider("Seuil de baisse (%)", 10, 90, 75, 5)
    seuil_hausse = st.slider("Seuil de hausse (%)", 10, 200, 50, 10)
    seuil_alerte_epidemique = st.number_input("Seuil alerte (cas/semaine)", 1, 100, 5)
# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def convertir_semaine_en_date(annee, semaine):
    """Convertit une semaine épidémiologique en date"""
    try:
        return datetime.strptime(f"{annee}-W{semaine:02d}-1", "%Y-W%W-%w")
    except:
        return datetime(annee, 1, 1) + timedelta(weeks=semaine-1)

# Convertir les semaines sélectionnées en dates
date_debut_periode = convertir_semaine_en_date(annee_debut, semaine_debut)
date_fin_periode = convertir_semaine_en_date(annee_fin, semaine_fin)

# ============================================================
# FONCTIONS DE CHARGEMENT GÉOGRAPHIQUE
# ============================================================

@st.cache_data
def load_health_areas_from_zip(zip_path, iso3_filter):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(tmpdir)

            shp_files = [f for f in os.listdir(tmpdir) if f.endswith('.shp')]
            if not shp_files:
                raise ValueError("Aucun fichier .shp trouvé")

            gdf_full = gpd.read_file(os.path.join(tmpdir, shp_files[0]))

            iso3_col = None
            for col in ['iso3', 'ISO3', 'iso_code', 'ISO_CODE']:
                if col in gdf_full.columns:
                    iso3_col = col
                    break

            if iso3_col is None:
                st.warning(f"⚠️ Colonne ISO3 non trouvée")
                return gpd.GeoDataFrame()

            gdf = gdf_full[gdf_full[iso3_col] == iso3_filter].copy()

            if gdf.empty:
                st.warning(f"⚠️ Aucune aire pour {iso3_filter}")
                return gpd.GeoDataFrame()

            name_col = None
            for col in ['health_area', 'HEALTH_AREA', 'name_fr', 'name', 'NAME', 'nom', 'NOM']:
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
        st.error(f"❌ Erreur : {e}")
        return gpd.GeoDataFrame()

@st.cache_data
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
        st.error(f"❌ Erreur : {e}")
        return gpd.GeoDataFrame()

# ============================================================
# FONCTIONS DE GÉNÉRATION DE DONNÉES DEMO
# ============================================================

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
# FONCTIONS D'ENRICHISSEMENT GEE
# ============================================================

@st.cache_data(ttl=86400)
def worldpop_children_stats(_sa_gdf, gee_available):
    if not gee_available:
        return pd.DataFrame({
            "health_area": _sa_gdf["health_area"],
            "Pop_Totale": [np.nan] * len(_sa_gdf),
            "Pop_Enfants": [np.nan] * len(_sa_gdf)
        })

    try:
        dataset = ee.ImageCollection("WorldPop/GP/100m/pop_age_sex")
        pop_img = dataset.mosaic()

        male_bands = ['M_0', 'M_1', 'M_5', 'M_10']
        female_bands = ['F_0', 'F_1', 'F_5', 'F_10']

        selected_males = pop_img.select(male_bands)
        selected_females = pop_img.select(female_bands)

        males_sum = selected_males.reduce(ee.Reducer.sum()).rename('males_total')
        females_sum = selected_females.reduce(ee.Reducer.sum()).rename('females_total')
        enfants = males_sum.add(females_sum).rename('enfants')

        total_pop = pop_img.select('population')

        final_mosaic = total_pop.addBands(enfants).addBands(males_sum).addBands(females_sum)

        data_list = []

        for idx, row in _sa_gdf.iterrows():
            geom = ee.Geometry(json.loads(gpd.GeoSeries([row.geometry]).to_json())['features'][0]['geometry'])

            stats = final_mosaic.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=geom,
                scale=100,
                maxPixels=1e9
            ).getInfo()

            pop_totale = stats.get('population', 0)
            garcons = stats.get('males_total', 0)
            filles = stats.get('females_total', 0)
            enfants_total = stats.get('enfants', 0)

            data_list.append({
                "health_area": row['health_area'],
                "Pop_Totale": int(pop_totale) if pop_totale > 0 else np.nan,
                "Pop_Garcons": int(garcons),
                "Pop_Filles": int(filles),
                "Pop_Enfants": int(enfants_total)
            })

        return pd.DataFrame(data_list)

    except Exception as e:
        st.sidebar.error(f"WorldPop : {str(e)}")
        return pd.DataFrame({
            "health_area": _sa_gdf["health_area"],
            "Pop_Totale": [np.nan] * len(_sa_gdf),
            "Pop_Enfants": [np.nan] * len(_sa_gdf)
        })

@st.cache_data(ttl=86400)
def urban_classification(_sa_gdf, gee_available):
    if not gee_available:
        return pd.DataFrame({
            "health_area": _sa_gdf["health_area"],
            "Urbanisation": ["Rural"] * len(_sa_gdf)
        })

    try:
        ghsl = ee.Image("JRC/GHSL/P2023A/GHS_SMOD/2020")
        smod = ghsl.select('smod_code')

        urban_info = smod.reduceRegions(
            collection=ee.FeatureCollection(_sa_gdf.__geo_interface__),
            reducer=ee.Reducer.mode(),
            scale=1000
        )

        urban_info_list = urban_info.getInfo()

        data_list = []
        for i, feat in enumerate(urban_info_list['features']):
            props = feat['properties']
            mode_val = props.get('mode', 10)

            if mode_val >= 30:
                urbanisation = "Urbain"
            elif mode_val >= 20:
                urbanisation = "Périurbain"
            else:
                urbanisation = "Rural"

            data_list.append({
                "health_area": props.get('health_area', ''),
                "Urbanisation": urbanisation
            })

        return pd.DataFrame(data_list)

    except Exception as e:
        st.sidebar.error(f"GHSL : {str(e)}")
        return pd.DataFrame({
            "health_area": _sa_gdf["health_area"],
            "Urbanisation": ["Rural"] * len(_sa_gdf)
        })

@st.cache_data(ttl=86400)
def fetchclimate_nasa_power(_sa_gdf, start_date, end_date):
    data_list = []

    for idx, row in _sa_gdf.iterrows():
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

                saison_seche_hum = (rh_mean < 0.7) if not np.isnan(rh_mean) else np.nan

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

    return pd.DataFrame(data_list)

# ============================================================
# FONCTION DE NORMALISATION DES COLONNES
# ============================================================

COLONNES_MAPPING = {
    "Aire_Sante": ["Aire_Sante", "aire_sante", "health_area", "HEALTH_AREA", "name_fr", "NAME", "nom", "NOM"],
    "Date_Debut_Eruption": ["Date_Debut_Eruption", "date_debut_eruption", "Date_Debut", "date_onset", "symptom_onset"],
    "Date_Notification": ["Date_Notification", "date_notification", "Date_Notif", "notification_date"],
    "ID_Cas": ["ID_Cas", "id_cas", "ID", "id", "Case_ID", "case_id"],
    "Age_Mois": ["Age_Mois", "age_mois", "Age", "age", "AGE", "Age_Months"],
    "Statut_Vaccinal": ["Statut_Vaccinal", "statut_vaccinal", "Vaccin", "vaccin", "Vaccination_Status"],
    "Sexe": ["Sexe", "sexe", "Sex", "sex", "Gender", "gender"],
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
# ============================================================
# CHARGEMENT DES AIRES DE SANTÉ
# ============================================================

if st.session_state.sa_gdf_cache is not None and option_aire == "Fichier local (ao_hlthArea.zip)":
    sa_gdf = st.session_state.sa_gdf_cache
    st.sidebar.success(f"✓ {len(sa_gdf)} aires (cache)")
else:
    with st.spinner("🔄 Chargement aires..."):
        if option_aire == "Fichier local (ao_hlthArea.zip)":
            zip_path = os.path.join("data", "ao_hlthArea.zip")
            if not os.path.exists(zip_path):
                st.error(f"❌ Fichier non trouvé : {zip_path}")
                st.stop()

            sa_gdf = load_health_areas_from_zip(zip_path, iso3_pays)

            if sa_gdf.empty:
                st.error(f"❌ Impossible de charger {pays_selectionne}")
                st.stop()
            else:
                st.sidebar.success(f"✓ {len(sa_gdf)} aires")
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
                    st.sidebar.success(f"✓ {len(sa_gdf)} aires")
                    st.session_state.sa_gdf_cache = sa_gdf

if sa_gdf.empty or sa_gdf is None:
    st.error("❌ Aucune aire chargée")
    st.stop()

# ============================================================
# CHARGEMENT DES DONNÉES DE CAS
# ============================================================

with st.spinner('Chargement données de cas...'):
    if mode_demo == "🧪 Mode démo (données simulées)":
        df = generate_dummy_linelists(sa_gdf, start=date_debut_periode, end=date_fin_periode)
        vaccination_df = generate_dummy_vaccination(sa_gdf)
        st.sidebar.info(f"{len(df)} cas simulés")
    else:
        if linelist_file is None:
            st.error("Veuillez uploader un fichier CSV")
            st.stop()

        try:
            # Détection du séparateur
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

            st.sidebar.info(f"🔍 Séparateur : `{repr(separator)}`")

            # Lecture du fichier
            try:
                df_raw = pd.read_csv(linelist_file, sep=separator, encoding='utf-8')
            except UnicodeDecodeError:
                linelist_file.seek(0)
                df_raw = pd.read_csv(linelist_file, sep=separator, encoding='latin1')

            st.sidebar.success(f"✅ {len(df_raw)} lignes chargées")

            # Mapping des colonnes
            COLUMNS_MAPPING_EXTENDED = {
                'health_area': ['health_area', 'healtharea', 'HEALTH_AREA', 'aire_sante', 'Aire_Sante', 'district', 'zone', 'name_fr', 'NAME', 'nom'],
                'Semaine_Epi': ['Semaine_Epi', 'SemaineEpi', 'semaine_epi', 'semaine', 'Semaine', 'week', 'Week', 'epi_week', 'SE'],
                'Annee': ['Annee', 'Année', 'annee', 'année', 'year', 'Year', 'an'],
                'Cas_Total': ['Cas_Total', 'CasTotal', 'cas_total', 'cas', 'Cas', 'cases', 'Cases', 'nb_cas'],
                'Date_Debut_Eruption': ['Date_Debut_Eruption', 'date_debut_eruption', 'Date_Debut', 'date_onset', 'symptom_onset']
            }

            rename_dict = {}
            for standard_col, possible_cols in COLUMNS_MAPPING_EXTENDED.items():
                for col in possible_cols:
                    if col in df_raw.columns and col != standard_col:
                        rename_dict[col] = standard_col
                        break

            if rename_dict:
                df_raw = df_raw.rename(columns=rename_dict)
                st.sidebar.success(f"🔄 Colonnes renommées : {len(rename_dict)}")

            # FORMAT AGRÉGÉ → LINELIST
            if 'Semaine_Epi' in df_raw.columns and ('Cas_Total' in df_raw.columns or 'cas' in df_raw.columns):
                st.sidebar.info("📊 Format agrégé détecté")

                if 'Cas_Total' not in df_raw.columns:
                    for col in ['cas', 'Cas', 'cases']:
                        if col in df_raw.columns:
                            df_raw['Cas_Total'] = df_raw[col]
                            break

                expanded_rows = []
                lignes_ignorees = 0

                for _, row in df_raw.iterrows():
                    try:
                        aire = row.get('health_area') or row.get('Aire_Sante', 'Inconnu')

                        semaine_val = row.get('Semaine_Epi')
                        if pd.isna(semaine_val):
                            lignes_ignorees += 1
                            continue
                        semaine = int(semaine_val)

                        if semaine < 1 or semaine > 53:
                            lignes_ignorees += 1
                            continue

                        cas_total_val = row.get('Cas_Total')
                        if pd.isna(cas_total_val) or cas_total_val <= 0:
                            lignes_ignorees += 1
                            continue
                        cas_total = int(cas_total_val)

                        annee_val = row.get('Annee')
                        if pd.isna(annee_val):
                            annee = datetime.now().year
                        else:
                            annee = int(annee_val)

                        if annee < 2000 or annee > datetime.now().year + 1:
                            lignes_ignorees += 1
                            continue

                        try:
                            base_date = datetime.strptime(f"{annee}-W{semaine:02d}-1", "%Y-W%W-%w")
                        except:
                            try:
                                base_date = datetime(int(annee), 1, 1) + timedelta(weeks=semaine-1)
                            except:
                                lignes_ignorees += 1
                                continue

                        for i in range(cas_total):
                            jour_aleatoire = np.random.randint(0, 7)
                            date_cas = base_date + timedelta(days=jour_aleatoire)

                            expanded_rows.append({
                                'ID_Cas': len(expanded_rows) + 1,
                                'Date_Debut_Eruption': date_cas,
                                'Date_Notification': date_cas + timedelta(days=np.random.randint(0, 10)),
                                'Aire_Sante': aire,
                                'Annee': annee,
                                'Semaine_Epi': semaine,
                                'Age_Mois': np.random.randint(6, 180),
                                'Statut_Vaccinal': 'Inconnu',
                                'Sexe': 'Inconnu',
                                'Issue': 'Inconnu'
                            })
                    except (ValueError, TypeError):
                        lignes_ignorees += 1
                        continue

                if lignes_ignorees > 0:
                    st.sidebar.warning(f"⚠️ {lignes_ignorees} lignes ignorées")

                if len(expanded_rows) == 0:
                    st.error("❌ Aucune donnée valide")
                    st.stop()

                df = pd.DataFrame(expanded_rows)
                st.sidebar.success(f"✅ {len(df)} cas créés")

            elif 'Date_Debut_Eruption' in df_raw.columns:
                # FORMAT LINELIST STANDARD
                df = df_raw.copy()
                for col in ['Date_Debut_Eruption', 'Date_Notification']:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')

                if 'Semaine_Epi' not in df.columns:
                    df['Semaine_Epi'] = df['Date_Debut_Eruption'].dt.isocalendar().week
                if 'Annee' not in df.columns:
                    df['Annee'] = df['Date_Debut_Eruption'].dt.isocalendar().year

                st.sidebar.success(f"✅ {len(df)} cas chargés")
            else:
                st.error("❌ Format CSV non reconnu")
                st.info("Formats acceptés : Format agrégé (Semaine_Epi + Cas_Total) ou Linelist (Date_Debut_Eruption)")
                st.stop()

        except Exception as e:
            st.error(f"❌ Erreur CSV : {e}")
            st.stop()

        # Vaccination
        if vaccination_file is not None:
            try:
                vaccination_df = pd.read_csv(vaccination_file)
                st.sidebar.success(f"✓ Vaccination chargée")
            except:
                vaccination_df = None
        else:
            vaccination_df = None

# ============================================================
# NORMALISATION ET FILTRAGE
# ============================================================

# Normaliser les colonnes
df = normaliser_colonnes(df, COLONNES_MAPPING)

# Créer ID_Cas si absent
if "ID_Cas" not in df.columns:
    df["ID_Cas"] = range(1, len(df) + 1)

# Créer colonnes temporelles
if 'Date_Debut_Eruption' in df.columns:
    df['Date_Debut_Eruption'] = pd.to_datetime(df['Date_Debut_Eruption'], errors='coerce')

    if 'Annee' not in df.columns:
        df['Annee'] = df['Date_Debut_Eruption'].dt.isocalendar().year
    if 'Semaine_Epi' not in df.columns:
        df['Semaine_Epi'] = df['Date_Debut_Eruption'].dt.isocalendar().week
else:
    st.error("❌ Colonne 'Date_Debut_Eruption' manquante")
    st.stop()

# Filtrage par semaines épidémiologiques
if 'Annee' in df.columns and 'Semaine_Epi' in df.columns:
    annee_min_data = df['Annee'].min()
    annee_max_data = df['Annee'].max()
    semaine_min_data = df[df['Annee'] == annee_min_data]['Semaine_Epi'].min()
    semaine_max_data = df[df['Annee'] == annee_max_data]['Semaine_Epi'].max()

    st.info(f"📅 **Données disponibles :** S{semaine_min_data:02d}/{annee_min_data} → S{semaine_max_data:02d}/{annee_max_data}")

    df_before_filter = len(df)

    df['Periode_ID'] = df['Annee'] * 100 + df['Semaine_Epi']
    periode_debut_id = annee_debut * 100 + semaine_debut
    periode_fin_id = annee_fin * 100 + semaine_fin

    df = df[(df['Periode_ID'] >= periode_debut_id) & (df['Periode_ID'] <= periode_fin_id)]
    df = df.drop(columns=['Periode_ID'])

    df_after_filter = len(df)

    if df_after_filter == 0:
        st.error(f"❌ Aucune donnée pour S{semaine_debut:02d}/{annee_debut} → S{semaine_fin:02d}/{annee_fin}")
        st.stop()

    st.success(f"✅ **{df_after_filter:,} cas** sur la période ({df_before_filter - df_after_filter} exclus)")
else:
    st.error("❌ Colonnes 'Annee' et 'Semaine_Epi' manquantes")
    st.stop()

if len(df) == 0:
    st.error("❌ Aucune donnée disponible")
    st.stop()

st.sidebar.success(f"✅ {len(df)} cas analysés")

# Vérifier Aire_Sante
if "Aire_Sante" not in df.columns:
    df["Aire_Sante"] = sa_gdf["health_area"].iloc[0]
    st.sidebar.warning("⚠️ Aire_Sante assignée par défaut")

# ============================================================
# ENRICHISSEMENT DES DONNÉES
# ============================================================

with st.spinner('Enrichissement des données...'):
    pop_df = worldpop_children_stats(sa_gdf, gee_ok)
    urban_df = urban_classification(sa_gdf, gee_ok)
    climate_df = fetchclimate_nasa_power(sa_gdf, date_debut_periode, date_fin_periode)

    sa_gdf_enrichi = sa_gdf.copy()
    sa_gdf_enrichi = sa_gdf_enrichi.merge(pop_df, on="health_area", how="left")
    sa_gdf_enrichi = sa_gdf_enrichi.merge(urban_df, on="health_area", how="left")
    sa_gdf_enrichi = sa_gdf_enrichi.merge(climate_df, on="health_area", how="left")

    if vaccination_df is not None:
        sa_gdf_enrichi = sa_gdf_enrichi.merge(vaccination_df, on="health_area", how="left")
    else:
        sa_gdf_enrichi["Taux_Vaccination"] = np.nan

    # Calculer superficie et densité
    sa_gdf_m = sa_gdf_enrichi.to_crs("ESRI:54009")
    sa_gdf_enrichi["Superficie_km2"] = sa_gdf_m.geometry.area / 1e6
    sa_gdf_enrichi["Densite_Pop"] = sa_gdf_enrichi["Pop_Totale"] / sa_gdf_enrichi["Superficie_km2"].replace(0, np.nan)
    sa_gdf_enrichi["Densite_Enfants"] = sa_gdf_enrichi["Pop_Enfants"] / sa_gdf_enrichi["Superficie_km2"].replace(0, np.nan)

    sa_gdf_enrichi = sa_gdf_enrichi.replace([np.inf, -np.inf], np.nan)

    st.sidebar.success("✓ Enrichissement terminé")

# Données disponibles
donnees_dispo = {
    "Population": not sa_gdf_enrichi["Pop_Totale"].isna().all(),
    "Urbanisation": not sa_gdf_enrichi["Urbanisation"].isna().all(),
    "Climat": not sa_gdf_enrichi["Humidite_Moy"].isna().all(),
    "Vaccination": not sa_gdf_enrichi["Taux_Vaccination"].isna().all()
}

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Données disponibles")
for nom, dispo in donnees_dispo.items():
    icone = "✅" if dispo else "❌"
    st.sidebar.text(f"{icone} {nom}")
# ============================================================
# STRUCTURE EN 3 ONGLETS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "📊 Dashboard & Analyse",
    "🗺️ Cartographie",
    "🔮 Modélisation & Prédiction"
])

# ============================================================
# ONGLET 1 : DASHBOARD
# ============================================================

with tab1:
    st.header("📊 Indicateurs Clés de Performance")

    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Cas totaux", f"{len(df):,}")

    with col2:
        if 'Statut_Vaccinal' in df.columns and df['Statut_Vaccinal'].notna().sum() > 0:
            vacc_valides = df[df['Statut_Vaccinal'] != 'Inconnu']
            if len(vacc_valides) > 0:
                taux_non_vac = (vacc_valides['Statut_Vaccinal'] == 'Non').mean() * 100
                delta_vac = taux_non_vac - 45
                st.metric("Non vaccinés", f"{taux_non_vac:.1f}%", delta=f"{delta_vac:.1f}%")
            else:
                st.metric("Non vaccinés", "N/A")
        else:
            st.metric("Non vaccinés", "N/A")

    with col3:
        if 'Age_Mois' in df.columns and df['Age_Mois'].notna().sum() > 0:
            age_median = df['Age_Mois'].median()
            st.metric("Âge médian", f"{int(age_median)} mois")
        else:
            st.metric("Âge médian", "N/A")

    with col4:
        if 'Issue' in df.columns and df['Issue'].notna().sum() > 0:
            issue_valides = df[df['Issue'] != 'Inconnu']
            if len(issue_valides) > 0 and (issue_valides['Issue'] == 'Décédé').sum() > 0:
                taux_deces = (issue_valides['Issue'] == 'Décédé').mean() * 100
                st.metric("Létalité", f"{taux_deces:.2f}%")
            else:
                st.metric("Létalité", "N/A")
        else:
            st.metric("Létalité", "N/A")

    with col5:
        n_aires_touchees = df['Aire_Sante'].nunique()
        pct_aires = (n_aires_touchees / len(sa_gdf)) * 100
        st.metric("Aires touchées", f"{n_aires_touchees}/{len(sa_gdf)}", delta=f"{pct_aires:.0f}%")

    st.markdown("---")

    # ============================================================
    # TOP 10 AIRES DE SANTÉ
    # ============================================================

    st.subheader("🏆 Top 10 Aires de Santé")

    # Calculer statistiques par aire
    aggdict = {'ID_Cas': 'count'}
    if 'Age_Mois' in df.columns:
        aggdict['Age_Mois'] = 'mean'
    if 'Statut_Vaccinal' in df.columns:
        aggdict['Statut_Vaccinal'] = lambda x: ((x == 'Non').sum() / len(x) * 100) if len(x) > 0 else 0

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

    # Fusionner avec données géographiques
    cases_by_area = cases_by_area.merge(
        sa_gdf_enrichi[['health_area', 'Pop_Totale', 'Pop_Enfants']],
        left_on='Aire_Sante',
        right_on='health_area',
        how='left'
    )

    cases_by_area['Pop_Totale'] = cases_by_area['Pop_Totale'].fillna(1)
    cases_by_area['Pop_Enfants'] = cases_by_area['Pop_Enfants'].fillna(1)

    # Calculer taux d'attaque
    cases_by_area['Taux_Attaque_10K'] = (
        (cases_by_area['Cas_Observes'] / cases_by_area['Pop_Totale'].replace(0, np.nan)) * 10000
    ).fillna(0)

    # 2 sous-onglets
    tab_taux, tab_cas = st.tabs(["📈 Par Taux d\'Attaque", "🔢 Par Nombre de Cas"])

    with tab_taux:
        st.markdown("**Top 10 - Taux d\'attaque le plus élevé (pour 10 000 habitants)**")

        top10_taux = cases_by_area.nlargest(10, 'Taux_Attaque_10K')

        fig_taux = px.bar(
            top10_taux,
            x='Taux_Attaque_10K',
            y='Aire_Sante',
            orientation='h',
            title="Top 10 - Taux d\'attaque",
            labels={'Taux_Attaque_10K': 'Taux pour 10K hab.', 'Aire_Sante': 'Aire de santé'},
            color='Taux_Attaque_10K',
            color_continuous_scale='Reds',
            text='Taux_Attaque_10K'
        )
        fig_taux.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_taux.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_taux, use_container_width=True)

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

    with tab_cas:
        st.markdown("**Top 10 - Nombre de cas le plus élevé**")

        top10_cas = cases_by_area.nlargest(10, 'Cas_Observes')

        fig_cas = px.bar(
            top10_cas,
            x='Cas_Observes',
            y='Aire_Sante',
            orientation='h',
            title="Top 10 - Nombre de cas",
            labels={'Cas_Observes': 'Nombre de cas', 'Aire_Sante': 'Aire de santé'},
            color='Cas_Observes',
            color_continuous_scale='Oranges',
            text='Cas_Observes'
        )
        fig_cas.update_traces(textposition='outside')
        fig_cas.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_cas, use_container_width=True)

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
            st.metric("Taux max", f"{taux_max:.1f}/10K", aire_taux_max)
        else:
            st.metric("Taux max", "N/A")

    with col2:
        taux_moyen = cases_by_area['Taux_Attaque_10K'].mean()
        st.metric("Taux moyen", f"{taux_moyen:.1f}/10K")

    with col3:
        aires_alerte = len(cases_by_area[cases_by_area['Taux_Attaque_10K'] > 10])
        st.metric("Aires alerte (>10/10K)", aires_alerte, delta_color="inverse")

    st.markdown("---")

    # ============================================================
    # ANALYSE TEMPORELLE
    # ============================================================

    st.header("📈 Analyse Temporelle")

    # Créer colonne Semaine_Label
    df['Semaine_Label'] = df['Annee'].astype(str) + '-S' + df['Semaine_Epi'].astype(str).str.zfill(2)

    # Cas par semaine
    weekly_cases = df.groupby('Semaine_Label').size().reset_index(name='Cas')

    fig_temporal = px.line(
        weekly_cases,
        x='Semaine_Label',
        y='Cas',
        title='Évolution hebdomadaire des cas',
        markers=True
    )
    fig_temporal.update_layout(xaxis_title="Semaine", yaxis_title="Nombre de cas", height=400)
    st.plotly_chart(fig_temporal, use_container_width=True)

    # Distribution âge (CONDITIONNEL)
    if 'Age_Mois' in df.columns and df['Age_Mois'].notna().sum() > 0:
        st.subheader("📊 Distribution par Âge")

        df['Groupe_Age'] = pd.cut(
            df['Age_Mois'],
            bins=[0, 12, 60, 120, 180],
            labels=['<1 an', '1-5 ans', '5-10 ans', '10-15 ans']
        )

        age_dist = df['Groupe_Age'].value_counts().reset_index()
        age_dist.columns = ['Groupe', 'Cas']

        # CORRECTION ICI : Utiliser des guillemets doubles
        fig_age = px.bar(
            age_dist,
            x='Groupe',
            y='Cas',
            title="Distribution par groupe d\'âge",
            color='Cas',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_age, use_container_width=True)
    else:
        st.info("ℹ️ Données d\'âge non disponibles")

# ============================================================
# ONGLET 2 : CARTOGRAPHIE
# ============================================================

with tab2:
    st.header("🗺️ Cartographie de la Situation")

    # Fusionner avec cases
    sa_gdf_with_cases = sa_gdf_enrichi.merge(
        cases_by_area[['Aire_Sante', 'Cas_Observes', 'Taux_Attaque_10K']],
        left_on='health_area',
        right_on='Aire_Sante',
        how='left'
    )

    sa_gdf_with_cases['Cas_Observes'] = sa_gdf_with_cases['Cas_Observes'].fillna(0).astype(int)
    sa_gdf_with_cases['Taux_Attaque_10K'] = sa_gdf_with_cases['Taux_Attaque_10K'].fillna(0)

    # Carte choroplèthe
    st.subheader("🗺️ Carte choroplèthe - Nombre de cas")

    # Centrer la carte
    center_lat = sa_gdf_with_cases.geometry.centroid.y.mean()
    center_lon = sa_gdf_with_cases.geometry.centroid.x.mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles='CartoDB positron')

    # Choroplèthe
    folium.Choropleth(
        geo_data=sa_gdf_with_cases,
        data=sa_gdf_with_cases,
        columns=['health_area', 'Cas_Observes'],
        key_on='feature.properties.health_area',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Nombre de cas',
        nan_fill_color='lightgray'
    ).add_to(m)

    # Popup
    for idx, row in sa_gdf_with_cases.iterrows():
        folium.Marker(
            location=[row.geometry.centroid.y, row.geometry.centroid.x],
            popup=f"<b>{row['health_area']}</b><br>Cas: {int(row['Cas_Observes'])}<br>Taux: {row['Taux_Attaque_10K']:.1f}/10K",
            icon=folium.Icon(color='red' if row['Cas_Observes'] > 10 else 'orange' if row['Cas_Observes'] > 5 else 'blue', icon='info-sign')
        ).add_to(m)

    st_folium(m, width=900, height=600)

    st.info(f"📍 {len(sa_gdf_with_cases[sa_gdf_with_cases['Cas_Observes'] > 0])} aires avec cas signalés")

# ============================================================
# ONGLET 3 : MODÉLISATION ET PRÉDICTION
# ============================================================

with tab3:
    st.header("🔮 Modélisation Prédictive")

    st.info(f"📊 Modèle sélectionné : **{modele_choisi}**")
    st.info(f"📅 Prédiction sur **{n_weeks_pred} semaines** (~{pred_mois} mois)")

    # Session state pour la prédiction
    if 'prediction_lancee' not in st.session_state:
        st.session_state.prediction_lancee = False

    if st.button("🚀 Lancer la Prédiction", type="primary", use_container_width=True):
        st.session_state.prediction_lancee = True

    if st.session_state.prediction_lancee:

        with st.spinner("🔄 Préparation des données pour la modélisation..."):

            # ========== PRÉPARATION FEATURES ==========

            # Agréger par aire et semaine
            weekly_features = df.groupby(['Aire_Sante', 'Annee', 'Semaine_Epi']).agg({
                'ID_Cas': 'count'
            }).reset_index()
            weekly_features.columns = ['Aire_Sante', 'Annee', 'Semaine_Epi', 'Cas_Observes']

            # Ajouter semaine label
            weekly_features['Semaine_Label'] = weekly_features['Annee'].astype(str) + '-S' + weekly_features['Semaine_Epi'].astype(str).str.zfill(2)

            # Lags (4 dernières semaines)
            weekly_features = weekly_features.sort_values(['Aire_Sante', 'Annee', 'Semaine_Epi'])

            for lag in [1, 2, 3, 4]:
                weekly_features[f'Cas_Lag{lag}'] = weekly_features.groupby('Aire_Sante')['Cas_Observes'].shift(lag)

            # Remplir NaN des lags avec 0
            for lag in [1, 2, 3, 4]:
                weekly_features[f'Cas_Lag{lag}'] = weekly_features[f'Cas_Lag{lag}'].fillna(0)

            # Merger avec données géographiques
            weekly_features = weekly_features.merge(
                sa_gdf_enrichi[['health_area', 'Pop_Totale', 'Pop_Enfants', 'Densite_Pop', 
                                'Densite_Enfants', 'Urbanisation', 'Temperature_Moy', 
                                'Humidite_Moy', 'Taux_Vaccination']],
                left_on='Aire_Sante',
                right_on='health_area',
                how='left'
            )

            # Encoder Urbanisation
            if 'Urbanisation' in weekly_features.columns:
                le = LabelEncoder()
                weekly_features['Urban_Encoded'] = le.fit_transform(weekly_features['Urbanisation'].fillna('Rural'))
            else:
                weekly_features['Urban_Encoded'] = 0

            # Calculer age moyen si disponible
            if 'Age_Mois' in df.columns:
                age_by_area = df.groupby('Aire_Sante')['Age_Mois'].mean().reset_index()
                age_by_area.columns = ['Aire_Sante', 'Age_Moyen']
                weekly_features = weekly_features.merge(age_by_area, on='Aire_Sante', how='left')
            else:
                weekly_features['Age_Moyen'] = 0

            # Calculer Non_Vaccines
            if 'Taux_Vaccination' in weekly_features.columns:
                weekly_features['Non_Vaccines'] = 100 - weekly_features['Taux_Vaccination'].fillna(80)
            else:
                weekly_features['Non_Vaccines'] = 20

            # Remplir NaN
            weekly_features = weekly_features.fillna(0)

            # Créer coefficient climatique
            if 'Temperature_Moy' in weekly_features.columns and 'Humidite_Moy' in weekly_features.columns:
                weekly_features['Coef_Climatique'] = (
                    weekly_features['Temperature_Moy'] * 0.4 +
                    weekly_features['Humidite_Moy'] * 0.6
                ) / 100
            else:
                weekly_features['Coef_Climatique'] = 0.5

            st.success(f"✅ {len(weekly_features)} observations préparées")

        with st.spinner("🤖 Entraînement du modèle..."):

            # Colonnes features
            feature_cols = [
                'Cas_Lag1', 'Cas_Lag2', 'Cas_Lag3', 'Cas_Lag4',
                'Pop_Totale', 'Pop_Enfants', 'Densite_Pop', 'Densite_Enfants',
                'Urban_Encoded', 'Age_Moyen', 'Non_Vaccines', 'Coef_Climatique'
            ]

            # Filtrer colonnes existantes
            feature_cols = [col for col in feature_cols if col in weekly_features.columns]

            X = weekly_features[feature_cols].values
            y = weekly_features['Cas_Observes'].values

            # Normalisation
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Choisir le modèle
            if modele_choisi == "GradientBoosting (Recommandé)":
                model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
            elif modele_choisi == "RandomForest":
                model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            elif modele_choisi == "Ridge Regression":
                model = Ridge(alpha=1.0)
            elif modele_choisi == "Lasso Regression":
                model = Lasso(alpha=0.1)
            else:  # Decision Tree
                model = DecisionTreeRegressor(max_depth=8, random_state=42)

            # Appliquer poids manuels si mode expert
            if mode_importance == "👨‍⚕️ Manuel (Expert)" and poids_normalises:
                # Regrouper features par catégorie
                column_weights = {}

                # Historique Cas
                for col in ['Cas_Lag1', 'Cas_Lag2', 'Cas_Lag3', 'Cas_Lag4']:
                    if col in feature_cols:
                        column_weights[col] = poids_normalises.get('Historique_Cas', 0.4) / 4

                # Vaccination
                for col in ['Taux_Vaccination', 'Non_Vaccines']:
                    if col in feature_cols:
                        column_weights[col] = poids_normalises.get('Vaccination', 0.35) / 2

                # Démographie
                for col in ['Pop_Totale', 'Pop_Enfants', 'Densite_Pop', 'Densite_Enfants', 'Age_Moyen']:
                    if col in feature_cols:
                        column_weights[col] = poids_normalises.get('Demographie', 0.15) / 5

                # Urbanisation
                if 'Urban_Encoded' in feature_cols:
                    column_weights['Urban_Encoded'] = poids_normalises.get('Urbanisation', 0.08)

                # Climat
                if 'Coef_Climatique' in feature_cols:
                    column_weights['Coef_Climatique'] = poids_normalises.get('Climat', 0.02)

                # Appliquer poids
                for idx, col in enumerate(feature_cols):
                    if col in column_weights:
                        X_scaled[:, idx] = X_scaled[:, idx] * column_weights[col]

            # Entraîner
            model.fit(X_scaled, y)

            # Score
            score = model.score(X_scaled, y)
            st.success(f"✅ Modèle entraîné | Score R² : {score:.3f}")

        with st.spinner("🔮 Génération des prédictions..."):

            # Dernière semaine dans les données
            derniere_semaine_epi = weekly_features['Semaine_Epi'].max()
            derniere_annee = weekly_features['Annee'].max()

            # Prédictions futures
            future_predictions = []

            for aire in weekly_features['Aire_Sante'].unique():
                aire_data = weekly_features[weekly_features['Aire_Sante'] == aire].sort_values(['Annee', 'Semaine_Epi'])

                if aire_data.empty:
                    continue

                last_obs = aire_data.iloc[-1]
                last_4_weeks = aire_data.tail(4)['Cas_Observes'].values

                if len(last_4_weeks) < 4:
                    last_4_weeks = np.pad(last_4_weeks, (4-len(last_4_weeks), 0), 'edge')

                for i in range(1, n_weeks_pred + 1):
                    nouvelle_semaine_epi = (derniere_semaine_epi + i - 1) % 52 + 1
                    nouvelle_annee = derniere_annee + (derniere_semaine_epi + i - 1) // 52

                    # Créer features
                    future_row = {
                        'Aire_Sante': aire,
                        'Annee': nouvelle_annee,
                        'Semaine_Epi': nouvelle_semaine_epi,
                        'Semaine_Label': f"{nouvelle_annee}-S{nouvelle_semaine_epi:02d}",
                        'Cas_Lag1': last_4_weeks[-1],
                        'Cas_Lag2': last_4_weeks[-2] if len(last_4_weeks) >= 2 else last_4_weeks[-1],
                        'Cas_Lag3': last_4_weeks[-3] if len(last_4_weeks) >= 3 else last_4_weeks[-1],
                        'Cas_Lag4': last_4_weeks[-4] if len(last_4_weeks) >= 4 else last_4_weeks[-1],
                        'Pop_Totale': last_obs.get('Pop_Totale', 0),
                        'Pop_Enfants': last_obs.get('Pop_Enfants', 0),
                        'Densite_Pop': last_obs.get('Densite_Pop', 0),
                        'Densite_Enfants': last_obs.get('Densite_Enfants', 0),
                        'Urban_Encoded': last_obs.get('Urban_Encoded', 0),
                        'Age_Moyen': last_obs.get('Age_Moyen', 0),
                        'Non_Vaccines': last_obs.get('Non_Vaccines', 20),
                        'Coef_Climatique': last_obs.get('Coef_Climatique', 0.5)
                    }

                    # Prédire
                    X_future = np.array([future_row[col] for col in feature_cols]).reshape(1, -1)
                    X_future_scaled = scaler.transform(X_future)

                    pred = model.predict(X_future_scaled)[0]
                    pred = max(0, pred)  # Pas de valeurs négatives

                    future_predictions.append({
                        'Aire_Sante': aire,
                        'Annee': nouvelle_annee,
                        'Semaine_Epi': nouvelle_semaine_epi,
                        'Semaine_Label': future_row['Semaine_Label'],
                        'Predicted_Cases': pred
                    })

                    # Mettre à jour lags
                    last_4_weeks = np.append(last_4_weeks[1:], pred)

            future_df = pd.DataFrame(future_predictions)
            future_df['Predicted_Cases'] = future_df['Predicted_Cases'].round(0).astype(int)

            st.success(f"✅ {len(future_df)} prédictions générées")

        # ========== AFFICHAGE DES RÉSULTATS ==========

        st.subheader("📊 Résultats des Prédictions")

        # Tableau synthèse
        moyenne_historique = weekly_features.groupby('Aire_Sante')['Cas_Observes'].mean().reset_index()
        moyenne_historique.columns = ['Aire_Sante', 'Moyenne_Historique']

        risk_df = future_df.groupby('Aire_Sante').agg({
            'Predicted_Cases': ['sum', 'max', 'mean']
        }).reset_index()

        risk_df.columns = ['Aire_Sante', 'Cas_Predits_Total', 'Cas_Predits_Max', 'Cas_Predits_Moyen']
        risk_df = risk_df.merge(moyenne_historique, on='Aire_Sante', how='left')

        risk_df['Variation_Pct'] = ((risk_df['Cas_Predits_Moyen'] - risk_df['Moyenne_Historique']) / 
                                     risk_df['Moyenne_Historique'].replace(0, 1)) * 100

        risk_df['Categorie_Variation'] = pd.cut(
            risk_df['Variation_Pct'],
            bins=[-np.inf, -seuil_baisse, -10, 10, seuil_hausse, np.inf],
            labels=['Forte baisse', 'Baisse modérée', 'Stable', 'Hausse modérée', 'Forte hausse']
        )

        # Convertir en string avant fillna
        risk_df['Categorie_Variation'] = risk_df['Categorie_Variation'].astype(str)
        risk_df['Categorie_Variation'] = risk_df['Categorie_Variation'].replace('nan', 'Aucune donnée')

        risk_df = risk_df.sort_values('Variation_Pct', ascending=False)

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

        # Graphiques
        col1, col2 = st.columns(2)

        with col1:
            top_risk = risk_df.head(10)

            fig_top = px.bar(
                top_risk,
                x='Cas_Predits_Total',
                y='Aire_Sante',
                orientation='h',
                title='Top 10 - Aires à risque',
                color='Variation_Pct',
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig_top, use_container_width=True)

        with col2:
            categories_count = risk_df['Categorie_Variation'].value_counts().reset_index()
            categories_count.columns = ['Catégorie', 'Nombre']

            fig_cat = px.pie(
                categories_count,
                names='Catégorie',
                values='Nombre',
                title='Répartition des variations'
            )
            st.plotly_chart(fig_cat, use_container_width=True)

        # Heatmap temporelle
        st.subheader("🗓️ Heatmap Hebdomadaire")

        heatmap_data = future_df.pivot_table(
            values='Predicted_Cases',
            index='Aire_Sante',
            columns='Semaine_Label',
            aggfunc='sum',
            fill_value=0
        )

        heatmap_data = heatmap_data.round(0).astype(int)

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='Reds',
            colorbar=dict(
                title=dict(
                    text="Cas<br>prédits",
                    side="right"
                )
            )
        ))

        fig_heatmap.update_layout(
            title='Prédictions par aire et par semaine',
            xaxis_title='Semaine',
            yaxis_title='Aire de santé',
            height=600
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)

        # Alertes
        st.subheader("⚠️ Alertes et Recommandations")

        aires_alerte = risk_df[risk_df['Cas_Predits_Max'] >= seuil_alerte_epidemique]

        if len(aires_alerte) > 0:
            st.error(f"🚨 {len(aires_alerte)} aires dépasseront le seuil d'alerte ({seuil_alerte_epidemique} cas/semaine)")

            st.dataframe(
                aires_alerte[['Aire_Sante', 'Cas_Predits_Max', 'Cas_Predits_Moyen', 'Variation_Pct']]
                .style.format({
                    'Cas_Predits_Max': '{:.0f}',
                    'Cas_Predits_Moyen': '{:.1f}',
                    'Variation_Pct': '{:.1f}%'
                })
                .background_gradient(subset=['Cas_Predits_Max'], cmap='Reds'),
                use_container_width=True
            )
        else:
            st.success("✅ Aucune aire ne dépassera le seuil d'alerte")

        # Export
        st.subheader("💾 Export des Résultats")

        col1, col2 = st.columns(2)

        with col1:
            csv = risk_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger Synthèse (CSV)",
                data=csv,
                file_name=f"predictions_rougeole_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

        with col2:
            csv_details = future_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger Détails (CSV)",
                data=csv_details,
                file_name=f"predictions_details_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
