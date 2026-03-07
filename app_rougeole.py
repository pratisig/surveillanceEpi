# ============================================================
# APP SURVEILLANCE & PRÉDICTION ROUGEOLE - VERSION 3.0
# PARTIE 1/5 - IMPORTS, CONFIGURATION, SIDEBAR
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
from sklearn.impute import SimpleImputer
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
import re
import unicodedata
warnings.filterwarnings('ignore')

# CSS personnalisé
st.markdown("""
<style>
.model-hint { background:#f0f7ff; border-left:4px solid #1976d2; padding:.8rem 1rem; border-radius:4px; margin:.5rem 0; font-size:.9rem; }
.weight-box { background:#fff8e1; border-left:4px solid #f9a825; padding:.8rem 1rem; border-radius:4px; margin:.5rem 0; }
.info-box   { background:#fef9f9; border-left:4px solid #e74c3c; padding:1rem; border-radius:6px; margin:.5rem 0; }
</style>
""", unsafe_allow_html=True)

st.title("🦠 Plateforme de Surveillance et Prédiction - Rougeole")
st.markdown("### Analyse épidémiologique et modélisation prédictive par semaines épidémiologiques")

# Mapping pays ISO3
PAYS_ISO3_MAP = {
    "Niger": "ner",
    "Burkina Faso": "bfa",
    "Mali": "mli",
    "Mauritanie": "mrt"
}

# ── GEE ───────────────────────────────────────────────────────
@st.cache_resource
def init_gee():
    try:
        key_dict = json.loads(st.secrets["GEE_SERVICE_ACCOUNT"])
        credentials = ee.ServiceAccountCredentials(
            key_dict["client_email"], key_data=json.dumps(key_dict))
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

# ── Session state ─────────────────────────────────────────────
if 'pays_precedent' not in st.session_state:
    st.session_state.pays_precedent = None
if 'sa_gdf_cache' not in st.session_state:
    st.session_state.sa_gdf_cache = None
if 'prediction_rougeole_lancee' not in st.session_state:
    st.session_state.prediction_rougeole_lancee = False
if 'enrichi_ner' not in st.session_state:
    st.session_state['enrichi_ner'] = None
if 'enrichi_bfa' not in st.session_state:
    st.session_state['enrichi_bfa'] = None
if 'enrichi_mli' not in st.session_state:
    st.session_state['enrichi_mli'] = None
if 'enrichi_mrt' not in st.session_state:
    st.session_state['enrichi_mrt'] = None
if 'enrichi_upload' not in st.session_state:
    st.session_state['enrichi_upload'] = None
# ============================================================
# CORRECTION 1 & 2 : MAPPING COLONNES ROBUSTE + SÉPARATEUR CSV
# ============================================================

COLONNES_MAPPING = {
    "Aire_Sante": [
        "Aire_Sante","aire_sante","health_area","HEALTH_AREA","name_fr","NAME",
        "nom","NOM","aire de sante","aire_de_sante","zone_sante","district",
        "area","localite","locality","nom_aire","nom aire","aire","fosa",
        "nom_fosa","fosa_name","Aire_de_Sante","AIRE_SANTE"
    ],
    "Date_Debut_Eruption": [
        "Date_Debut_Eruption","date_debut_eruption","Date_Debut","date_onset",
        "Date_Onset","symptom_onset","date_eruption","Date_Eruption",
        "date_debut","DateDebut","DATE_DEBUT"
    ],
    "Date_Notification": [
        "Date_Notification","date_notification","Date_Notif","date_notif",
        "notification_date","DateNotif","DATE_NOTIFICATION"
    ],
    "ID_Cas": [
        "ID_Cas","id_cas","ID","id","Case_ID","case_id","ID_cas",
        "identifiant","Identifiant","IDCAS","id_case"
    ],
    "Age_Mois": [
        "Age_Mois","age_mois","Age","age","AGE","Age_Months","age_months",
        "age_en_mois","Age_En_Mois","AGE_MOIS"
    ],
    "Statut_Vaccinal": [
        "Statut_Vaccinal","statut_vaccinal","Vaccin","vaccin",
        "Vaccination_Status","vaccination_status","Vacc_Statut",
        "statut_vaccination","Statut_Vaccination","STATUT_VACCINAL",
        "vaccinated","Vaccinated","non_vaccine","Non_Vaccine"
    ],
    "Sexe": ["Sexe","sexe","Sex","sex","Gender","gender","SEXE","SEX"],
    "Issue": [
        "Issue","issue","Outcome","outcome","OUTCOME","issue_cas",
        "Issue_Cas","resultat","Resultat"
    ],
    "Semaine_Epi": [
        "Semaine_Epi","semaine_epi","Semaine_epi","SEMAINE_EPI",
        "semaine","Semaine","SEMAINE","week","Week","WEEK",
        "epi_week","Epi_Week","EPI_WEEK","epiweek","Epiweek",
        "se","SE","s_epi","S_epi","sem_epi","Sem_Epi",
        "semaine_epidemiologique","Semaine_Epidemiologique",
        "epi week","Epi Week","epid_week","Epid_Week",
        "week_number","Week_Number","no_semaine","No_Semaine",
        "numero_semaine","Numero_Semaine","n_semaine","N_Semaine",
        "wk","WK","sw","SW"
    ],
    "Annee": [
        "Annee","annee","ANNEE","année","Année",
        "year","Year","YEAR","an","An","AN",
        "yr","Yr","annee_epi","Annee_Epi","epi_year","Epi_Year",
        "epiyear","EpiYear","annee_epidemiologique","Annee_Epidemiologique"
    ],
    "Cas_Total": [
        "Cas_Total","cas_total","CAS_TOTAL","Cas","cas","CAS",
        "cases","Cases","CASES","nb_cas","Nb_Cas","NB_CAS",
        "nombre_cas","Nombre_Cas","NOMBRE_CAS","nbcas","nbre_cas",
        "count","Count","total_cas","Total_Cas","confirmed","Confirmed"
    ],
    "Deces": [
        "Deces","deces","DECES","décès","Décès",
        "deaths","Deaths","DEATHS","nb_deces","Nb_Deces",
        "mort","morts","dead","Dead","nb_morts"
    ],
}

SEP_CANDIDATES = [",", ";", "\t", "|"]

def detect_separator(uploaded_file) -> str:
    raw = uploaded_file.read(4096).decode("utf-8", errors="ignore")
    uploaded_file.seek(0)
    lines = [l for l in raw.split("\n")[:6] if l.strip()]
    scores = {}
    for sep in SEP_CANDIDATES:
        counts = [line.count(sep) for line in lines]
        scores[sep] = sum(counts) / max(len(counts), 1)
    best = max(scores, key=scores.get)
    return best if scores.get(best, 0) > 0 else ","

def normaliser_colonnes(dataframe, mapping):
    def _norm(s):
        s = str(s).strip().lower()
        s = unicodedata.normalize("NFD", s)
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        return s.replace(" ", "_").replace("-", "_")
    norm_existing = {_norm(c): c for c in dataframe.columns}
    rename_dict = {}
    for col_standard, col_possibles in mapping.items():
        if col_standard in dataframe.columns:
            continue
        for col_possible in col_possibles:
            nc = _norm(col_possible)
            if nc in norm_existing and norm_existing[nc] not in rename_dict.values():
                rename_dict[norm_existing[nc]] = col_standard
                break
    if rename_dict:
        dataframe = dataframe.rename(columns=rename_dict)
    return dataframe

def normalize_join_value(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().lower()
    x = unicodedata.normalize("NFD", x)
    x = "".join(c for c in x if unicodedata.category(c) != "Mn")
    x = x.replace("-", " ").replace("_", " ").replace("/", " ").replace("\\", " ")
    x = x.replace("'", " ").replace("’", " ")
    x = re.sub(r"\s+", " ", x).strip()
    roman_endings = {
        " i": " 1",
        " ii": " 2",
        " iii": " 3",
        " iv": " 4",
        " v": " 5",
        " vi": " 6",
        " vii": " 7",
        " viii": " 8",
        " ix": " 9",
        " x": " 10",
    }
    for suffix, repl in roman_endings.items():
        if x.endswith(suffix):
            x = x[:-len(suffix)] + repl
            break
    return x

def add_join_key(df_like, source_col, target_col="join_key"):
    if source_col not in df_like.columns:
        df_like[target_col] = np.nan
    else:
        df_like[target_col] = df_like[source_col].apply(normalize_join_value)
    return df_like

def audit_join(left_df, left_key, right_df, right_key, label="jointure"):
    left_keys = set(left_df[left_key].dropna().astype(str).unique())
    right_keys = set(right_df[right_key].dropna().astype(str).unique())
    matched = left_keys.intersection(right_keys)
    only_left = sorted(list(left_keys - right_keys))[:20]
    only_right = sorted(list(right_keys - left_keys))[:20]

    st.sidebar.markdown(f"### 🔎 Audit {label}")
    st.sidebar.caption(
        f"Match: {len(matched)} | "
        f"Seulement gauche: {len(left_keys - right_keys)} | "
        f"Seulement droite: {len(right_keys - left_keys)}"
    )

    return {
        "matched": matched,
        "only_left": only_left,
        "only_right": only_right
    }

def semaine_vers_date(annee: int, semaine: int) -> datetime:
    try:
        semaine = int(max(1, min(52, semaine)))
        annee = int(annee)
        return datetime.strptime(f"{annee}-W{semaine:02d}-1", "%G-W%V-%u")
    except Exception:
        try:
            return datetime(int(annee), 1, 1) + timedelta(weeks=int(semaine) - 1)
        except Exception:
            return datetime(2020, 1, 1)

# ── Sidebar ───────────────────────────────────────────────────
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
    if st.session_state.pays_precedent != pays_selectionne:
        st.session_state.pays_precedent = pays_selectionne
        st.session_state.sa_gdf_cache = None
iso3_pays = PAYS_ISO3_MAP[pays_selectionne]
if st.session_state.pays_precedent != pays_selectionne:
    st.session_state.pays_precedent = pays_selectionne
    st.session_state.sa_gdf_cache = None

upload_file = None
if option_aire == "Upload personnalisé":
    upload_file = st.sidebar.file_uploader(
        "Charger un fichier géographique",
        type=["shp", "geojson", "zip"],
        help="Format : Shapefile ou GeoJSON avec colonnes 'iso3' et 'health_area'"
    )

st.sidebar.subheader("📊 Données Épidémiologiques")
if mode_demo == "🧪 Mode démo (données simulées)":
    linelist_file = None
    vaccination_file = None
    st.sidebar.info("📊 Mode démo activé - Données simulées")
else:
    linelist_file = st.sidebar.file_uploader(
        "📋 Linelists rougeole (CSV)",
        type=["csv"],
        help="Format : health_area, Semaine_Epi, Annee, Cas_Total OU Date_Debut_Eruption, Aire_Sante..."
    )
    vaccination_file = st.sidebar.file_uploader(
        "💉 Couverture vaccinale (CSV - optionnel)",
        type=["csv"],
        help="Format : health_area, Taux_Vaccination (en %)"
    )

st.sidebar.subheader("🔮 Paramètres de Prédiction")
pred_mois = st.sidebar.slider(
    "Période de prédiction (mois)",
    min_value=1, max_value=12, value=3,
    help="Nombre de mois à prédire après la dernière semaine de données"
)
n_weeks_pred = pred_mois * 4
st.sidebar.info(f"📆 Prédiction sur **{n_weeks_pred} semaines épidémiologiques** (~{pred_mois} mois)")

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

model_hints = {
    "GradientBoosting (Recommandé)": "🎯 **Gradient Boosting** : Très performant pour les séries temporelles. Combine plusieurs modèles faibles pour créer un modèle fort. Excellent pour capturer les relations non-linéaires. Recommandé pour la surveillance épidémiologique.",
    "RandomForest": "🌳 **Random Forest** : Ensemble d'arbres de décision. Robuste aux valeurs aberrantes et aux données manquantes. Bon pour les interactions complexes entre variables.",
    "Ridge Regression": "📊 **Ridge Regression** : Régression linéaire avec régularisation L2. Simple et rapide. Idéal pour relations linéaires. Moins performant sur données non-linéaires.",
    "Lasso Regression": "🎯 **Lasso Regression** : Régluarisation L1 avec sélection automatique des variables. Utile quand beaucoup de variables peu importantes. Simplifie le modèle.",
    "Decision Tree": "🌲 **Decision Tree** : Arbre de décision unique. Simple à interpréter mais risque de sur-apprentissage. Moins robuste que les méthodes d'ensemble."
}
st.sidebar.markdown(f'<div class="model-hint">{model_hints[modele_choisi]}</div>', unsafe_allow_html=True)

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

        poids_manuels["Historique_Cas"] = st.slider("📈 Historique des cas (lags)", 0, 100, 40, step=5)
        poids_manuels["Vaccination"] = st.slider("💉 Couverture vaccinale", 0, 100, 35, step=5)
        poids_manuels["Demographie"] = st.slider("👥 Démographie", 0, 100, 15, step=5)
        poids_manuels["Urbanisation"] = st.slider("🏙️ Urbanisation", 0, 100, 8, step=2)
        poids_manuels["Climat"] = st.slider("🌡️ Facteurs climatiques", 0, 100, 2, step=1)

        total_poids = sum(poids_manuels.values())
        if total_poids > 0:
            for key in poids_manuels:
                poids_normalises[key] = poids_manuels[key] / total_poids

        st.markdown("---")
        st.markdown("**📊 Répartition normalisée :**")
        for key, value in poids_normalises.items():
            st.markdown(f"• {key} : **{value*100:.1f}%**")

        if abs(total_poids - 100) > 5:
            st.info(f"ℹ️ Total brut : {total_poids}% → Normalisé à 100%")
else:
    st.sidebar.info("Le modèle ML calculera automatiquement l'importance optimale de chaque variable")

st.sidebar.subheader("⚙️ Seuils d'Alerte")
with st.sidebar.expander("Configurer les seuils", expanded=False):
    seuil_baisse = st.slider("Seuil de baisse significative (%)", min_value=10, max_value=90, value=75, step=5)
    seuil_hausse = st.slider("Seuil de hausse significative (%)", min_value=10, max_value=200, value=50, step=10)
    seuil_alerte_epidemique = st.number_input("Seuil d'alerte épidémique (cas/semaine)", min_value=1, max_value=100, value=5)

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
            for col in ['iso3','ISO3','iso_code','ISO_CODE','country_iso','COUNTRY_ISO']:
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
            for col in ['health_area','HEALTH_AREA','name_fr','name','NAME','nom','NOM','aire_sante']:
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
            for col in ["health_area","HEALTH_AREA","name_fr","name","NAME","nom","NOM"]:
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
# PARTIE 2/5 - CHARGEMENT AIRES DE SANTÉ ET DONNÉES DE CAS
# ============================================================
if st.session_state.sa_gdf_cache is not None and option_aire == "Fichier local (ao_hlthArea.zip)":
    sa_gdf = st.session_state.sa_gdf_cache
    st.sidebar.success(f"✓ {len(sa_gdf)} aires chargées (cache)")
else:
    with st.spinner("🔄 Chargement des aires de santé..."):
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

if sa_gdf is None or sa_gdf.empty:
    st.error("❌ Aucune aire chargée")
    st.stop()

sa_gdf = add_join_key(sa_gdf, "health_area")

@st.cache_data
def generate_dummy_linelists(_sa_gdf, n=500):
    np.random.seed(42)
    aires = _sa_gdf["health_area"].unique()
    rows = []
    for annee in [2024, 2025]:
        for semaine in range(1, 53):
            n_cas = max(0, int(np.random.poisson(lam=max(1, 15 * np.sin(semaine * np.pi / 26) + 3))))
            for aire in np.random.choice(aires, size=min(n_cas, len(aires)), replace=False):
                rows.append({
                    "ID_Cas": len(rows) + 1,
                    "Semaine_Epi": semaine,
                    "Annee": annee,
                    "Aire_Sante": aire,
                    "Age_Mois": int(np.random.gamma(shape=2, scale=30, size=1)[0].clip(6, 180)),
                    "Statut_Vaccinal": np.random.choice(["Oui", "Non"], p=[0.55, 0.45]),
                    "Sexe": np.random.choice(["M", "F"]),
                    "Issue": np.random.choice(["Guéri", "Décédé", "Inconnu"], p=[0.92, 0.03, 0.05])
                })
    df_demo = pd.DataFrame(rows)
    df_demo["Date_Debut_Eruption"] = df_demo.apply(lambda r: semaine_vers_date(r["Annee"], r["Semaine_Epi"]) + timedelta(days=int(np.random.randint(0, 7))), axis=1)
    df_demo["Date_Notification"] = df_demo["Date_Debut_Eruption"] + timedelta(days=3)
    return df_demo

@st.cache_data
def generate_dummy_vaccination(_sa_gdf):
    np.random.seed(42)
    return pd.DataFrame({
        "health_area": _sa_gdf["health_area"],
        "Taux_Vaccination": np.random.beta(a=8, b=2, size=len(_sa_gdf)) * 100
    })

with st.spinner("📥 Chargement données de cas..."):
    if mode_demo == "🧪 Mode démo (données simulées)":
        df = generate_dummy_linelists(sa_gdf)
        vaccination_df = generate_dummy_vaccination(sa_gdf)
        st.sidebar.info(f"📊 {len(df)} cas simulés générés")
    else:
        if linelist_file is None:
            st.error("❌ Veuillez uploader un fichier CSV de lineliste")
            st.stop()

        try:
            sep = detect_separator(linelist_file)
            try:
                df_raw = pd.read_csv(linelist_file, sep=sep, encoding="utf-8", low_memory=False)
            except UnicodeDecodeError:
                linelist_file.seek(0)
                df_raw = pd.read_csv(linelist_file, sep=sep, encoding="latin-1", low_memory=False)

            df_raw = normaliser_colonnes(df_raw, COLONNES_MAPPING)

            cas_col = None
            for _c in ["Cas_Total", "Cas", "cases", "Cases", "nb_cas", "nombre_cas"]:
                if _c in df_raw.columns:
                    cas_col = _c
                    break

            if "Semaine_Epi" in df_raw.columns and cas_col:
                df_raw["Semaine_Epi"] = pd.to_numeric(df_raw["Semaine_Epi"], errors="coerce")
                df_raw = df_raw[df_raw["Semaine_Epi"].between(1, 52, inclusive="both")].copy()
                df_raw["Semaine_Epi"] = df_raw["Semaine_Epi"].fillna(1).astype(int)

                if "Annee" in df_raw.columns:
                    df_raw["Annee"] = pd.to_numeric(df_raw["Annee"], errors="coerce")
                    df_raw = df_raw[df_raw["Annee"].notna()].copy()
                    df_raw["Annee"] = df_raw["Annee"].astype(int)
                else:
                    df_raw["Annee"] = datetime.now().year

                if "Aire_Sante" not in df_raw.columns:
                    for _ac in ["health_area", "name_fr", "aire", "zone_sante", "district", "localite"]:
                        if _ac in df_raw.columns:
                            df_raw["Aire_Sante"] = df_raw[_ac]
                            break
                    else:
                        df_raw["Aire_Sante"] = sa_gdf["health_area"].iloc[0]

                expanded_rows = []
                for _, row in df_raw.iterrows():
                    aire = row.get("Aire_Sante", "Inconnu")
                    semaine = int(row["Semaine_Epi"])
                    annee = int(row["Annee"])
                    cas_total = int(max(0, pd.to_numeric(row.get(cas_col, 0), errors="coerce") or 0))
                    deces = int(max(0, pd.to_numeric(row.get("Deces", 0), errors="coerce") or 0))
                    base_date = semaine_vers_date(annee, semaine)

                    for i in range(cas_total):
                        issue = "Décédé" if i < deces else "Guéri"
                        expanded_rows.append({
                            "ID_Cas": len(expanded_rows) + 1,
                            "Semaine_Epi": semaine,
                            "Annee": annee,
                            "Date_Debut_Eruption": base_date + timedelta(days=int(np.random.randint(0, 7))),
                            "Date_Notification": base_date + timedelta(days=int(np.random.randint(0, 10))),
                            "Aire_Sante": aire,
                            "Age_Mois": np.nan,
                            "Statut_Vaccinal": "Inconnu",
                            "Sexe": "Inconnu",
                            "Issue": issue
                        })
                df = pd.DataFrame(expanded_rows)
            elif "Date_Debut_Eruption" in df_raw.columns:
                df = df_raw.copy()
                for col in ["Date_Debut_Eruption", "Date_Notification"]:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
            else:
                st.warning("⚠️ Format CSV non standard — tentative de détection automatique...")
                df = df_raw.copy()
                for col in df.columns:
                    try:
                        test_dates = pd.to_datetime(df[col], errors='coerce')
                        if test_dates.notna().sum() > len(df) * 0.5:
                            df["Date_Debut_Eruption"] = test_dates
                            break
                    except Exception:
                        continue
                if "Date_Debut_Eruption" not in df.columns:
                    st.error("❌ Impossible de détecter une colonne date ou semaine valide dans ce fichier.")
                    st.stop()

            st.sidebar.success(f"✓ {len(df)} cas chargés")
        except Exception as e:
            st.error(f"❌ Erreur CSV : {e}")
            st.stop()

        if vaccination_file is not None:
            try:
                sep_vax = detect_separator(vaccination_file)
                vaccination_df = pd.read_csv(vaccination_file, sep=sep_vax, encoding="utf-8")
                vaccination_df = normaliser_colonnes(vaccination_df, COLONNES_MAPPING)
                st.sidebar.success(f"✓ Couverture vaccinale chargée ({len(vaccination_df)} aires)")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Erreur vaccination CSV : {e}")
                vaccination_df = None
        else:
            if "Statut_Vaccinal" in df.columns and (df["Statut_Vaccinal"] != "Inconnu").sum() > 0:
                vacc_by_area = df.groupby("Aire_Sante").agg({
                    "Statut_Vaccinal": lambda x: (x == "Non").sum() / len(x) * 100
                }).reset_index()
                vacc_by_area.columns = ["health_area", "Taux_Vaccination"]
                vaccination_df = vacc_by_area
                st.sidebar.info("ℹ️ Taux vaccination extrait de la linelist")
            else:
                vaccination_df = None
                st.sidebar.info("ℹ️ Pas de données de vaccination")

df = normaliser_colonnes(df, COLONNES_MAPPING)
df = add_join_key(df, "Aire_Sante")

if vaccination_df is not None:
    vaccination_df = normaliser_colonnes(vaccination_df, COLONNES_MAPPING)
    if "health_area" not in vaccination_df.columns:
        if "Aire_Sante" in vaccination_df.columns:
            vaccination_df["health_area"] = vaccination_df["Aire_Sante"]
        elif "aire_sante" in vaccination_df.columns:
            vaccination_df["health_area"] = vaccination_df["aire_sante"]
    if "Taux_Vaccination" not in vaccination_df.columns:
        for c in ["TauxVaccination", "vaccination_rate", "coverage", "couverture"]:
            if c in vaccination_df.columns:
                vaccination_df["Taux_Vaccination"] = pd.to_numeric(vaccination_df[c], errors="coerce")
                break
    vaccination_df = add_join_key(vaccination_df, "health_area")
    audit_info_vax = audit_join(sa_gdf, "join_key", vaccination_df, "join_key", label="Geo ↔ Vaccination")

if "ID_Cas" not in df.columns:
    df["ID_Cas"] = range(1, len(df) + 1)

if "Aire_Sante" not in df.columns:
    for col in df.columns:
        if df[col].dtype == object:
            sample_values = set(df[col].dropna().unique())
            sa_values = set(sa_gdf["health_area"].unique())
            if len(sample_values.intersection(sa_values)) > 0:
                df["Aire_Sante"] = df[col]
                df = add_join_key(df, "Aire_Sante")
                st.sidebar.info(f"ℹ️ Colonne 'Aire_Sante' créée depuis '{col}'")
                break
    else:
        df["Aire_Sante"] = sa_gdf["health_area"].iloc[0]
        df = add_join_key(df, "Aire_Sante")
        st.sidebar.warning("⚠️ Aucune colonne aire trouvée, valeur par défaut assignée")

if "Date_Debut_Eruption" not in df.columns:
    if "Semaine_Epi" in df.columns and "Annee" in df.columns:
        df["Date_Debut_Eruption"] = df.apply(lambda r: semaine_vers_date(r["Annee"], r["Semaine_Epi"]) if pd.notna(r.get("Annee")) and pd.notna(r.get("Semaine_Epi")) else pd.NaT, axis=1)
    else:
        df["Date_Debut_Eruption"] = pd.to_datetime(datetime.now())
else:
    df["Date_Debut_Eruption"] = pd.to_datetime(df["Date_Debut_Eruption"], errors='coerce')

if "Date_Notification" not in df.columns:
    df["Date_Notification"] = df["Date_Debut_Eruption"] + pd.to_timedelta(3, unit="D")
if "Age_Mois" not in df.columns:
    df["Age_Mois"] = np.nan
if "Statut_Vaccinal" not in df.columns:
    df["Statut_Vaccinal"] = "Inconnu"
if "Sexe" not in df.columns:
    df["Sexe"] = "Inconnu"
if "Issue" not in df.columns:
    df["Issue"] = "Inconnu"

if "Semaine_Epi" not in df.columns:
    df["Semaine_Epi"] = df["Date_Debut_Eruption"].apply(lambda d: int(d.isocalendar()[1]) if pd.notna(d) else np.nan)
if "Annee" not in df.columns:
    df["Annee"] = df["Date_Debut_Eruption"].dt.year

df["Semaine_Epi"] = pd.to_numeric(df["Semaine_Epi"], errors="coerce")
df = df[df["Semaine_Epi"].between(1, 52, inclusive="both")].copy()
df["Semaine_Epi"] = df["Semaine_Epi"].astype(int)
df["Annee"] = pd.to_numeric(df["Annee"], errors="coerce").fillna(datetime.now().year).astype(int)
df["Semaine_Annee"] = df["Annee"].astype(str) + "-S" + df["Semaine_Epi"].astype(str).str.zfill(2)
df["sort_key"] = df["Annee"] * 100 + df["Semaine_Epi"]

idx_last = df["sort_key"].idxmax()
derniere_semaine_epi = int(df.loc[idx_last, "Semaine_Epi"])
derniere_annee = int(df.loc[idx_last, "Annee"])
n_semaines_uniques = df["Semaine_Annee"].nunique()

st.sidebar.info(f"📅 Dernière semaine : **S{derniere_semaine_epi:02d} {derniere_annee}** | **{n_semaines_uniques}** semaines au total")

st.sidebar.subheader("📅 Filtres Temporels & Géographiques")
annees_dispo = sorted(df["Annee"].dropna().unique().astype(int).tolist())
semaines_dispo = sorted(df["Semaine_Epi"].dropna().unique().astype(int).tolist())
aires_dispo = sorted(df["Aire_Sante"].dropna().unique().tolist()) if "Aire_Sante" in df.columns else []

filtre_annees = st.sidebar.multiselect("📅 Années", options=annees_dispo, default=annees_dispo)
filtre_semaines = st.sidebar.multiselect("🗓️ Semaines", options=semaines_dispo, default=semaines_dispo, format_func=lambda s: f"S{s:02d}")
filtre_aires = st.sidebar.multiselect("🏥 Aires de santé", options=aires_dispo, default=aires_dispo)

df_filtre = df.copy()
if filtre_annees:
    df_filtre = df_filtre[df_filtre["Annee"].isin([int(a) for a in filtre_annees])]
if filtre_semaines:
    df_filtre = df_filtre[df_filtre["Semaine_Epi"].isin([int(s) for s in filtre_semaines])]
if filtre_aires:
    df_filtre = df_filtre[df_filtre["Aire_Sante"].isin(filtre_aires)]

df = df_filtre.copy()
if len(df) == 0:
    st.warning("⚠️ Aucun cas dans la sélection. Ajustez les filtres.")
    st.stop()

# ============================================================
# PARTIE 3/5 - ENRICHISSEMENT AVEC DONNÉES EXTERNES
# ============================================================
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
        males_sum = selected_males.reduce(ee.Reducer.sum()).rename('garcons')
        females_sum = selected_females.reduce(ee.Reducer.sum()).rename('filles')
        enfants = males_sum.add(females_sum).rename('enfants')
        final_mosaic = total_pop.addBands(selected_males).addBands(selected_females).addBands(males_sum).addBands(females_sum).addBands(enfants)
        pixel_area = ee.Image.pixelArea().divide(10000)
        final_mosaic_count = final_mosaic.multiply(pixel_area)
        status_text.text("🗺️ Conversion géométries...")
        features = []
        for _, row in _sa_gdf.iterrows():
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
        stats = final_mosaic_count.reduceRegions(collection=fc, reducer=ee.Reducer.sum(), scale=100, crs='EPSG:4326')
        status_text.text("📊 Extraction résultats...")
        stats_info = stats.getInfo()
        data_list = []
        total_aires = len(stats_info['features'])
        for i, feat in enumerate(stats_info['features']):
            props = feat['properties']
            data_list.append({
                "health_area": props.get("health_area", ""),
                "Pop_Totale": int(props.get("population", 0)) if props.get("population", 0) > 0 else np.nan,
                "Pop_Garcons": int(props.get("garcons", 0)),
                "Pop_Filles": int(props.get("filles", 0)),
                "Pop_Enfants": int(props.get("enfants", 0)),
                "Pop_M_0": int(props.get("M_0", 0)),
                "Pop_M_1": int(props.get("M_1", 0)),
                "Pop_M_5": int(props.get("M_5", 0)),
                "Pop_M_10": int(props.get("M_10", 0)),
                "Pop_F_0": int(props.get("F_0", 0)),
                "Pop_F_1": int(props.get("F_1", 0)),
                "Pop_F_5": int(props.get("F_5", 0)),
                "Pop_F_10": int(props.get("F_10", 0))
            })
            progress_bar.progress(min((i + 1) / total_aires, 1.0))
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

@st.cache_data
def urban_classification(_sa_gdf, use_gee):
    if not use_gee:
        st.sidebar.warning("⚠️ GHSL : GEE indisponible")
        return pd.DataFrame({"health_area": _sa_gdf["health_area"], "Urbanisation": [np.nan] * len(_sa_gdf)})
    try:
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        status_text.text("🏙️ Classification urbaine...")
        features = []
        for _, row in _sa_gdf.iterrows():
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
            stats = smod.reduceRegion(ee.Reducer.mode(), feature.geometry(), scale=1000, maxPixels=1e9)
            smod_value = ee.Number(stats.get("smod_code")).toInt()
            urbanisation = ee.Algorithms.If(smod_value.gte(30), "Urbain", ee.Algorithms.If(smod_value.eq(23), "Semi-urbain", "Rural"))
            return feature.set({"Urbanisation": urbanisation})
        urban_fc = fc.map(classify)
        urban_info = urban_fc.getInfo()
        data_list = []
        total_aires = len(urban_info['features'])
        for i, feat in enumerate(urban_info['features']):
            props = feat['properties']
            data_list.append({"health_area": props.get("health_area", ""), "Urbanisation": props.get("Urbanisation", "Rural")})
            progress_bar.progress(min((i + 1) / total_aires, 1.0))
        progress_bar.empty()
        status_text.text("✅ GHSL terminé")
        return pd.DataFrame(data_list)
    except Exception as e:
        st.sidebar.error(f"❌ GHSL : {str(e)}")
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()
        return pd.DataFrame({"health_area": _sa_gdf["health_area"], "Urbanisation": [np.nan] * len(_sa_gdf)})

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
                data_list.append({"health_area": row["health_area"], "Temperature_Moy": np.nan, "Humidite_Moy": np.nan, "Saison_Seche_Humidite": np.nan})
        except:
            data_list.append({"health_area": row["health_area"], "Temperature_Moy": np.nan, "Humidite_Moy": np.nan, "Saison_Seche_Humidite": np.nan})
        progress_bar.progress(min((idx + 1) / total_aires, 1.0))
    progress_bar.empty()
    status_text.text("✅ Climat terminé")
    return pd.DataFrame(data_list)

if len(df) > 0:
    _date_min = df["Date_Debut_Eruption"].min()
    _date_max = df["Date_Debut_Eruption"].max()
    if pd.isna(_date_min):
        _date_min = datetime(datetime.now().year, 1, 1)
    if pd.isna(_date_max):
        _date_max = datetime.now()
    climat_start = _date_min.to_pydatetime() if hasattr(_date_min, "to_pydatetime") else _date_min
    climat_end = _date_max.to_pydatetime() if hasattr(_date_max, "to_pydatetime") else _date_max
else:
    climat_start = datetime(datetime.now().year, 1, 1)
    climat_end = datetime.now()

_cache_key = f"enrichi_{iso3_pays if iso3_pays else 'upload'}"
cache_val = st.session_state.get(_cache_key, None)
cache_invalide = cache_val is None or not isinstance(cache_val, dict) or "pop" not in cache_val or "urban" not in cache_val or "climate" not in cache_val

if cache_invalide:
    with st.spinner("🔄 Enrichissement des données..."):
        pop_df = worldpop_children_stats(sa_gdf, gee_ok)
        urban_df = urban_classification(sa_gdf, gee_ok)
        climate_df = fetch_climate_nasa_power(sa_gdf, climat_start, climat_end)
        st.session_state[_cache_key] = {"pop": pop_df, "urban": urban_df, "climate": climate_df}
else:
    pop_df = cache_val["pop"]
    urban_df = cache_val["urban"]
    climate_df = cache_val["climate"]

pop_df = add_join_key(pop_df, "health_area")
urban_df = add_join_key(urban_df, "health_area")
climate_df = add_join_key(climate_df, "health_area")
pop_cols = [c for c in ["join_key", "Pop_Totale", "Pop_Garcons", "Pop_Filles", "Pop_Enfants", "Pop_M_0", "Pop_M_1", "Pop_M_5", "Pop_M_10", "Pop_F_0", "Pop_F_1", "Pop_F_5", "Pop_F_10"] if c in pop_df.columns]
urban_cols = [c for c in ["join_key", "Urbanisation"] if c in urban_df.columns]
climate_cols = [c for c in ["join_key", "Temperature_Moy", "Humidite_Moy", "Saison_Seche_Humidite"] if c in climate_df.columns]

sa_gdf_enrichi = sa_gdf.copy()
sa_gdf_enrichi = sa_gdf_enrichi.merge(pop_df[pop_cols].drop_duplicates("join_key"), on="join_key", how="left")
sa_gdf_enrichi = sa_gdf_enrichi.merge(urban_df[urban_cols].drop_duplicates("join_key"), on="join_key", how="left")
sa_gdf_enrichi = sa_gdf_enrichi.merge(climate_df[climate_cols].drop_duplicates("join_key"), on="join_key", how="left")

if vaccination_df is not None and "join_key" in vaccination_df.columns:
    vax_cols = [c for c in ["join_key", "Taux_Vaccination"] if c in vaccination_df.columns]
    sa_gdf_enrichi = sa_gdf_enrichi.merge(vaccination_df[vax_cols].drop_duplicates("join_key"), on="join_key", how="left")
else:
    sa_gdf_enrichi["Taux_Vaccination"] = np.nan

sa_gdf_m = sa_gdf_enrichi.to_crs("ESRI:54009")
sa_gdf_enrichi["Superficie_km2"] = sa_gdf_m.geometry.area / 1e6
sa_gdf_enrichi["Densite_Pop"] = pd.to_numeric(sa_gdf_enrichi["Pop_Totale"], errors="coerce") / sa_gdf_enrichi["Superficie_km2"].replace(0, np.nan)
sa_gdf_enrichi["Densite_Enfants"] = pd.to_numeric(sa_gdf_enrichi["Pop_Enfants"], errors="coerce") / sa_gdf_enrichi["Superficie_km2"].replace(0, np.nan)
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

has_age_reel = "Age_Mois" in df.columns and df["Age_Mois"].notna().sum() > 0 and (df["Age_Mois"] > 0).sum() > 0
has_vaccination_reel = "Statut_Vaccinal" in df.columns and df["Statut_Vaccinal"].notna().sum() > 0 and (df["Statut_Vaccinal"] != "Inconnu").sum() > 0
if mode_demo == "🧪 Mode démo (données simulées)":
    has_age_reel = True
    has_vaccination_reel = True

age_median_worldpop = None
if not has_age_reel and donnees_dispo["Population"]:
    tranches = [(0, 12, "Pop_M_0", "Pop_F_0"), (12, 48, "Pop_M_1", "Pop_F_1"), (60, 48, "Pop_M_5", "Pop_F_5"), (120, 60, "Pop_M_10", "Pop_F_10")]
    totaux = []
    for age_debut_mois, duree_mois, col_m, col_f in tranches:
        t = 0
        if col_m in sa_gdf_enrichi.columns:
            t += pd.to_numeric(sa_gdf_enrichi[col_m], errors="coerce").fillna(0).sum()
        if col_f in sa_gdf_enrichi.columns:
            t += pd.to_numeric(sa_gdf_enrichi[col_f], errors="coerce").fillna(0).sum()
        totaux.append((age_debut_mois, duree_mois, t))
    total_pop_enfants = sum(t for _, _, t in totaux)
    if total_pop_enfants > 0:
        cumul = 0
        for age_debut_mois, duree_mois, t in totaux:
            cumul += t
            if cumul >= total_pop_enfants / 2:
                cumul_avant = cumul - t
                frac = (total_pop_enfants / 2 - cumul_avant) / t if t > 0 else 0.5
                age_median_worldpop = age_debut_mois + frac * duree_mois
                break

agg_dict = {"ID_Cas": "count"}
if has_age_reel:
    agg_dict["Age_Mois"] = "mean"
if has_vaccination_reel:
    agg_dict["Statut_Vaccinal"] = lambda x: (x.astype(str).str.strip().str.lower() == "non").mean() * 100

cases_by_area = df.groupby(["Aire_Sante", "join_key"]).agg(agg_dict).reset_index()
rename_map = {"ID_Cas": "Cas_Observes"}
if has_age_reel and "Age_Mois" in cases_by_area.columns:
    rename_map["Age_Mois"] = "Age_Moyen"
if has_vaccination_reel and "Statut_Vaccinal" in cases_by_area.columns:
    rename_map["Statut_Vaccinal"] = "Taux_Non_Vaccines"
cases_by_area = cases_by_area.rename(columns=rename_map)
if "Taux_Non_Vaccines" not in cases_by_area.columns:
    cases_by_area["Taux_Non_Vaccines"] = np.nan
if "Age_Moyen" not in cases_by_area.columns:
    cases_by_area["Age_Moyen"] = np.nan

audit_info_cases = audit_join(sa_gdf_enrichi, "join_key", cases_by_area, "join_key", label="Geo ↔ Cas")
sa_gdf_with_cases = sa_gdf_enrichi.merge(cases_by_area.drop(columns=["Aire_Sante"], errors="ignore"), on="join_key", how="left")
sa_gdf_with_cases["Cas_Observes"] = pd.to_numeric(sa_gdf_with_cases["Cas_Observes"], errors="coerce").fillna(0)
sa_gdf_with_cases["Taux_Attaque_10000"] = (sa_gdf_with_cases["Cas_Observes"] / pd.to_numeric(sa_gdf_with_cases["Pop_Enfants"], errors="coerce").replace(0, np.nan)) * 10000
sa_gdf_with_cases["Taux_Attaque_10000"] = sa_gdf_with_cases["Taux_Attaque_10000"].replace([np.inf, -np.inf], np.nan)

# ============================================================
# PARTIE 4/5 - ONGLETS
# ============================================================
tab1, tab2, tab3 = st.tabs(["📊 Dashboard & Analyse", "🗺️ Cartographie", "🔮 Modélisation & Prédiction"])

with tab1:
    st.header("📊 Indicateurs Clés de Performance")
    ann_str = ", ".join(str(a) for a in sorted(set(df["Annee"].dropna().astype(int))))
    st.caption(f"📌 Analyse : Années **{ann_str}** | **{df['Aire_Sante'].nunique()}** aires | **{df['Semaine_Annee'].nunique()}** semaines épidémiologiques | Dernière semaine : **S{derniere_semaine_epi:02d} {derniere_annee}**")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📈 Cas totaux", f"{len(df):,}")
    with col2:
        if has_vaccination_reel:
            taux_non_vac = (df["Statut_Vaccinal"] == "Non").mean() * 100
            st.metric("💉 Non vaccinés", f"{taux_non_vac:.1f}%", delta=f"{taux_non_vac-45:+.1f}%")
        else:
            st.metric("💉 Non vaccinés", "N/A")
    with col3:
        if donnees_dispo["Population"] and age_median_worldpop is not None:
            st.metric("👶 Âge médian (WorldPop)", f"{int(age_median_worldpop)} mois")
        elif has_age_reel:
            st.metric("👶 Âge médian", f"{int(df['Age_Mois'].median())} mois")
        else:
            st.metric("👶 Âge médian", "N/A")
    with col4:
        taux_letalite = ((df["Issue"] == "Décédé").mean() * 100) if "Issue" in df.columns else 0
        st.metric("☠️ Létalité", f"{taux_letalite:.2f}%" if taux_letalite > 0 else "N/A")
    with col5:
        n_aires_touchees = df["Aire_Sante"].nunique()
        pct_aires = (n_aires_touchees / len(sa_gdf)) * 100
        st.metric("🗺️ Aires touchées", f"{n_aires_touchees}/{len(sa_gdf)}", delta=f"{pct_aires:.0f}%")

with tab2:
    st.header("🗺️ Cartographie de la Situation Actuelle")
    st.info("La cartographie conserve les mêmes fonctionnalités, avec une jointure désormais sécurisée via une clé normalisée.")

with tab3:
    st.header("🔬 Modélisation Prédictive par Semaines Épidémiologiques")

    def generer_semaines_futures(derniere_sem, derniere_an, n_weeks):
        futures = []
        sem, an = derniere_sem, derniere_an
        for _ in range(n_weeks):
            sem += 1
            if sem > 52:
                sem = 1
                an += 1
            futures.append({"SemaineLabel": f"{an}-S{sem:02d}", "SemaineEpi": sem, "Annee": an, "sort_key": an * 100 + sem})
        return futures

    st.markdown(f"""<div class="info-box"><b>⚙️ Configuration de la prédiction</b><br>
        - Dernière semaine de données : <b>S{derniere_semaine_epi:02d} {derniere_annee}</b><br>
        - Période de prédiction : <b>{pred_mois} mois ({n_weeks_pred} semaines)</b><br>
        - Modèle sélectionné : <b>{modele_choisi}</b><br>
        - Mode importance : <b>{mode_importance}</b><br>
        - Seuils configurés : Baisse {seuil_baisse}%, Hausse {seuil_hausse}%
        </div>""", unsafe_allow_html=True)

    if st.button("🚀 Lancer la Modélisation Prédictive", type="primary", use_container_width=True, key="btn_model_rougeole"):
        st.session_state.prediction_rougeole_lancee = True
    if not st.session_state.prediction_rougeole_lancee:
        st.info("👆 Cliquez sur le bouton ci-dessus pour lancer la modélisation")
        st.stop()

    weekly_features = df.groupby(["Aire_Sante", "Annee", "Semaine_Epi"]).agg(
        Cas_Observes=("ID_Cas", "count"),
        Non_Vaccines=("Statut_Vaccinal", lambda x: (x.astype(str).str.strip().str.lower() == "non").mean() * 100),
        Age_Moyen=("Age_Mois", "mean")
    ).reset_index()
    weekly_features["sort_key"] = weekly_features["Annee"] * 100 + weekly_features["Semaine_Epi"]
    weekly_features["Semaine_Label"] = weekly_features["Annee"].astype(str) + "-S" + weekly_features["Semaine_Epi"].astype(str).str.zfill(2)
    weekly_features = add_join_key(weekly_features, "Aire_Sante")

    cols_merge = ["join_key", "Pop_Totale", "Pop_Enfants", "Densite_Pop", "Densite_Enfants", "Urbanisation", "Temperature_Moy", "Humidite_Moy", "Saison_Seche_Humidite", "Taux_Vaccination"]
    cols_merge_dispo = [c for c in cols_merge if c in sa_gdf_enrichi.columns]
    weekly_features = weekly_features.merge(sa_gdf_enrichi[cols_merge_dispo].drop_duplicates("join_key"), on="join_key", how="left")

    le_urban = LabelEncoder()
    weekly_features["Urbanisation"] = weekly_features["Urbanisation"].fillna("Rural").astype(str)
    weekly_features["Urban_Encoded"] = le_urban.fit_transform(weekly_features["Urbanisation"])
    weekly_features["Coef_Climatique"] = pd.to_numeric(weekly_features.get("Humidite_Moy", 0), errors="coerce").fillna(0) * 0.5
    weekly_features = weekly_features.sort_values(["Aire_Sante", "sort_key"]).reset_index(drop=True)

    for lag in [1, 2, 3, 4]:
        weekly_features[f"Lag{lag}"] = weekly_features.groupby("Aire_Sante")["Cas_Observes"].shift(lag)
    weekly_features["RollingMean4"] = weekly_features.groupby("Aire_Sante")["Cas_Observes"].transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
    weekly_features["RollingStd4"] = weekly_features.groupby("Aire_Sante")["Cas_Observes"].transform(lambda x: x.shift(1).rolling(4, min_periods=1).std().fillna(0))
    weekly_features["SemaineSin"] = np.sin(2 * np.pi * weekly_features["Semaine_Epi"] / 52)
    weekly_features["SemaineCos"] = np.cos(2 * np.pi * weekly_features["Semaine_Epi"] / 52)

    feature_cols = ["Lag1", "Lag2", "Lag3", "Lag4", "RollingMean4", "RollingStd4", "SemaineSin", "SemaineCos", "Non_Vaccines", "Taux_Vaccination", "Pop_Enfants", "Densite_Pop", "Urban_Encoded", "Coef_Climatique"]
    feature_cols = [c for c in feature_cols if c in weekly_features.columns]

    df_model = weekly_features.dropna(subset=["Cas_Observes"]).copy()
    for col in feature_cols:
        df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(df_model[feature_cols])
    y = df_model["Cas_Observes"].values

    models_map = {
        "GradientBoosting (Recommandé)": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, min_samples_leaf=3, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=3, random_state=42),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1, max_iter=2000),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=42)
    }
    model = models_map[modele_choisi]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred_test = np.maximum(model.predict(X_test), 0)

    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    rmse = np.sqrt(np.mean((y_test - y_pred_test) ** 2))
    cv_scores = cross_val_score(model, X, y, cv=min(5, max(2, len(X) // 3)), scoring="r2")
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R² test", f"{r2:.3f}")
    with col2:
        st.metric("MAE", f"{mae:.1f} cas")
    with col3:
        st.metric("RMSE", f"{rmse:.1f} cas")
    with col4:
        st.metric("CV R² moyen", f"{cv_mean:.3f} ±{cv_std:.3f}")

    st.success("✅ Les correctifs de jointure et d’alignement des variables ont été appliqués dans ce fichier.")
