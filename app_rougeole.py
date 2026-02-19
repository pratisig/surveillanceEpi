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
warnings.filterwarnings('ignore')

# CSS personnalisé
st.markdown("""
<style>
.model-hint { background:#f0f7ff; border-left:4px solid #1976d2; padding:.8rem 1rem; border-radius:4px; margin:.5rem 0; font-size:.9rem; }
.weight-box { background:#fff8e1; border-left:4px solid #f9a825; padding:.8rem 1rem; border-radius:4px; margin:.5rem 0; }
.info-box   { background:#fef9f9; border-left:4px solid #e74c3c; padding:1rem; border-radius:6px; margin:.5rem 0; }
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
    import unicodedata
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

# ── CORRECTION 3 : Filtres temporels dynamiques ───────────────
# (les multiselect sont construits APRÈS chargement des données,
#  voir Partie 2 — ici on initialise juste les valeurs par défaut)

# Paramètres de prédiction
st.sidebar.subheader("🔮 Paramètres de Prédiction")
pred_mois = st.sidebar.slider(
    "Période de prédiction (mois)",
    min_value=1, max_value=12, value=3,
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

model_hints = {
    "GradientBoosting (Recommandé)": "🎯 **Gradient Boosting** : Très performant pour les séries temporelles. Combine plusieurs modèles faibles pour créer un modèle fort. Excellent pour capturer les relations non-linéaires. Recommandé pour la surveillance épidémiologique.",
    "RandomForest": "🌳 **Random Forest** : Ensemble d'arbres de décision. Robuste aux valeurs aberrantes et aux données manquantes. Bon pour les interactions complexes entre variables.",
    "Ridge Regression": "📊 **Ridge Regression** : Régression linéaire avec régularisation L2. Simple et rapide. Idéal pour relations linéaires. Moins performant sur données non-linéaires.",
    "Lasso Regression": "🎯 **Lasso Regression** : Régularisation L1 avec sélection automatique des variables. Utile quand beaucoup de variables peu importantes. Simplifie le modèle.",
    "Decision Tree": "🌲 **Decision Tree** : Arbre de décision unique. Simple à interpréter mais risque de sur-apprentissage. Moins robuste que les méthodes d'ensemble."
}
st.sidebar.markdown(f'<div class="model-hint">{model_hints[modele_choisi]}</div>', unsafe_allow_html=True)

# ── MODE EXPERT — Importance des variables (RESTAURÉ) ─────────
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
            "📈 Historique des cas (lags)", 0, 100, 40, step=5,
            help="Importance des cas passés (4 dernières semaines)"
        )
        poids_manuels["Vaccination"] = st.slider(
            "💉 Couverture vaccinale", 0, 100, 35, step=5,
            help="Importance du taux de vaccination et non-vaccinés"
        )
        poids_manuels["Demographie"] = st.slider(
            "👥 Démographie", 0, 100, 15, step=5,
            help="Importance de la population et densité"
        )
        poids_manuels["Urbanisation"] = st.slider(
            "🏙️ Urbanisation", 0, 100, 8, step=2,
            help="Importance du type d'habitat (urbain/rural)"
        )
        poids_manuels["Climat"] = st.slider(
            "🌡️ Facteurs climatiques", 0, 100, 2, step=1,
            help="Importance de la température, humidité, saison"
        )

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

# ── Seuils d'alerte (RESTAURÉ) ────────────────────────────────
st.sidebar.subheader("⚙️ Seuils d'Alerte")
with st.sidebar.expander("Configurer les seuils", expanded=False):
    seuil_baisse = st.slider(
        "Seuil de baisse significative (%)",
        min_value=10, max_value=90, value=75, step=5,
        help="Afficher les aires avec baisse ≥ X% par rapport à la moyenne"
    )
    seuil_hausse = st.slider(
        "Seuil de hausse significative (%)",
        min_value=10, max_value=200, value=50, step=10,
        help="Afficher les aires avec hausse ≥ X% par rapport à la moyenne"
    )
    seuil_alerte_epidemique = st.number_input(
        "Seuil d'alerte épidémique (cas/semaine)",
        min_value=1, max_value=100, value=5,
        help="Nombre de cas par semaine déclenchant une alerte"
    )

# ── Fonctions chargement géographique (inchangées) ────────────
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
# PARTIE 2/6 - CHARGEMENT AIRES DE SANTÉ ET DONNÉES DE CAS
# ============================================================

# Chargement des aires de santé
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

# ── Données fictives mode démo ────────────────────────────────
@st.cache_data
def generate_dummy_linelists(_sa_gdf, n=500):
    np.random.seed(42)
    aires = _sa_gdf["health_area"].unique()
    rows = []
    for annee in [2024, 2025]:
        for semaine in range(1, 53):
            n_cas = max(0, int(np.random.poisson(
                lam=max(1, 15 * np.sin(semaine * np.pi / 26) + 3))))
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
    df_demo["Date_Debut_Eruption"] = df_demo.apply(
        lambda r: semaine_vers_date(r["Annee"], r["Semaine_Epi"])
                  + timedelta(days=int(np.random.randint(0, 7))), axis=1)
    df_demo["Date_Notification"] = df_demo["Date_Debut_Eruption"] + timedelta(days=3)
    return df_demo

@st.cache_data
def generate_dummy_vaccination(_sa_gdf):
    np.random.seed(42)
    return pd.DataFrame({
        "health_area": _sa_gdf["health_area"],
        "Taux_Vaccination": np.random.beta(a=8, b=2, size=len(_sa_gdf)) * 100
    })

# Chargement des données de cas
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
            # ── Détection séparateur ──────────────────────────
            sep = detect_separator(linelist_file)
            try:
                df_raw = pd.read_csv(linelist_file, sep=sep, encoding="utf-8", low_memory=False)
            except UnicodeDecodeError:
                linelist_file.seek(0)
                df_raw = pd.read_csv(linelist_file, sep=sep, encoding="latin-1", low_memory=False)

            # ── Normalisation des noms de colonnes ────────────
            df_raw = normaliser_colonnes(df_raw, COLONNES_MAPPING)

            # ── Détection colonne cas ─────────────────────────
            cas_col = None
            for _c in ["Cas_Total", "Cas", "cases", "Cases", "nb_cas", "nombre_cas"]:
                if _c in df_raw.columns:
                    cas_col = _c
                    break

            # ── CAS A : données agrégées Semaine_Epi + Cas_Total ──
            if "Semaine_Epi" in df_raw.columns and cas_col:

                # CORRECTION : nettoyage COMPLET avant toute conversion int()
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
                    for _ac in ["health_area","name_fr","aire","zone_sante","district","localite"]:
                        if _ac in df_raw.columns:
                            df_raw["Aire_Sante"] = df_raw[_ac]
                            break
                    else:
                        df_raw["Aire_Sante"] = sa_gdf["health_area"].iloc[0]

                expanded_rows = []
                for _, row in df_raw.iterrows():
                    aire    = row.get("Aire_Sante", "Inconnu")
                    semaine = int(row["Semaine_Epi"])   # ✅ déjà nettoyé
                    annee   = int(row["Annee"])         # ✅ déjà nettoyé

                    # CORRECTION : fillna + coerce avant int() sur cas/décès
                    cas_total = int(max(0, pd.to_numeric(row.get(cas_col, 0), errors="coerce") or 0))
                    deces     = int(max(0, pd.to_numeric(row.get("Deces",     0), errors="coerce") or 0))

                    base_date = semaine_vers_date(annee, semaine)

                    for i in range(cas_total):
                        issue = "Décédé" if i < deces else "Guéri"
                        expanded_rows.append({
                            "ID_Cas":              len(expanded_rows) + 1,
                            "Semaine_Epi":         semaine,
                            "Annee":               annee,
                            "Date_Debut_Eruption": base_date + timedelta(days=int(np.random.randint(0, 7))),
                            "Date_Notification":   base_date + timedelta(days=int(np.random.randint(0, 10))),
                            "Aire_Sante":          aire,
                            "Age_Mois":            np.nan,        # indisponible en agrégé
                            "Statut_Vaccinal":     "Inconnu",
                            "Sexe":                "Inconnu",
                            "Issue":               issue
                        })
                df = pd.DataFrame(expanded_rows)

            # ── CAS B : linelist individuelle avec Date_Debut_Eruption ──
            elif "Date_Debut_Eruption" in df_raw.columns:
                df = df_raw.copy()
                for col in ["Date_Debut_Eruption", "Date_Notification"]:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')

            # ── CAS C : tentative détection automatique date ──
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

        # ── Vaccination ───────────────────────────────────────
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


# ── Normalisation colonnes ─────────────────────────────────────
df = normaliser_colonnes(df, COLONNES_MAPPING)

if "ID_Cas" not in df.columns:
    df["ID_Cas"] = range(1, len(df) + 1)

if "Aire_Sante" not in df.columns:
    for col in df.columns:
        if df[col].dtype == object:
            sample_values = set(df[col].dropna().unique())
            sa_values = set(sa_gdf["health_area"].unique())
            if len(sample_values.intersection(sa_values)) > 0:
                df["Aire_Sante"] = df[col]
                st.sidebar.info(f"ℹ️ Colonne 'Aire_Sante' créée depuis '{col}'")
                break
    else:
        df["Aire_Sante"] = sa_gdf["health_area"].iloc[0]
        st.sidebar.warning("⚠️ Aucune colonne aire trouvée, valeur par défaut assignée")

if "Date_Debut_Eruption" not in df.columns:
    if "Semaine_Epi" in df.columns and "Annee" in df.columns:
        df["Date_Debut_Eruption"] = df.apply(
            lambda r: semaine_vers_date(r["Annee"], r["Semaine_Epi"])
            if pd.notna(r.get("Annee")) and pd.notna(r.get("Semaine_Epi")) else pd.NaT, axis=1)
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

# ── Calcul semaine épidémiologique ─────────────────────────────
if "Semaine_Epi" not in df.columns:
    df["Semaine_Epi"] = df["Date_Debut_Eruption"].apply(
        lambda d: int(d.isocalendar()[1]) if pd.notna(d) else np.nan)
if "Annee" not in df.columns:
    df["Annee"] = df["Date_Debut_Eruption"].dt.year

df["Semaine_Epi"] = pd.to_numeric(df["Semaine_Epi"], errors="coerce")
df = df[df["Semaine_Epi"].between(1, 52, inclusive="both")].copy()
df["Semaine_Epi"] = df["Semaine_Epi"].astype(int)
df["Annee"] = pd.to_numeric(df["Annee"], errors="coerce").fillna(datetime.now().year).astype(int)
df["Semaine_Annee"] = df["Annee"].astype(str) + "-S" + df["Semaine_Epi"].astype(str).str.zfill(2)
df["sort_key"] = df["Annee"] * 100 + df["Semaine_Epi"]

# ── Détection dernière semaine ────────────────────────────────
derniere_semaine_epi = int(df.loc[df["sort_key"].idxmax(), "Semaine_Epi"])
derniere_annee       = int(df.loc[df["sort_key"].idxmax(), "Annee"])
st.sidebar.info(f"📅 Dernière semaine détectée : **S{derniere_semaine_epi:02d} {derniere_annee}**")

# ── CORRECTION 3 : Filtres dynamiques (remplace start_date/end_date) ──
st.sidebar.subheader("📅 Filtres Temporels & Géographiques")

annees_dispo   = sorted(df["Annee"].dropna().unique().astype(int).tolist())
semaines_dispo = sorted(df["Semaine_Epi"].dropna().unique().astype(int).tolist())
aires_dispo    = sorted(df["Aire_Sante"].dropna().unique().tolist()) if "Aire_Sante" in df.columns else []

filtre_annees   = st.sidebar.multiselect("📅 Années", options=annees_dispo, default=annees_dispo)
filtre_semaines = st.sidebar.multiselect("🗓️ Semaines", options=semaines_dispo, default=semaines_dispo,
                                          format_func=lambda s: f"S{s:02d}")
filtre_aires    = st.sidebar.multiselect("🏥 Aires de santé", options=aires_dispo, default=aires_dispo)

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


# ── Enrichissement du GeoDataFrame ────────────────────────────
# CORRECTION : start_date et end_date déduits des semaines filtrées
# (remplace les anciens st.date_input start_date/end_date)
if len(df) > 0:
    _date_min = df["Date_Debut_Eruption"].min()
    _date_max = df["Date_Debut_Eruption"].max()
    # Fallback si dates absentes ou NaT
    if pd.isna(_date_min):
        _date_min = datetime(datetime.now().year, 1, 1)
    if pd.isna(_date_max):
        _date_max = datetime.now()
    climat_start = _date_min.to_pydatetime() if hasattr(_date_min, "to_pydatetime") else _date_min
    climat_end   = _date_max.to_pydatetime() if hasattr(_date_max, "to_pydatetime") else _date_max
else:
    climat_start = datetime(datetime.now().year, 1, 1)
    climat_end   = datetime.now()

with st.spinner("🔄 Enrichissement des données..."):
    pop_df     = worldpop_children_stats(sa_gdf, gee_ok)
    urban_df   = urban_classification(sa_gdf, gee_ok)
    climate_df = fetch_climate_nasa_power(sa_gdf, climat_start, climat_end)

sa_gdf_enrichi = sa_gdf.copy()
sa_gdf_enrichi = sa_gdf_enrichi.merge(pop_df,     on="health_area", how="left")
sa_gdf_enrichi = sa_gdf_enrichi.merge(urban_df,   on="health_area", how="left")
sa_gdf_enrichi = sa_gdf_enrichi.merge(climate_df, on="health_area", how="left")

if vaccination_df is not None:
    sa_gdf_enrichi = sa_gdf_enrichi.merge(vaccination_df, on="health_area", how="left")
else:
    sa_gdf_enrichi["Taux_Vaccination"] = np.nan

# Reprojection en métrique égale-aire pour les superficies
sa_gdf_m = sa_gdf_enrichi.to_crs("ESRI:54009")   # Mollweide (mètres)
sa_gdf_enrichi["Superficie_km2"] = sa_gdf_m.geometry.area / 1e6

sa_gdf_enrichi["Densite_Pop"] = (
    sa_gdf_enrichi["Pop_Totale"] /
    sa_gdf_enrichi["Superficie_km2"].replace(0, np.nan)
)

sa_gdf_enrichi["Densite_Enfants"] = (
    sa_gdf_enrichi["Pop_Enfants"] /
    sa_gdf_enrichi["Superficie_km2"].replace(0, np.nan)
)

sa_gdf_enrichi = sa_gdf_enrichi.replace([np.inf, -np.inf], np.nan)

st.sidebar.success("✓ Enrichissement terminé")

st.sidebar.markdown("---")
st.sidebar.subheader("📋 Données disponibles")

donnees_dispo = {
    "Population":   not sa_gdf_enrichi["Pop_Totale"].isna().all(),
    "Urbanisation": not sa_gdf_enrichi["Urbanisation"].isna().all(),
    "Climat":       not sa_gdf_enrichi["Humidite_Moy"].isna().all(),
    "Vaccination":  not sa_gdf_enrichi["Taux_Vaccination"].isna().all()
}

for nom, dispo in donnees_dispo.items():
    icone = "✅" if dispo else "❌"
    st.sidebar.text(f"{icone} {nom}")


# ============================================================
# CORRECTION 4 : FLAGS DE DISPONIBILITÉ DES DONNÉES RÉELLES
# Ces flags conditionnent les KPI et graphiques (Parties 4 & 5)
# ============================================================

# Age disponible dans les données réelles soumises ?
# (exclut les 0 par défaut du mode CSV agrégé sans colonne âge)
has_age_reel = (
    "Age_Mois" in df.columns
    and df["Age_Mois"].notna().sum() > 0
    and (df["Age_Mois"] > 0).sum() > 0
)

# Vaccination disponible dans les données réelles soumises ?
# (exclut les "Inconnu" par défaut assignés quand colonne absente)
has_vaccination_reel = (
    "Statut_Vaccinal" in df.columns
    and df["Statut_Vaccinal"].notna().sum() > 0
    and (df["Statut_Vaccinal"] != "Inconnu").sum() > 0
)

# En mode démo les deux sont simulés donc disponibles
if mode_demo == "🧪 Mode démo (données simulées)":
    has_age_reel = True
    has_vaccination_reel = True

# ── Âge médian depuis WorldPop si données réelles absentes ────
# CORRECTION 4 : calcul fallback sur pyramide des âges WorldPop
age_median_worldpop = None
if not has_age_reel and donnees_dispo["Population"]:
    # Tranches WorldPop disponibles : 0, 1, 5, 10 (âges de début en années)
    # Durée de chaque tranche : 0→1an, 1→4ans, 5→4ans, 10→5ans
    tranches = [
        (0,  1,  "Pop_M_0",  "Pop_F_0"),   # 0-11 mois → médiane 6 mois
        (12, 48, "Pop_M_1",  "Pop_F_1"),   # 1-4 ans   → médiane 30 mois
        (60, 48, "Pop_M_5",  "Pop_F_5"),   # 5-9 ans   → médiane 84 mois
        (120,60, "Pop_M_10", "Pop_F_10"),  # 10-14 ans → médiane 150 mois
    ]
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
                # Interpolation linéaire dans la tranche
                cumul_avant = cumul - t
                if t > 0:
                    frac = (total_pop_enfants / 2 - cumul_avant) / t
                else:
                    frac = 0.5
                age_median_worldpop = age_debut_mois + frac * duree_mois
                break

# ── Agrégation par aire (utilisant df filtré) ─────────────────
agg_dict = {"ID_Cas": "count"}

if has_age_reel:
    agg_dict["Age_Mois"] = "mean"

if has_vaccination_reel:
    agg_dict["Statut_Vaccinal"] = lambda x: (x == "Non").mean() * 100

cases_by_area = df.groupby("Aire_Sante").agg(agg_dict).reset_index()

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

sa_gdf_with_cases = sa_gdf_enrichi.merge(
    cases_by_area,
    left_on="health_area",
    right_on="Aire_Sante",
    how="left"
)

sa_gdf_with_cases["Cas_Observes"] = sa_gdf_with_cases["Cas_Observes"].fillna(0)

sa_gdf_with_cases["Taux_Attaque_10000"] = (
    sa_gdf_with_cases["Cas_Observes"] /
    sa_gdf_with_cases["Pop_Enfants"].replace(0, np.nan) * 10000
).replace([np.inf, -np.inf], np.nan)
# ============================================================
# PARTIE 4/5 — ONGLETS PRINCIPAUX
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "📊 Dashboard & Analyse",
    "🗺️ Cartographie",
    "🔮 Modélisation & Prédiction"
])

# ============================================================
# TAB 1 — DASHBOARD & ANALYSE
# ============================================================
with tab1:
    st.header("📊 Indicateurs Clés de Performance")

    ann_str = ", ".join(str(a) for a in sorted(set(df["Annee"].dropna().astype(int))))
    st.caption(f"📌 Analyse : Années **{ann_str}** | **{df['Aire_Sante'].nunique()}** aires | **{df['Semaine_Epi'].nunique()}** semaines | Dernière semaine : **S{derniere_semaine_epi:02d} {derniere_annee}**")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("📈 Cas totaux", f"{len(df):,}")

    with col2:
        # CORRECTION 4 : vaccination conditionnelle
        if has_vaccination_reel:
            taux_non_vac = (df["Statut_Vaccinal"] == "Non").mean() * 100
            delta_vac = taux_non_vac - 45
            st.metric("💉 Non vaccinés", f"{taux_non_vac:.1f}%", delta=f"{delta_vac:+.1f}%")
        else:
            st.metric("💉 Non vaccinés", "N/A")

    with col3:
        # CORRECTION 1 : âge médian depuis WorldPop en priorité
        if donnees_dispo["Population"] and age_median_worldpop is not None:
            st.metric("👶 Âge médian (WorldPop)", f"{int(age_median_worldpop)} mois")
        elif has_age_reel:
            st.metric("👶 Âge médian", f"{int(df['Age_Mois'].median())} mois")
        else:
            st.metric("👶 Âge médian", "N/A")

    with col4:
        taux_letalite = ((df["Issue"] == "Décédé").mean() * 100) if "Issue" in df.columns else 0
        if taux_letalite > 0:
            st.metric("☠️ Létalité", f"{taux_letalite:.2f}%")
        else:
            st.metric("☠️ Létalité", "N/A")

    with col5:
        n_aires_touchees = df["Aire_Sante"].nunique()
        pct_aires = (n_aires_touchees / len(sa_gdf)) * 100
        st.metric("🗺️ Aires touchées", f"{n_aires_touchees}/{len(sa_gdf)}", delta=f"{pct_aires:.0f}%")

    # ── Courbe épidémique ─────────────────────────────────────
    st.header("📈 Analyse Temporelle par Semaines Épidémiologiques")

    weekly_cases = df.groupby(["Annee","Semaine_Epi","sort_key"]).size().reset_index(name="Cas")
    weekly_cases["Semaine_Label"] = weekly_cases["Annee"].astype(str) + "-S" + weekly_cases["Semaine_Epi"].astype(str).str.zfill(2)
    weekly_cases = weekly_cases.sort_values("sort_key")

    fig_epi = go.Figure()
    fig_epi.add_trace(go.Scatter(
        x=weekly_cases["Semaine_Label"], y=weekly_cases["Cas"],
        mode="lines+markers", name="Cas observés",
        line=dict(color="#d32f2f", width=3), marker=dict(size=6),
        hovertemplate="<b>%{x}</b><br>Cas : %{y}<extra></extra>"
    ))

    from scipy.signal import savgol_filter
    if len(weekly_cases) > 5:
        wl = min(7, len(weekly_cases) if len(weekly_cases) % 2 == 1 else len(weekly_cases) - 1)
        tendance = savgol_filter(weekly_cases["Cas"], window_length=wl, polyorder=2)
        fig_epi.add_trace(go.Scatter(
            x=weekly_cases["Semaine_Label"], y=tendance,
            mode="lines", name="Tendance",
            line=dict(color="#1976d2", width=2, dash="dash")
        ))

    fig_epi.add_hline(y=seuil_alerte_epidemique, line_dash="dot", line_color="orange",
                      annotation_text=f"Seuil d'alerte ({seuil_alerte_epidemique} cas/sem)",
                      annotation_position="right")
    fig_epi.update_layout(
        title="Courbe épidémique par semaines épidémiologiques",
        xaxis_title="Semaine épidémiologique", yaxis_title="Nombre de cas",
        hovermode="x unified", height=400,
        xaxis=dict(tickangle=-45, nticks=20)
    )
    st.plotly_chart(fig_epi, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        semaine_max = weekly_cases.loc[weekly_cases["Cas"].idxmax()]
        st.metric("🔴 Semaine pic maximal", semaine_max["Semaine_Label"], f"{int(semaine_max['Cas'])} cas")
    with col2:
        st.metric("📊 Moyenne hebdomadaire", f"{weekly_cases['Cas'].mean():.1f} cas")
    with col3:
        if len(weekly_cases) >= 2:
            variation = weekly_cases.iloc[-1]["Cas"] - weekly_cases.iloc[-2]["Cas"]
            cas_prec = weekly_cases.iloc[-2]["Cas"]
            pct_var = (variation / cas_prec * 100) if cas_prec > 0 else 0
            st.metric("📉 Variation dernière semaine", f"{int(variation):+d} cas", f"{pct_var:+.1f}%")
        else:
            st.metric("📉 Variation dernière semaine", "N/A")

    # ── Distribution par âge (CORRECTION 5 : conditionnel) ───
    st.subheader("👶 Distribution par Tranches d'Âge")
    if has_age_reel:
        df["Tranche_Age"] = pd.cut(df["Age_Mois"],
            bins=[0,12,60,120,180], labels=["0-1 an","1-5 ans","5-10 ans","10-15 ans"])
        agg_dict_age = {"ID_Cas": "count"}
        if has_vaccination_reel:
            agg_dict_age["Statut_Vaccinal"] = lambda x: (x == "Non").mean() * 100
        age_stats = df.groupby("Tranche_Age").agg(agg_dict_age).reset_index()
        rename_age = {"ID_Cas": "Nombre_Cas"}
        if has_vaccination_reel and "Statut_Vaccinal" in age_stats.columns:
            rename_age["Statut_Vaccinal"] = "Pct_Non_Vaccines"
        age_stats = age_stats.rename(columns=rename_age)
        col1, col2 = st.columns(2)
        with col1:
            fig_age = px.bar(age_stats, x="Tranche_Age", y="Nombre_Cas",
                title="Cas par tranche d'âge", color="Nombre_Cas",
                color_continuous_scale="Reds", text="Nombre_Cas")
            fig_age.update_traces(textposition="outside")
            st.plotly_chart(fig_age, use_container_width=True)
        with col2:
            if has_vaccination_reel and "Pct_Non_Vaccines" in age_stats.columns and age_stats["Pct_Non_Vaccines"].sum() > 0:
                fig_vacc_age = px.bar(age_stats, x="Tranche_Age", y="Pct_Non_Vaccines",
                    title="% non vaccinés par âge", color="Pct_Non_Vaccines",
                    color_continuous_scale="Oranges", text="Pct_Non_Vaccines")
                fig_vacc_age.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(fig_vacc_age, use_container_width=True)
            else:
                st.info("ℹ️ Données de vaccination par âge non disponibles dans ce fichier")
    else:
        st.info("ℹ️ Données d'âge non disponibles dans ce fichier")

    # ── CORRECTION 6 : Top 10 — 2 onglets ────────────────────
    st.header("🏆 Top 10 des Aires les Plus Touchées")
    top_data = sa_gdf_with_cases[["health_area","Cas_Observes","Taux_Attaque_10000"]].copy()
    top_data = top_data[top_data["Cas_Observes"] > 0]
    has_taux = "Taux_Attaque_10000" in top_data.columns and top_data["Taux_Attaque_10000"].notna().sum() > 0

    if has_taux:
        tab_ta, tab_cas = st.tabs(["📊 Par taux d'attaque (/10 000 hab.)", "📊 Par nombre de cas"])
        with tab_ta:
            top10_ta = top_data.nlargest(10, "Taux_Attaque_10000").sort_values("Taux_Attaque_10000")
            fig_ta = go.Figure(go.Bar(x=top10_ta["Taux_Attaque_10000"], y=top10_ta["health_area"],
                orientation="h", marker_color="#e53935",
                text=top10_ta["Taux_Attaque_10000"].apply(lambda v: f"{v:.1f}" if pd.notna(v) else "N/A"),
                textposition="outside"))
            fig_ta.update_layout(title="Top 10 — Taux d'attaque (/10 000 hab.)",
                xaxis_title="Taux d'attaque", height=400, template="plotly_white")
            st.plotly_chart(fig_ta, use_container_width=True)
        with tab_cas:
            top10_cas = top_data.nlargest(10, "Cas_Observes").sort_values("Cas_Observes")
            fig_cas = go.Figure(go.Bar(x=top10_cas["Cas_Observes"], y=top10_cas["health_area"],
                orientation="h", marker_color="#c62828",
                text=top10_cas["Cas_Observes"].astype(int), textposition="outside"))
            fig_cas.update_layout(title="Top 10 — Nombre de cas",
                xaxis_title="Nombre de cas", height=400, template="plotly_white")
            st.plotly_chart(fig_cas, use_container_width=True)
    else:
        top10_cas = top_data.nlargest(10, "Cas_Observes").sort_values("Cas_Observes")
        fig_cas = go.Figure(go.Bar(x=top10_cas["Cas_Observes"], y=top10_cas["health_area"],
            orientation="h", marker_color="#c62828",
            text=top10_cas["Cas_Observes"].astype(int), textposition="outside"))
        fig_cas.update_layout(title="Top 10 — Nombre de cas", height=400, template="plotly_white")
        st.plotly_chart(fig_cas, use_container_width=True)

    # ── CORRECTION 4 : Pyramide des âges WorldPop (RESTAURÉE) ─
    st.header("📊 Pyramide des Âges - Population Enfantine")
    if donnees_dispo["Population"]:
        pyramid_data = []
        for idx, row in sa_gdf_enrichi.iterrows():
            aire = row['health_area']
            pop_0_1_m   = row.get('Pop_M_0', 0) + row.get('Pop_M_1', 0)
            pop_5_9_m   = row.get('Pop_M_5', 0)
            pop_10_14_m = row.get('Pop_M_10', 0)
            pop_0_1_f   = row.get('Pop_F_0', 0) + row.get('Pop_F_1', 0)
            pop_5_9_f   = row.get('Pop_F_5', 0)
            pop_10_14_f = row.get('Pop_F_10', 0)
            pyramid_data.append({
                'Aire': aire,
                'Garçons_0-4': pop_0_1_m,  'Garçons_5-9': pop_5_9_m,   'Garçons_10-14': pop_10_14_m,
                'Filles_0-4':  pop_0_1_f,  'Filles_5-9':  pop_5_9_f,   'Filles_10-14':  pop_10_14_f
            })
        pyramid_df = pd.DataFrame(pyramid_data)
        tG04 = pyramid_df['Garçons_0-4'].sum();  tG59 = pyramid_df['Garçons_5-9'].sum();  tG1014 = pyramid_df['Garçons_10-14'].sum()
        tF04 = pyramid_df['Filles_0-4'].sum();   tF59 = pyramid_df['Filles_5-9'].sum();   tF1014 = pyramid_df['Filles_10-14'].sum()
        pyramid_plot_df = pd.DataFrame({
            'Age': ['0-4','5-9','10-14'],
            'Garçons': [-tG04, -tG59, -tG1014],
            'Filles':  [ tF04,  tF59,  tF1014]
        })
        fig_pyr = go.Figure()
        fig_pyr.add_trace(go.Bar(y=pyramid_plot_df['Age'], x=pyramid_plot_df['Garçons'],
            name='Garçons', orientation='h', marker=dict(color='#42a5f5'),
            text=[f"{abs(x):,.0f}" for x in pyramid_plot_df['Garçons']], textposition='inside',
            hovertemplate='<b>%{y} ans</b><br>Garçons: %{text}<extra></extra>'))
        fig_pyr.add_trace(go.Bar(y=pyramid_plot_df['Age'], x=pyramid_plot_df['Filles'],
            name='Filles', orientation='h', marker=dict(color='#ec407a'),
            text=[f"{x:,.0f}" for x in pyramid_plot_df['Filles']], textposition='inside',
            hovertemplate='<b>%{y} ans</b><br>Filles: %{text}<extra></extra>'))
        max_val = max(abs(pyramid_plot_df['Garçons'].min()), pyramid_plot_df['Filles'].max())
        fig_pyr.update_layout(
            title='Pyramide des Âges - Population Enfantine (0-14 ans)',
            xaxis=dict(title='Population',
                tickvals=[-max_val,-max_val/2,0,max_val/2,max_val],
                ticktext=[f"{int(max_val):,}",f"{int(max_val/2):,}","0",f"{int(max_val/2):,}",f"{int(max_val):,}"],
                range=[-max_val*1.1, max_val*1.1]),
            yaxis=dict(title="Tranche d'âge"),
            barmode='overlay', height=400, bargap=0.1,
            showlegend=True, legend=dict(x=0.85,y=0.95), hovermode='y unified'
        )
        st.plotly_chart(fig_pyr, use_container_width=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            total_garcons = tG04 + tG59 + tG1014
            st.metric("👦 Garçons (0-14 ans)", f"{int(total_garcons):,}")
        with col2:
            total_filles = tF04 + tF59 + tF1014
            st.metric("👧 Filles (0-14 ans)", f"{int(total_filles):,}")
        with col3:
            ratio = (total_garcons / total_filles * 100) if total_filles > 0 else 0
            st.metric("⚖️ Ratio G/F", f"{ratio:.1f}%")
    else:
        st.info("📊 Données de population non disponibles. Pyramide des âges non affichable.")

    # ── Nowcasting - Correction des Délais de Notification ────────
st.subheader("⏱️ Nowcasting - Correction des Délais de Notification")
st.info("**Nowcasting :** Technique d'ajustement permettant d'estimer le nombre réel "
        "de cas en tenant compte des délais de notification.")

if "Date_Notification" in df.columns and "Date_Debut_Eruption" in df.columns:
    # CORRECTION : forcer numérique avant tout calcul
    delai_raw = (
        pd.to_datetime(df["Date_Notification"], errors="coerce") -
        pd.to_datetime(df["Date_Debut_Eruption"], errors="coerce")
    ).dt.days
    df["Delai_Notification"] = pd.to_numeric(delai_raw, errors="coerce")
    delai_available = df["Delai_Notification"].notna().sum() > 0
else:
    df["Delai_Notification"] = 3
    delai_available = False

# CORRECTION : conversion explicite en float natif Python (pas numpy)
# pour éviter TypeError dans add_vline (Plotly n'accepte pas numpy float)
_delai_series = pd.to_numeric(df["Delai_Notification"], errors="coerce").dropna()

delai_moyen  = float(_delai_series.mean())   if len(_delai_series) > 0 else float('nan')
delai_median = float(_delai_series.median()) if len(_delai_series) > 0 else float('nan')
delai_std    = float(_delai_series.std())    if len(_delai_series) > 0 else float('nan')

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Délai moyen",
              f"{delai_moyen:.1f} jours" if delai_available and not np.isnan(delai_moyen) else "N/A")
with col2:
    st.metric("Délai médian",
              f"{delai_median:.0f} jours" if delai_available and not np.isnan(delai_median) else "N/A")
with col3:
    st.metric("Écart-type",
              f"{delai_std:.1f} jours" if delai_available and not np.isnan(delai_std) else "N/A")
with col4:
    derniere_semaine_label = weekly_cases.iloc[-1]['Semaine_Label']
    cas_derniere_semaine   = int(weekly_cases.iloc[-1]['Cas'])
    if delai_available and not np.isnan(delai_moyen):
        facteur_correction = 1 + (delai_moyen / 7)
        cas_corriges = int(cas_derniere_semaine * facteur_correction)
        st.metric(f"Cas corrigés ({derniere_semaine_label})",
                  cas_corriges, delta=f"+{cas_corriges - cas_derniere_semaine}")
    else:
        st.metric(f"Cas corrigés ({derniere_semaine_label})",
                  cas_derniere_semaine, delta="N/A")

if delai_available and not np.isnan(delai_moyen):
    # Filtrer les délais valides et raisonnables pour l'histogramme
    df_delai_plot = df[
        df["Delai_Notification"].notna() &
        df["Delai_Notification"].between(-5, 60)
    ].copy()

    if len(df_delai_plot) > 0:
        fig_delai = px.histogram(
            df_delai_plot,
            x="Delai_Notification",
            nbins=20,
            title="Distribution des délais de notification",
            labels={"Delai_Notification": "Délai (jours)", "count": "Nombre de cas"},
            color_discrete_sequence=['#d32f2f']
        )

        # CORRECTION : add_vline nécessite float Python pur (pas np.float64)
        # et l'axe X doit être numérique (histogramme → OK ici)
        if not np.isnan(delai_moyen):
            fig_delai.add_vline(
                x=float(delai_moyen),
                line_dash="dash", line_color="blue",
                annotation_text=f"Moyenne : {delai_moyen:.1f}j",
                annotation_position="top right"
            )
        if not np.isnan(delai_median):
            fig_delai.add_vline(
                x=float(delai_median),
                line_dash="dash", line_color="green",
                annotation_text=f"Médiane : {delai_median:.0f}j",
                annotation_position="top left"
            )

        fig_delai.update_layout(
            xaxis_title="Délai (jours)",
            yaxis_title="Nombre de cas",
            template="plotly_white",
            height=350
        )
        st.plotly_chart(fig_delai, use_container_width=True)
    else:
        st.info("ℹ️ Délais de notification non représentables (valeurs hors plage)")
else:
    st.info("ℹ️ Données de délai de notification non disponibles")


# ============================================================
# CARTOGRAPHIE DE LA SITUATION ACTUELLE
# ============================================================
with tab2:
st.header("🗺️ Cartographie de la Situation Actuelle")

# ── Sécurisation des types avant affichage ────────────────────
def safe_float(val):
    """Retourne float ou np.nan — jamais None ni str."""
    try:
        f = float(val)
        return np.nan if np.isinf(f) else f
    except (TypeError, ValueError):
        return np.nan

def safe_int(val, default=0):
    """Retourne int ou default — jamais NaN."""
    try:
        f = float(val)
        return default if np.isnan(f) or np.isinf(f) else int(f)
    except (TypeError, ValueError):
        return default

def fmt_val(val, fmt="{:.1f}", suffix="", fallback="N/A"):
    """Formate une valeur numérique ou retourne fallback."""
    f = safe_float(val)
    if np.isnan(f):
        return fallback
    return fmt.format(f) + suffix

# ── Centroïde carte ───────────────────────────────────────────
try:
    center_lat = float(sa_gdf_with_cases.geometry.centroid.y.mean())
    center_lon = float(sa_gdf_with_cases.geometry.centroid.x.mean())
    if np.isnan(center_lat) or np.isnan(center_lon):
        center_lat, center_lon = 15.0, 2.0  # fallback Afrique de l'Ouest
except Exception:
    center_lat, center_lon = 15.0, 2.0

# ── Carte Folium ──────────────────────────────────────────────
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=6,
    tiles="CartoDB positron",
    control_scale=True
)

import branca.colormap as cm

max_cases = safe_float(sa_gdf_with_cases["Cas_Observes"].max())
max_cases = 1 if np.isnan(max_cases) or max_cases == 0 else max_cases

colormap = cm.LinearColormap(
    colors=['#e8f5e9', '#81c784', '#ffeb3b', '#ff9800', '#f44336', '#b71c1c'],
    vmin=0,
    vmax=max_cases,
    caption="Nombre de cas observés"
)
colormap.add_to(m)

# ── Ajout des polygones avec popups enrichis ──────────────────
for _, row in sa_gdf_with_cases.iterrows():

    # Sécurisation de toutes les valeurs
    aire_name    = str(row.get('health_area', 'N/A'))
    cas_obs      = safe_int(row.get('Cas_Observes', 0))
    pop_enfants  = safe_float(row.get('Pop_Enfants',       np.nan))
    pop_totale   = safe_float(row.get('Pop_Totale',        np.nan))
    taux_attaque = safe_float(row.get('Taux_Attaque_10000',np.nan))
    urbanisation = row.get('Urbanisation', 'N/A')
    urbanisation = str(urbanisation) if pd.notna(urbanisation) else 'N/A'
    densite      = safe_float(row.get('Densite_Pop',       np.nan))
    taux_vacc    = safe_float(row.get('Taux_Vaccination',  np.nan))
    temp_moy     = safe_float(row.get('Temperature_Moy',   np.nan))
    hum_moy      = safe_float(row.get('Humidite_Moy',      np.nan))

    # Couleur choroplèthe
    fill_color  = colormap(cas_obs) if cas_obs <= max_cases else '#b71c1c'
    line_color  = '#b71c1c' if cas_obs >= seuil_alerte_epidemique else '#555555'
    line_weight = 2.5 if cas_obs >= seuil_alerte_epidemique else 0.5

    # Badge variation
    badge_alerte = ""
    if cas_obs >= seuil_alerte_epidemique:
        badge_alerte = f'<span style="background:#d32f2f;color:white;padding:2px 8px;border-radius:10px;font-size:11px;">⚠️ ALERTE</span>'

    popup_html = f"""
    <div style="font-family:Arial; font-size:13px; width:360px; line-height:1.5;">
      <div style="background:#1976d2;color:white;padding:10px 14px;border-radius:6px 6px 0 0;margin:-10px -10px 10px -10px;">
        <b style="font-size:15px;">{aire_name}</b><br>
        {badge_alerte}
      </div>

      <b style="color:#d32f2f;">📊 Situation Épidémiologique</b>
      <table style="width:100%;border-collapse:collapse;margin:6px 0;">
        <tr style="background:#ffeaea;">
          <td style="padding:5px 8px;"><b>Cas observés</b></td>
          <td style="padding:5px 8px;text-align:right;">
            <b style="font-size:18px;color:#d32f2f;">{cas_obs}</b>
          </td>
        </tr>
        <tr>
          <td style="padding:5px 8px;">Taux d'attaque</td>
          <td style="padding:5px 8px;text-align:right;">{fmt_val(taux_attaque, "{:.1f}", " /10 000 enf.")}</td>
        </tr>
      </table>

      <b style="color:#1565c0;">👥 Population</b>
      <table style="width:100%;border-collapse:collapse;margin:6px 0;">
        <tr style="background:#e3f2fd;">
          <td style="padding:5px 8px;">Pop. totale</td>
          <td style="padding:5px 8px;text-align:right;">{fmt_val(pop_totale, "{:,.0f}", "")}</td>
        </tr>
        <tr>
          <td style="padding:5px 8px;">Enfants (0-14 ans)</td>
          <td style="padding:5px 8px;text-align:right;">{fmt_val(pop_enfants, "{:,.0f}", "")}</td>
        </tr>
        <tr style="background:#e3f2fd;">
          <td style="padding:5px 8px;">Densité pop.</td>
          <td style="padding:5px 8px;text-align:right;">{fmt_val(densite, "{:.1f}", " hab/km²")}</td>
        </tr>
        <tr>
          <td style="padding:5px 8px;">Type habitat</td>
          <td style="padding:5px 8px;text-align:right;"><b>{urbanisation}</b></td>
        </tr>
      </table>

      <b style="color:#2e7d32;">💉 Vaccination & Climat</b>
      <table style="width:100%;border-collapse:collapse;margin:6px 0;">
        <tr style="background:#e8f5e9;">
          <td style="padding:5px 8px;">Taux vaccination</td>
          <td style="padding:5px 8px;text-align:right;">{fmt_val(taux_vacc, "{:.1f}", "%")}</td>
        </tr>
        <tr>
          <td style="padding:5px 8px;">Température moy.</td>
          <td style="padding:5px 8px;text-align:right;">{fmt_val(temp_moy, "{:.1f}", " °C")}</td>
        </tr>
        <tr style="background:#e8f5e9;">
          <td style="padding:5px 8px;">Humidité moy.</td>
          <td style="padding:5px 8px;text-align:right;">{fmt_val(hum_moy, "{:.1f}", "%")}</td>
        </tr>
      </table>
    </div>
    """

    # Géométrie — sécurisée
    try:
        geom = row['geometry']
        if geom is None or geom.is_empty:
            continue

        folium.GeoJson(
            geom.__geo_interface__,
            style_function=lambda x,
                c=fill_color, w=line_weight, bc=line_color: {
                'fillColor':   c,
                'color':       bc,
                'weight':      w,
                'fillOpacity': 0.7,
                'opacity':     0.9
            },
            tooltip=folium.Tooltip(
                f"<b>{aire_name}</b><br>{cas_obs} cas",
                sticky=True
            ),
            popup=folium.Popup(popup_html, max_width=420)
        ).add_to(m)

    except Exception:
        continue  # Passer les géométries invalides sans planter

# ── Heatmap géographique ──────────────────────────────────────
heat_data = [
    [float(row.geometry.centroid.y),
     float(row.geometry.centroid.x),
     float(row['Cas_Observes'])]
    for _, row in sa_gdf_with_cases.iterrows()
    if safe_int(row.get('Cas_Observes', 0)) > 0
    and row.geometry is not None
]

if heat_data:
    HeatMap(
        heat_data,
        radius=20, blur=25, max_zoom=13,
        gradient={0.0: 'blue', 0.4: 'lime', 0.7: 'yellow', 1.0: 'red'}
    ).add_to(m)

# ── Légende HTML ──────────────────────────────────────────────
mc3 = max_cases // 3 if max_cases > 3 else 1
legend_html = f"""
<div style="position:fixed;bottom:50px;left:50px;width:240px;background:white;
     border:2px solid grey;z-index:9999;font-size:13px;padding:12px;border-radius:6px;
     box-shadow:2px 2px 6px rgba(0,0,0,0.3);">
  <p style="margin:0 0 8px;font-weight:bold;font-size:14px;">📊 Légende</p>
  <p style="margin:4px 0;">
    <span style="background:#e8f5e9;padding:2px 10px;border:1px solid #ccc;">Faible</span>
    &nbsp;0 – {mc3:.0f} cas</p>
  <p style="margin:4px 0;">
    <span style="background:#ffeb3b;padding:2px 10px;border:1px solid #ccc;">Moyen</span>
    &nbsp;{mc3:.0f} – {2*mc3:.0f} cas</p>
  <p style="margin:4px 0;">
    <span style="background:#f44336;color:white;padding:2px 10px;">Élevé</span>
    &nbsp;&gt; {2*mc3:.0f} cas</p>
  <hr style="margin:8px 0;">
  <p style="margin:4px 0;color:#d32f2f;">
    <b>⚠️ Seuil alerte :</b> {seuil_alerte_epidemique} cas/sem
  </p>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# ── Affichage — KEY OBLIGATOIRE pour éviter la page blanche ──
st_folium(m, width=1400, height=650, key="carte_situation_actuelle_rougeole")

# ── Métriques sous la carte ───────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    aires_alerte = len(sa_gdf_with_cases[
        sa_gdf_with_cases['Cas_Observes'] >= seuil_alerte_epidemique])
    st.metric("🚨 Aires en alerte", aires_alerte,
              f"{aires_alerte/len(sa_gdf)*100:.1f}%")
with col2:
    aires_sans_cas = len(sa_gdf_with_cases[
        sa_gdf_with_cases['Cas_Observes'] == 0])
    st.metric("✅ Aires sans cas", aires_sans_cas,
              f"{aires_sans_cas/len(sa_gdf)*100:.1f}%")
with col3:
    densite_pop_moy = safe_float(sa_gdf_with_cases['Densite_Pop'].mean())
    st.metric("📍 Densité pop. moy.",
              fmt_val(densite_pop_moy, "{:.1f}", " hab/km²"))

# ============================================================
# TAB 3 — MODÉLISATION PRÉDICTIVE (RESTAURÉE + CORRECTIONS)
# ============================================================
with tab3:
    st.header("🔮 Modélisation Prédictive par Semaines Épidémiologiques")

    # CORRECTION 7 : génération semaines futures STRICTEMENT après dernière semaine observée
    def generer_semaines_futures(derniere_sem, derniere_an, n_weeks):
        futures = []
        sem, an = derniere_sem, derniere_an
        for _ in range(n_weeks):
            sem += 1
            if sem > 52:
                sem = 1
                an += 1
            futures.append({
                "Semaine_Label": f"{an}-S{sem:02d}",
                "Semaine_Epi":   sem,
                "Annee":         an,
                "sort_key":      an * 100 + sem
            })
        return futures

    st.markdown(f"""
    <div class="info-box">
    <b>Configuration de la prédiction :</b><br>
    - Dernière semaine de données : <b>S{derniere_semaine_epi:02d} ({derniere_annee})</b><br>
    - Période de prédiction : <b>{pred_mois} mois ({n_weeks_pred} semaines)</b><br>
    - Modèle sélectionné : <b>{modele_choisi}</b><br>
    - Mode importance : <b>{mode_importance}</b><br>
    - Seuils configurés : Baisse ≥{seuil_baisse}%, Hausse ≥{seuil_hausse}%
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🚀 Lancer la Modélisation Prédictive", type="primary",
                     use_container_width=True, key="btn_model_rougeole"):
            st.session_state.prediction_rougeole_lancee = True
    with col2:
        if st.button("🔄 Réinitialiser", use_container_width=True, key="btn_reset_rougeole"):
            st.session_state.prediction_rougeole_lancee = False

    if not st.session_state.prediction_rougeole_lancee:
        st.info("👆 Cliquez sur le bouton ci-dessus pour lancer la modélisation")
        st.stop()

    with st.spinner("🤖 Préparation des données et entraînement..."):

        weekly_features = df.groupby(["Aire_Sante","Annee","Semaine_Epi"]).agg(
            Cas_Observes=("ID_Cas","count"),
            Non_Vaccines=("Statut_Vaccinal", lambda x: (x == "Non").mean() * 100),
            Age_Moyen=("Age_Mois","mean")
        ).reset_index()

        weekly_features['Semaine_Label'] = (
            weekly_features['Annee'].astype(str) + '-S' +
            weekly_features['Semaine_Epi'].astype(str).str.zfill(2)
        )
        weekly_features['sort_key'] = weekly_features['Annee'] * 100 + weekly_features['Semaine_Epi']

        weekly_features = weekly_features.merge(
            sa_gdf_enrichi[[
                "health_area","Pop_Totale","Pop_Enfants",
                "Densite_Pop","Densite_Enfants","Urbanisation",
                "Temperature_Moy","Humidite_Moy","Saison_Seche_Humidite",
                "Taux_Vaccination"
            ]],
            left_on="Aire_Sante", right_on="health_area", how="left"
        )

        weekly_features['Age_Moyen'] = weekly_features['Age_Moyen'].fillna(
            weekly_features['Age_Moyen'].median() if weekly_features['Age_Moyen'].notna().any() else 0)
        weekly_features['Non_Vaccines'] = weekly_features['Non_Vaccines'].fillna(
            weekly_features['Non_Vaccines'].mean() if weekly_features['Non_Vaccines'].notna().any() else 50.0)

        le_urban = LabelEncoder()
        weekly_features["Urban_Encoded"] = le_urban.fit_transform(
            weekly_features["Urbanisation"].fillna("Rural"))

        if donnees_dispo["Climat"]:
            scaler_climat = MinMaxScaler()
            climate_cols = ["Temperature_Moy","Humidite_Moy","Saison_Seche_Humidite"]
            for col in climate_cols:
                if col in weekly_features.columns:
                    col_mean = weekly_features[col].mean()
                    weekly_features[col] = weekly_features[col].fillna(col_mean if not pd.isna(col_mean) else 0)
            climate_scaled = scaler_climat.fit_transform(weekly_features[climate_cols].values)
            for idx_c, col in enumerate(climate_cols):
                weekly_features[f"{col}_Norm"] = climate_scaled[:, idx_c]
            weekly_features["Coef_Climatique"] = (
                weekly_features.get("Temperature_Moy_Norm", 0) * 0.4 +
                weekly_features.get("Humidite_Moy_Norm", 0) * 0.4 +
                weekly_features.get("Saison_Seche_Humidite_Norm", 0) * 0.2
            )

        weekly_features = weekly_features.sort_values(['Aire_Sante','Annee','Semaine_Epi'])
        for lag in [1,2,3,4]:
            weekly_features[f'Cas_Lag_{lag}'] = (
                weekly_features.groupby('Aire_Sante')['Cas_Observes'].shift(lag))

        numeric_cols = weekly_features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            weekly_features[col] = weekly_features[col].replace([np.inf,-np.inf], np.nan)
            col_mean = weekly_features[col].mean()
            weekly_features[col] = weekly_features[col].fillna(col_mean if not pd.isna(col_mean) else 0)

        st.subheader("📚 Entraînement du Modèle")

        feature_cols = ["Cas_Observes","Age_Moyen","Semaine_Epi",
                        "Cas_Lag_1","Cas_Lag_2","Cas_Lag_3","Cas_Lag_4"]

        feature_groups = {
            "Historique_Cas": ["Cas_Lag_1","Cas_Lag_2","Cas_Lag_3","Cas_Lag_4"],
            "Vaccination": [], "Demographie": [], "Urbanisation": [], "Climat": []
        }

        if donnees_dispo["Population"]:
            feature_cols.extend(["Pop_Totale","Pop_Enfants","Densite_Pop","Densite_Enfants"])
            feature_groups["Demographie"] = ["Pop_Totale","Pop_Enfants","Densite_Pop","Densite_Enfants"]
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
            feature_cols.extend(["Taux_Vaccination","Non_Vaccines"])
            feature_groups["Vaccination"] = ["Taux_Vaccination","Non_Vaccines"]
            st.info("✅ Données vaccinales intégrées au modèle")
        elif "Non_Vaccines" in weekly_features.columns:
            feature_cols.append("Non_Vaccines")
            feature_groups["Vaccination"] = ["Non_Vaccines"]

        st.markdown(f"**Variables utilisées :** {len(feature_cols)} features")

        for col in feature_cols:
            weekly_features[col] = weekly_features[col].fillna(0)

        weekly_features_clean = weekly_features.dropna(subset=feature_cols)
        if len(weekly_features_clean) < 20:
            st.warning("⚠️ Données insuffisantes (minimum 20 observations requises)")
            st.stop()

        X = weekly_features_clean[feature_cols].copy().fillna(0)
        y = weekly_features_clean["Cas_Observes"].copy().fillna(0)

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
                "Poids": [f"{v*100:.1f}%" for v in column_weights.values()]
            })
            st.dataframe(weights_df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            X_to_fit = X_weighted
        else:
            X_to_fit = X

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_to_fit)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Choix du modèle
        if "GradientBoosting" in modele_choisi:
            model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        elif "RandomForest" in modele_choisi:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif "Ridge" in modele_choisi:
            model = Ridge(alpha=1.0)
        elif "Lasso" in modele_choisi:
            model = Lasso(alpha=1.0)
        else:
            model = DecisionTreeRegressor(max_depth=5, random_state=42)

        model.fit(X_scaled, y)

        # Cross-validation
        try:
            cv_scores = cross_val_score(model, X_scaled, y, cv=min(5, len(X_scaled)//4), scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std  = cv_scores.std()
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("R² (CV)", f"{cv_mean:.3f}")
            with col2: st.metric("Écart-type CV", f"{cv_std:.3f}")
            with col3: st.metric("Observations", f"{len(X_scaled):,}")
        except Exception:
            cv_std = 0.1

        # Importance des variables (mode automatique)
        if mode_importance == "🤖 Automatique (ML)" and hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'Variable': feature_cols,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            fig_imp = px.bar(importance_df, x='Importance', y='Variable', orientation='h',
                title='Importance des variables (ML)', color='Importance',
                color_continuous_scale='Blues')
            st.plotly_chart(fig_imp, use_container_width=True)

        # ── CORRECTION 7 : Prédictions futures STRICTEMENT après dernière semaine ──
        semaines_futures = generer_semaines_futures(derniere_semaine_epi, derniere_annee, n_weeks_pred)
        aires_to_predict = weekly_features_clean["Aire_Sante"].unique()

        future_predictions = []
        progress_bar = st.progress(0)
        status_text  = st.empty()

        for i_aire, aire in enumerate(aires_to_predict):
            progress_bar.progress((i_aire + 1) / len(aires_to_predict))
            status_text.text(f"Prédiction : {aire} ({i_aire+1}/{len(aires_to_predict)})")

            aire_data = weekly_features_clean[weekly_features_clean["Aire_Sante"] == aire].sort_values("sort_key")
            if len(aire_data) == 0:
                continue

            last_obs       = aire_data.iloc[-1]
            last_4_weeks   = aire_data['Cas_Observes'].values[-4:].tolist()
            while len(last_4_weeks) < 4:
                last_4_weeks = [last_4_weeks[0]] + last_4_weeks

            aire_info = sa_gdf_enrichi[sa_gdf_enrichi["health_area"] == aire]
            aire_static = {}
            if len(aire_info) > 0:
                r = aire_info.iloc[0]
                for col in feature_cols:
                    if col in r.index:
                        aire_static[col] = r[col] if pd.notna(r[col]) else 0

            for j_week, sem_info in enumerate(semaines_futures):
                future_row = dict(last_obs)
                future_row["Aire_Sante"]   = aire
                future_row["Semaine_Epi"]  = sem_info["Semaine_Epi"]
                future_row["Annee"]        = sem_info["Annee"]
                future_row["Semaine_Label"]= sem_info["Semaine_Label"]
                future_row["sort_key"]     = sem_info["sort_key"]

                # Lags depuis prédictions précédentes
                prev_preds = [p["Predicted_Cases"] for p in future_predictions if p["Aire_Sante"] == aire]

                if j_week == 0:
                    future_row["Cas_Lag_1"] = last_4_weeks[-1]
                    future_row["Cas_Lag_2"] = last_4_weeks[-2] if len(last_4_weeks) >= 2 else last_4_weeks[-1]
                    future_row["Cas_Lag_3"] = last_4_weeks[-3] if len(last_4_weeks) >= 3 else last_4_weeks[-1]
                    future_row["Cas_Lag_4"] = last_4_weeks[-4] if len(last_4_weeks) >= 4 else last_4_weeks[-1]
                else:
                    future_row["Cas_Observes"] = prev_preds[-1] if prev_preds else last_obs["Cas_Observes"]
                    future_row["Cas_Lag_1"] = prev_preds[-1] if len(prev_preds) >= 1 else last_4_weeks[-1]
                    future_row["Cas_Lag_2"] = prev_preds[-2] if len(prev_preds) >= 2 else last_4_weeks[-2]
                    future_row["Cas_Lag_3"] = prev_preds[-3] if len(prev_preds) >= 3 else last_4_weeks[-3]
                    future_row["Cas_Lag_4"] = prev_preds[-4] if len(prev_preds) >= 4 else last_4_weeks[-4]

                X_future_values = []
                for col in feature_cols:
                    val = future_row.get(col, aire_static.get(col, 0))
                    if pd.isna(val):
                        val = 0
                    X_future_values.append(float(val))

                X_future = np.array([X_future_values])
                if mode_importance == "👨‍⚕️ Manuel (Expert)":
                    for idx_c, col in enumerate(feature_cols):
                        if col in column_weights:
                            X_future[0, idx_c] *= column_weights[col]

                X_future = np.nan_to_num(X_future, nan=0.0)
                X_future_scaled = scaler.transform(X_future)
                X_future_scaled = np.nan_to_num(X_future_scaled, nan=0.0)

                predicted_cases = max(0, model.predict(X_future_scaled)[0])
                if cv_std > 0:
                    noise = np.random.normal(0, predicted_cases * cv_std * 0.1)
                    predicted_cases = max(0, predicted_cases + noise)

                future_row["Predicted_Cases"] = predicted_cases
                future_predictions.append(future_row)

        progress_bar.empty()
        status_text.empty()

        future_df = pd.DataFrame(future_predictions)
        future_df['Predicted_Cases'] = future_df['Predicted_Cases'].round(0).astype(int)

        st.success(f"✓ {len(future_df)} prédictions générées ({len(aires_to_predict)} aires × {n_weeks_pred} semaines)")

        moyenne_historique = weekly_features.groupby("Aire_Sante")["Cas_Observes"].mean().reset_index()
        moyenne_historique.columns = ["Aire_Sante","Moyenne_Historique"]

        risk_df = future_df.groupby("Aire_Sante").agg(
            Cas_Predits_Total=("Predicted_Cases","sum"),
            Cas_Predits_Max=("Predicted_Cases","max"),
            Cas_Predits_Moyen=("Predicted_Cases","mean"),
            Semaine_Pic=("Predicted_Cases", lambda x: future_df.loc[x.idxmax(),"Semaine_Label"] if len(x) > 0 else "N/A")
        ).reset_index()

        risk_df['Cas_Predits_Total'] = risk_df['Cas_Predits_Total'].round(0).astype(int)
        risk_df['Cas_Predits_Max']   = risk_df['Cas_Predits_Max'].round(0).astype(int)
        risk_df['Cas_Predits_Moyen'] = risk_df['Cas_Predits_Moyen'].round(1)
        risk_df = risk_df.merge(moyenne_historique, on="Aire_Sante", how="left")

        risk_df["Variation_Pct"] = (
            (risk_df["Cas_Predits_Moyen"] - risk_df["Moyenne_Historique"]) /
            risk_df["Moyenne_Historique"].replace(0, 1)
        ) * 100

        risk_df["Categorie_Variation"] = pd.cut(
            risk_df["Variation_Pct"],
            bins=[-np.inf, -seuil_baisse, -10, 10, seuil_hausse, np.inf],
            labels=["Forte baisse","Baisse modérée","Stable","Hausse modérée","Forte hausse"]
        )
        risk_df = risk_df.sort_values("Variation_Pct", ascending=False)

        # ── KPI prédictions ───────────────────────────────────
        st.subheader("📊 KPI Prédictions")
        kp1, kp2, kp3, kp4 = st.columns(4)
        with kp1:
            st.metric("🔮 Total cas prédits", f"{risk_df['Cas_Predits_Total'].sum():,}")
        with kp2:
            n_hausse = (risk_df['Categorie_Variation'].isin(['Forte hausse','Hausse modérée'])).sum()
            st.metric("📈 Aires en hausse", f"{n_hausse}")
        with kp3:
            n_baisse = (risk_df['Categorie_Variation'].isin(['Forte baisse','Baisse modérée'])).sum()
            st.metric("📉 Aires en baisse", f"{n_baisse}")
        with kp4:
            aire_pic = risk_df.loc[risk_df['Cas_Predits_Total'].idxmax(),'Aire_Sante'] if len(risk_df) > 0 else "N/A"
            st.metric("🔴 Aire la + à risque", aire_pic)

        # ── Tableau de synthèse ───────────────────────────────
        st.subheader("📊 Tableau de Synthèse des Prédictions")
        st.dataframe(
            risk_df.style.format({
                'Cas_Predits_Total': '{:.0f}', 'Cas_Predits_Max': '{:.0f}',
                'Cas_Predits_Moyen': '{:.1f}', 'Moyenne_Historique': '{:.1f}',
                'Variation_Pct': '{:.1f}%'
            }), use_container_width=True
        )

        # ── Top 10 ─────────────────────────────────────────────
        st.subheader("📈 Top 10 Aires à Risque")
        top_risk = risk_df.head(10)
        fig_top = px.bar(top_risk, x='Cas_Predits_Total', y='Aire_Sante', orientation='h',
            title='Top 10 Aires à Risque (Cas prédits totaux)',
            labels={'Cas_Predits_Total':'Cas prédits','Aire_Sante':'Aire de santé'},
            color='Variation_Pct', color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig_top, use_container_width=True)

        # ── Timeline prédictions par aire (RESTAURÉ) ──────────
        st.subheader("📅 Timeline des Prédictions")
        aires_pred = sorted(future_df["Aire_Sante"].unique().tolist())
        aire_sel = st.selectbox("Sélectionner une aire de santé", aires_pred, key="aire_timeline_pred")

        df_pred_aire = future_df[future_df["Aire_Sante"] == aire_sel].sort_values("sort_key")
        df_hist_aire = weekly_features[weekly_features["Aire_Sante"] == aire_sel].sort_values("sort_key")

        fig_timeline = go.Figure()
        if len(df_hist_aire) > 0:
            fig_timeline.add_trace(go.Scatter(
                x=df_hist_aire["Semaine_Label"], y=df_hist_aire["Cas_Observes"],
                mode="lines+markers", name="Historique",
                line=dict(color="#2c3e50", width=2), marker=dict(size=5)
            ))
        if len(df_pred_aire) > 0:
            fig_timeline.add_trace(go.Scatter(
                x=df_pred_aire["Semaine_Label"], y=df_pred_aire["Predicted_Cases"],
                mode="lines+markers", name="Prédictions",
                line=dict(color="#e74c3c", width=2, dash="dash"),
                marker=dict(size=6, symbol="diamond"),
                fill="tozeroy", fillcolor="rgba(231,76,60,0.08)"
            ))
            if len(df_hist_aire) > 0:
                fig_timeline.add_vline(
                    x=df_hist_aire["Semaine_Label"].iloc[-1],
                    line_dash="dot", line_color="grey", line_width=1.5,
                    annotation_text="Fin données", annotation_position="top"
                )
        fig_timeline.update_layout(
            title=f"Historique + Prédictions — {aire_sel}",
            xaxis_title="Semaine épidémiologique", yaxis_title="Nombre de cas",
            height=420, template="plotly_white",
            xaxis=dict(tickangle=-45, nticks=25),
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

        # ── CORRECTION 3 : Heatmap des PRÉDICTIONS (pas des données épidémio) ──
        st.subheader("🗓️ Heatmap Hebdomadaire des Prédictions")
        heatmap_data = future_df.pivot_table(
            values='Predicted_Cases', index='Aire_Sante',
            columns='Semaine_Label', aggfunc='sum', fill_value=0
        )
        heatmap_data = heatmap_data.round(0).astype(int)
        # Trier les aires par total décroissant
        heatmap_data['_total'] = heatmap_data.sum(axis=1)
        heatmap_data = heatmap_data.sort_values('_total', ascending=False).drop(columns='_total')

        h_heat = max(400, min(1200, len(heatmap_data) * 22))
        fig_heatmap = go.Figure(go.Heatmap(
            z=heatmap_data.values,
            x=list(heatmap_data.columns),
            y=list(heatmap_data.index),
            colorscale=[
                [0.0,  "rgb(255,255,255)"],
                [0.05, "rgb(255,245,220)"],
                [0.20, "rgb(255,200,100)"],
                [0.40, "rgb(255,140,40)"],
                [0.65, "rgb(220,50,20)"],
                [0.85, "rgb(170,0,0)"],
                [1.0,  "rgb(80,0,0)"],
            ],
            colorbar=dict(title="Cas prédits", thickness=15),
            hoverongaps=False,
            hovertemplate="<b>%{y}</b><br>Semaine : %{x}<br>Cas prédits : %{z}<extra></extra>",
        ))
        fig_heatmap.update_layout(
            title=f"Prédictions par Aire et par Semaine ({n_weeks_pred} semaines)",
            xaxis_title="Semaine", yaxis_title="Aire de Santé",
            height=h_heat, template="plotly_white",
            xaxis=dict(tickangle=-60, tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
            margin=dict(l=160, r=80, t=60, b=120)
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # ── CARTE PRÉDICTIONS RESTAURÉE (complète avec popups) ─
        st.subheader("🗺️ Cartographie des Prédictions")

        gdf_predictions = sa_gdf_enrichi.merge(
            risk_df[['Aire_Sante','Cas_Predits_Total','Cas_Predits_Max','Variation_Pct','Categorie_Variation','Semaine_Pic']],
            left_on='health_area', right_on='Aire_Sante', how='left'
        )
        gdf_predictions['Cas_Predits_Total']    = gdf_predictions['Cas_Predits_Total'].fillna(0).astype(int)
        gdf_predictions['Cas_Predits_Max']      = gdf_predictions['Cas_Predits_Max'].fillna(0).astype(int)
        gdf_predictions['Variation_Pct']        = gdf_predictions['Variation_Pct'].fillna(0)
        gdf_predictions['Categorie_Variation']  = gdf_predictions['Categorie_Variation'].fillna('Stable')
        gdf_predictions['Semaine_Pic']          = gdf_predictions['Semaine_Pic'].fillna('N/A')

        center_lat_p = gdf_predictions.geometry.centroid.y.mean()
        center_lon_p = gdf_predictions.geometry.centroid.x.mean()

        m_predictions = folium.Map(location=[center_lat_p, center_lon_p],
                                   zoom_start=6, tiles='CartoDB positron')

        folium.Choropleth(
            geo_data=gdf_predictions, data=gdf_predictions,
            columns=['health_area','Cas_Predits_Total'],
            key_on='feature.properties.health_area',
            fill_color='YlOrRd', fill_opacity=0.7, line_opacity=0.2,
            legend_name=f'Cas prédits totaux ({n_weeks_pred} semaines)',
            name='Cas prédits totaux'
        ).add_to(m_predictions)

        folium.Choropleth(
            geo_data=gdf_predictions, data=gdf_predictions,
            columns=['health_area','Variation_Pct'],
            key_on='feature.properties.health_area',
            fill_color='RdYlGn_r', fill_opacity=0.7, line_opacity=0.2,
            legend_name='Variation (%) vs moyenne historique',
            name='Variation (%)', show=False
        ).add_to(m_predictions)

        for idx, row in gdf_predictions.iterrows():
            cat = str(row['Categorie_Variation'])
            if cat == 'Forte hausse':     color, icon = 'red',        'arrow-up'
            elif cat == 'Hausse modérée': color, icon = 'orange',     'arrow-up'
            elif cat == 'Stable':         color, icon = 'blue',       'minus'
            elif cat == 'Baisse modérée': color, icon = 'lightgreen', 'arrow-down'
            else:                         color, icon = 'green',      'arrow-down'

            popup_html = f"""
            <div style="width:350px;font-family:Arial;font-size:13px;">
              <h4 style="color:#E4032E;margin:0;padding-bottom:8px;border-bottom:2px solid #E4032E;">
                {row['health_area']}</h4>
              <table style="width:100%;margin-top:10px;border-collapse:collapse;">
                <tr style="background:#f9f9f9;">
                  <td style="padding:6px;font-weight:bold;">🔮 Cas prédits (total)</td>
                  <td style="padding:6px;text-align:right;">{row['Cas_Predits_Total']}</td></tr>
                <tr>
                  <td style="padding:6px;font-weight:bold;">📈 Cas max (semaine)</td>
                  <td style="padding:6px;text-align:right;">{row['Cas_Predits_Max']}</td></tr>
                <tr style="background:#f9f9f9;">
                  <td style="padding:6px;font-weight:bold;">📅 Semaine pic</td>
                  <td style="padding:6px;text-align:right;">{row['Semaine_Pic']}</td></tr>
                <tr>
                  <td style="padding:6px;font-weight:bold;">📊 Variation</td>
                  <td style="padding:6px;text-align:right;color:{'red' if row['Variation_Pct'] > 0 else 'green'};">
                    {row['Variation_Pct']:.1f}%</td></tr>
                <tr style="background:#f0f0f0;">
                  <td colspan="2" style="padding:6px;text-align:center;font-weight:bold;">{cat}</td></tr>
              </table>
            </div>"""

            radius = min(5 + row['Cas_Predits_Total'] / 10, 25)
            folium.CircleMarker(
                location=[row.geometry.centroid.y, row.geometry.centroid.x],
                radius=radius,
                popup=folium.Popup(popup_html, max_width=400),
                color=color, fill=True, fillColor=color, fillOpacity=0.7, weight=2
            ).add_to(m_predictions)

        folium.LayerControl().add_to(m_predictions)
        st_folium(m_predictions, width=1200, height=600, key='carte_predictions_rougeole')

        st.markdown(f"""
        <div style="background:#f0f2f6;padding:1rem;border-radius:8px;margin-top:1rem;">
        <b>🎨 Légende des catégories :</b><br>
        🔴 <b>Forte hausse</b> : Variation ≥{seuil_hausse}% (Action urgente requise)<br>
        🟠 <b>Hausse modérée</b> : Variation entre 10% et {seuil_hausse}%<br>
        🔵 <b>Stable</b> : Variation entre -10% et +10%<br>
        🟡 <b>Baisse modérée</b> : Variation entre -{seuil_baisse}% et -10%<br>
        🟢 <b>Forte baisse</b> : Variation ≤-{seuil_baisse}%
        </div>""", unsafe_allow_html=True)

        # ── Carte zones à risque élevé ─────────────────────────
        st.subheader("🎯 Carte des Zones à Risque Élevé")
        aires_critiques = gdf_predictions[gdf_predictions['Categorie_Variation'] == 'Forte hausse']
        if len(aires_critiques) > 0:
            m_risque = folium.Map(location=[center_lat_p, center_lon_p],
                                  zoom_start=6, tiles='CartoDB positron')
            folium.GeoJson(gdf_predictions,
                style_function=lambda x: {'fillColor':'#e0e0e0','color':'#999999','weight':1,'fillOpacity':0.3},
                name='Toutes les aires').add_to(m_risque)
            for idx, row in aires_critiques.iterrows():
                folium.GeoJson(row.geometry,
                    style_function=lambda x: {'fillColor':'#ff0000','color':'#8B0000','weight':3,'fillOpacity':0.6}
                ).add_to(m_risque)
                folium.Marker(
                    location=[row.geometry.centroid.y, row.geometry.centroid.x],
                    popup=folium.Popup(f"""
                    <div style="width:250px;font-family:Arial;">
                      <h4 style="color:red;margin:0;">⚠️ ALERTE</h4>
                      <p><b>{row['health_area']}</b></p>
                      <p>Cas prédits : <b>{row['Cas_Predits_Total']}</b></p>
                      <p>Hausse : <b style="color:red;">+{row['Variation_Pct']:.1f}%</b></p>
                      <p>Pic : <b>{row['Semaine_Pic']}</b></p>
                    </div>""", max_width=300),
                    icon=folium.Icon(color='red', icon='exclamation-triangle', prefix='fa')
                ).add_to(m_risque)
            st_folium(m_risque, width=1200, height=600, key='carte_risque_rougeole')
            st.error(f"🚨 **{len(aires_critiques)} aires identifiées à risque élevé** - Intervention prioritaire recommandée")
        else:
            st.success("✅ Aucune zone à risque élevé identifiée dans les prédictions")

        # ── Carte de chaleur prédictions ─────────────────────
        if gdf_predictions['Cas_Predits_Total'].sum() > 100:
            st.subheader("🔥 Carte de Chaleur des Cas Prédits")
            heat_data_pred = [[row.geometry.centroid.y, row.geometry.centroid.x, row['Cas_Predits_Total']]
                            for idx, row in gdf_predictions.iterrows()
                if row['Cas_Predits_Total'] > 0]

        if len(heat_data_pred) > 0:
            m_heat = folium.Map(
                location=[center_lat_p, center_lon_p],
                zoom_start=6,
                tiles='CartoDB positron'
            )
            HeatMap(
                heat_data_pred,
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

            st_folium(m_heat, width=1200, height=600, key='heatmap_chaleur_pred_rougeole')
            st.info("💡 Les zones rouges/oranges indiquent les concentrations de cas prédits les plus élevées")

    # ============================================================
    # ALERTES ET RECOMMANDATIONS
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

    # ============================================================
    # TÉLÉCHARGEMENTS
    # ============================================================

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

    # Export GeoJSON
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
