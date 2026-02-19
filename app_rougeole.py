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
# CORRECTION 1 & 2 : MAPPING COLONNES ROBUSTE + SÉPARATEUR CSV
# ============================================================

# Mapping colonnes étendu (variantes majuscules/minuscules/accents)
COLONNES_MAPPING = {
    "Aire_Sante": [
        "Aire_Sante", "aire_sante", "health_area", "HEALTH_AREA", "name_fr", "NAME",
        "nom", "NOM", "aire de sante", "aire_de_sante", "zone_sante", "district",
        "area", "localite", "locality", "nom_aire", "nom aire", "aire", "fosa",
        "nom_fosa", "fosa_name", "Aire_de_Sante", "AIRE_SANTE"
    ],
    "Date_Debut_Eruption": [
        "Date_Debut_Eruption", "date_debut_eruption", "Date_Debut", "date_onset",
        "Date_Onset", "symptom_onset", "date_eruption", "Date_Eruption",
        "date_debut", "DateDebut", "DATE_DEBUT"
    ],
    "Date_Notification": [
        "Date_Notification", "date_notification", "Date_Notif", "date_notif",
        "notification_date", "DateNotif", "DATE_NOTIFICATION"
    ],
    "ID_Cas": [
        "ID_Cas", "id_cas", "ID", "id", "Case_ID", "case_id", "ID_cas",
        "identifiant", "Identifiant", "IDCAS", "id_case"
    ],
    "Age_Mois": [
        "Age_Mois", "age_mois", "Age", "age", "AGE", "Age_Months", "age_months",
        "age_en_mois", "Age_En_Mois", "AGE_MOIS"
    ],
    "Statut_Vaccinal": [
        "Statut_Vaccinal", "statut_vaccinal", "Vaccin", "vaccin",
        "Vaccination_Status", "vaccination_status", "Vacc_Statut",
        "statut_vaccination", "Statut_Vaccination", "STATUT_VACCINAL",
        "vaccinated", "Vaccinated", "non_vaccine", "Non_Vaccine"
    ],
    "Sexe": [
        "Sexe", "sexe", "Sex", "sex", "Gender", "gender", "SEXE", "SEX"
    ],
    "Issue": [
        "Issue", "issue", "Outcome", "outcome", "OUTCOME", "issue_cas",
        "Issue_Cas", "resultat", "Resultat"
    ],
    # ── Colonnes historiques agrégées (Semaine_epi + Annee) ──
    "Semaine_Epi": [
        "Semaine_Epi", "semaine_epi", "Semaine_epi", "SEMAINE_EPI",
        "semaine", "Semaine", "SEMAINE", "week", "Week", "WEEK",
        "epi_week", "Epi_Week", "EPI_WEEK", "epiweek", "Epiweek",
        "se", "SE", "s_epi", "S_epi", "sem_epi", "Sem_Epi",
        "semaine_epidemiologique", "Semaine_Epidemiologique",
        "epi week", "Epi Week", "epid_week", "Epid_Week",
        "week_number", "Week_Number", "no_semaine", "No_Semaine",
        "numero_semaine", "Numero_Semaine", "n_semaine", "N_Semaine",
        "wk", "WK", "sw", "SW"
    ],
    "Annee": [
        "Annee", "annee", "ANNEE", "année", "Année",
        "year", "Year", "YEAR", "an", "An", "AN",
        "yr", "Yr", "annee_epi", "Annee_Epi", "epi_year", "Epi_Year",
        "epiyear", "EpiYear", "annee_epidemiologique", "Annee_Epidemiologique"
    ],
    "Cas_Total": [
        "Cas_Total", "cas_total", "CAS_TOTAL", "Cas", "cas", "CAS",
        "cases", "Cases", "CASES", "nb_cas", "Nb_Cas", "NB_CAS",
        "nombre_cas", "Nombre_Cas", "NOMBRE_CAS", "nbcas", "nbre_cas",
        "count", "Count", "total_cas", "Total_Cas", "confirmed", "Confirmed"
    ],
    "Deces": [
        "Deces", "deces", "DECES", "décès", "Décès",
        "deaths", "Deaths", "DEATHS", "nb_deces", "Nb_Deces",
        "mort", "morts", "dead", "Dead", "nb_morts"
    ],
}

SEP_CANDIDATES = [",", ";", "\t", "|"]

def detect_separator(uploaded_file) -> str:
    """Détecte automatiquement le séparateur CSV."""
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
    """Renommer les colonnes du dataframe selon le mapping standardisé."""
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
    """Convertit une semaine épidémiologique (1-52) en date de début (lundi ISO)."""
    try:
        semaine = int(max(1, min(52, semaine)))
        annee = int(annee)
        return datetime.strptime(f"{annee}-W{semaine:02d}-1", "%G-W%V-%u")
    except Exception:
        try:
            return datetime(int(annee), 1, 1) + timedelta(weeks=int(semaine) - 1)
        except Exception:
            return datetime(2020, 1, 1)

# ── Sidebar ──────────────────────────────────────────────────
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
        help="Format : health_area, Semaine_Epi, Annee, Cas_Total OU Date_Debut_Eruption, Aire_Sante..."
    )
    vaccination_file = st.sidebar.file_uploader(
        "💉 Couverture vaccinale (CSV - optionnel)",
        type=["csv"],
        help="Format : health_area, Taux_Vaccination (en %)"
    )

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

model_hints = {
    "GradientBoosting (Recommandé)": "🎯 **Gradient Boosting** : Très performant pour les séries temporelles.",
    "RandomForest": "🌳 **Random Forest** : Robuste aux valeurs aberrantes.",
    "Ridge Regression": "📊 **Ridge Regression** : Simple et rapide.",
    "Lasso Regression": "🎯 **Lasso Regression** : Sélection automatique des variables.",
    "Decision Tree": "🌲 **Decision Tree** : Simple à interpréter."
}
st.sidebar.info(model_hints[modele_choisi])

# Seuil d'alerte épidémique
st.sidebar.subheader("🚨 Seuil d'Alerte")
seuil_alerte_epidemique = st.sidebar.number_input(
    "Seuil d'alerte (cas/semaine)",
    min_value=1,
    value=10,
    help="Nombre de cas par semaine déclenchant une alerte épidémique"
)
# Fonctions chargement shapefile (inchangées)
def load_health_areas_from_zip(zip_path, iso3=None):
    try:
        tmpdir = tempfile.mkdtemp()
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(tmpdir)
        shp_files = []
        for root, dirs, files in os.walk(tmpdir):
            for f in files:
                if f.endswith('.shp'):
                    shp_files.append(os.path.join(root, f))
        if not shp_files:
            return gpd.GeoDataFrame()
        gdf = gpd.read_file(shp_files[0])
        if iso3 and 'iso3' in gdf.columns:
            gdf = gdf[gdf['iso3'].str.lower() == iso3.lower()]
        if 'health_area' not in gdf.columns:
            for col in gdf.columns:
                if any(kw in col.lower() for kw in ['name','nom','aire','sante','health','area']):
                    gdf = gdf.rename(columns={col: 'health_area'})
                    break
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        return gdf
    except Exception as e:
        st.error(f"❌ Erreur chargement ZIP : {e}")
        return gpd.GeoDataFrame()

def load_shapefile_from_upload(upload_file):
    try:
        tmpdir = tempfile.mkdtemp()
        if upload_file.name.endswith('.zip'):
            with zipfile.ZipFile(upload_file, 'r') as zf:
                zf.extractall(tmpdir)
            shp_files = []
            for root, dirs, files in os.walk(tmpdir):
                for f in files:
                    if f.endswith('.shp'):
                        shp_files.append(os.path.join(root, f))
            if not shp_files:
                return gpd.GeoDataFrame()
            gdf = gpd.read_file(shp_files[0])
        elif upload_file.name.endswith('.geojson'):
            gdf = gpd.read_file(upload_file)
        else:
            return gpd.GeoDataFrame()
        if 'health_area' not in gdf.columns:
            for col in gdf.columns:
                if any(kw in col.lower() for kw in ['name','nom','aire','sante','health','area']):
                    gdf = gdf.rename(columns={col: 'health_area'})
                    break
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        return gdf
    except Exception as e:
        st.error(f"❌ Erreur chargement fichier : {e}")
        return gpd.GeoDataFrame()

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

if sa_gdf.empty or sa_gdf is None:
    st.error("❌ Aucune aire chargée")
    st.stop()

# ── Données fictives (mode démo) ──────────────────────────────
@st.cache_data
def generate_dummy_linelists(_sa_gdf, n=500):
    np.random.seed(42)
    # Générer sur 2 années (2024-2025), semaines 1-52
    aires = _sa_gdf["health_area"].unique()
    rows = []
    for annee in [2024, 2025]:
        for semaine in range(1, 53):
            n_cas = max(0, int(np.random.poisson(
                lam=max(1, 15 * np.sin(semaine * np.pi / 26) + 3))))
            for aire in np.random.choice(aires, size=min(n_cas, len(aires)), replace=False):
                age_mois = int(np.random.gamma(shape=2, scale=30, size=1)[0].clip(6, 180))
                rows.append({
                    "ID_Cas": len(rows) + 1,
                    "Semaine_Epi": semaine,
                    "Annee": annee,
                    "Aire_Sante": aire,
                    "Age_Mois": age_mois,
                    "Statut_Vaccinal": np.random.choice(["Oui", "Non"], p=[0.55, 0.45]),
                    "Sexe": np.random.choice(["M", "F"]),
                    "Issue": np.random.choice(["Guéri", "Décédé", "Inconnu"], p=[0.92, 0.03, 0.05])
                })
    df_demo = pd.DataFrame(rows)
    # Construire date depuis semaine+annee pour compatibilité
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

# ── Chargement des données de cas ─────────────────────────────
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
            # ── CORRECTION 1 : Détection automatique du séparateur ──
            sep = detect_separator(linelist_file)
            try:
                df_raw = pd.read_csv(linelist_file, sep=sep, encoding="utf-8", low_memory=False)
            except UnicodeDecodeError:
                linelist_file.seek(0)
                df_raw = pd.read_csv(linelist_file, sep=sep, encoding="latin-1", low_memory=False)

            # ── CORRECTION 1 : Mapping colonnes robuste ──
            df_raw = normaliser_colonnes(df_raw, COLONNES_MAPPING)

            # ── CORRECTION 2 : Traitement selon format détecté ──
            # CAS A : données historiques agrégées avec Semaine_Epi + Annee + Cas_Total
            if "Semaine_Epi" in df_raw.columns and ("Cas_Total" in df_raw.columns or "Cas" in df_raw.columns):
                cas_col = "Cas_Total" if "Cas_Total" in df_raw.columns else "Cas"
                # Normaliser Semaine_Epi: entier 1-52
                df_raw["Semaine_Epi"] = pd.to_numeric(df_raw["Semaine_Epi"], errors="coerce")
                df_raw = df_raw[df_raw["Semaine_Epi"].between(1, 52, inclusive="both")].copy()
                df_raw["Semaine_Epi"] = df_raw["Semaine_Epi"].astype(int)
                # Normaliser Annee
                if "Annee" in df_raw.columns:
                    df_raw["Annee"] = pd.to_numeric(df_raw["Annee"], errors="coerce")
                    df_raw = df_raw[df_raw["Annee"].notna()].copy()
                    df_raw["Annee"] = df_raw["Annee"].astype(int)
                else:
                    df_raw["Annee"] = datetime.now().year

                # Expansion ligne par cas
                expanded_rows = []
                for _, row in df_raw.iterrows():
                    aire = row.get("Aire_Sante") or row.get("health_area") or row.get("name_fr")
                    semaine = int(row["Semaine_Epi"])
                    annee = int(row.get("Annee", datetime.now().year))
                    cas_total = int(pd.to_numeric(row.get(cas_col, 0), errors="coerce") or 0)
                    deces = int(pd.to_numeric(row.get("Deces", 0), errors="coerce") or 0)
                    # Calculer date depuis semaine+annee (CORRECTION 2)
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
                            "Age_Mois": 0,
                            "Statut_Vaccinal": "Inconnu",
                            "Sexe": "Inconnu",
                            "Issue": issue
                        })
                df = pd.DataFrame(expanded_rows)

            # CAS B : linelist individuelle avec Date_Debut_Eruption
            elif "Date_Debut_Eruption" in df_raw.columns:
                df = df_raw.copy()
                for col in ["Date_Debut_Eruption", "Date_Notification"]:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')

            # CAS C : format non reconnu — tentative générique
            else:
                st.warning("⚠️ Format CSV non standard — tentative de détection automatique...")
                df = df_raw.copy()
                # Chercher toute colonne date
                for col in df.columns:
                    try:
                        test_dates = pd.to_datetime(df[col], errors='coerce')
                        if test_dates.notna().sum() > len(df) * 0.5:
                            df["Date_Debut_Eruption"] = test_dates
                            break
                    except:
                        continue
                if "Date_Debut_Eruption" not in df.columns:
                    st.error("❌ Impossible de détecter une colonne date ou semaine valide. Vérifiez votre fichier.")
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
            if "Statut_Vaccinal" in df.columns:
                vacc_by_area = df.groupby("Aire_Sante").agg({
                    "Statut_Vaccinal": lambda x: ((x == "Non").sum() / len(x) * 100) if len(x) > 0 else 0
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

if "Date_Debut_Eruption" in df.columns:
    df["Date_Debut_Eruption"] = pd.to_datetime(df["Date_Debut_Eruption"], errors='coerce')
else:
    # Si on n'a que Semaine_Epi + Annee, construire la date
    if "Semaine_Epi" in df.columns and "Annee" in df.columns:
        df["Date_Debut_Eruption"] = df.apply(
            lambda r: semaine_vers_date(r["Annee"], r["Semaine_Epi"])
            if pd.notna(r.get("Annee")) and pd.notna(r.get("Semaine_Epi")) else pd.NaT,
            axis=1)
    else:
        df["Date_Debut_Eruption"] = pd.to_datetime(datetime.now())
        st.sidebar.warning("⚠️ Aucune colonne date trouvée")

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
# CORRECTION 2 : si Semaine_Epi et Annee déjà présents (données agrégées), les utiliser
# sinon les déduire depuis Date_Debut_Eruption
if "Semaine_Epi" not in df.columns:
    df["Semaine_Epi"] = df["Date_Debut_Eruption"].apply(
        lambda d: int(d.isocalendar()[1]) if pd.notna(d) else np.nan)

if "Annee" not in df.columns:
    df["Annee"] = df["Date_Debut_Eruption"].dt.year

# Normaliser Semaine_Epi en entier 1-52
df["Semaine_Epi"] = pd.to_numeric(df["Semaine_Epi"], errors="coerce")
df = df[df["Semaine_Epi"].between(1, 52, inclusive="both")].copy()
df["Semaine_Epi"] = df["Semaine_Epi"].astype(int)
df["Annee"] = pd.to_numeric(df["Annee"], errors="coerce").fillna(datetime.now().year).astype(int)

df["Semaine_Annee"] = df["Annee"].astype(str) + "-S" + df["Semaine_Epi"].astype(str).str.zfill(2)
df["sort_key"] = df["Annee"] * 100 + df["Semaine_Epi"]

# ── CORRECTION 3 : Détection automatique dernière semaine ──────
derniere_semaine_epi = int(df.loc[df["sort_key"].idxmax(), "Semaine_Epi"])
derniere_annee = int(df.loc[df["sort_key"].idxmax(), "Annee"])
st.sidebar.info(f"📅 Dernière semaine détectée : **S{derniere_semaine_epi:02d} {derniere_annee}**")

# ── CORRECTION 3 : Filtres dynamiques (remplace start_date/end_date) ──
st.sidebar.subheader("📅 Filtres Temporels & Géographiques")

annees_dispo = sorted(df["Annee"].dropna().unique().astype(int).tolist())
semaines_dispo = sorted(df["Semaine_Epi"].dropna().unique().astype(int).tolist())
aires_dispo = sorted(df["Aire_Sante"].dropna().unique().tolist()) if "Aire_Sante" in df.columns else []

filtre_annees = st.sidebar.multiselect(
    "📅 Années à analyser",
    options=annees_dispo,
    default=annees_dispo,
    help="Par défaut : toutes les années disponibles"
)
filtre_semaines = st.sidebar.multiselect(
    "🗓️ Semaines épidémiologiques",
    options=semaines_dispo,
    default=semaines_dispo,
    format_func=lambda s: f"S{s:02d}",
    help="Par défaut : toutes les semaines"
)
filtre_aires = st.sidebar.multiselect(
    "🏥 Aires de santé",
    options=aires_dispo,
    default=aires_dispo,
    help="Par défaut : toutes les aires"
)

# Application des filtres → df_filtre remplace l'ancien filtre par date
df_filtre = df.copy()
if filtre_annees:
    df_filtre = df_filtre[df_filtre["Annee"].isin([int(a) for a in filtre_annees])]
if filtre_semaines:
    df_filtre = df_filtre[df_filtre["Semaine_Epi"].isin([int(s) for s in filtre_semaines])]
if filtre_aires:
    df_filtre = df_filtre[df_filtre["Aire_Sante"].isin(filtre_aires)]

# Remplacer df par df_filtre pour tout le reste de l'application
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
# PARTIE 4/6 - ONGLETS PRINCIPAUX (CORRECTION 9 : 3 ONGLETS)
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

    # Résumé filtre
    ann_str = ", ".join(str(a) for a in sorted(set(df["Annee"].dropna().astype(int))))
    nb_aires_filtre = df["Aire_Sante"].nunique()
    nb_sem_filtre = df["Semaine_Epi"].nunique()
    st.caption(f"📌 Analyse : Années **{ann_str}** | **{nb_aires_filtre}** aires | **{nb_sem_filtre}** semaines | Dernière semaine : **S{derniere_semaine_epi:02d} {derniere_annee}**")

    # ── CORRECTION 4 : KPI conditionnels ──────────────────────
    total_cas = len(df)
    n_aires_touchees = df["Aire_Sante"].nunique()
    pct_aires = (n_aires_touchees / len(sa_gdf)) * 100
    taux_letalite = ((df["Issue"] == "Décédé").mean() * 100) if "Issue" in df.columns else 0

    # KPI de base (toujours affichés)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📈 Cas totaux", f"{total_cas:,}")
    with col2:
        # CORRECTION 4 : Vaccination — uniquement si données réelles disponibles
        if has_vaccination_reel:
            taux_non_vac = (df["Statut_Vaccinal"] == "Non").mean() * 100
            delta_vac = taux_non_vac - 45
            st.metric("💉 Non vaccinés", f"{taux_non_vac:.1f}%", delta=f"{delta_vac:+.1f}%")
        else:
            st.metric("💉 Non vaccinés", "N/A")
    with col3:
        # CORRECTION 4 : Âge médian — depuis données réelles OU WorldPop
        if has_age_reel:
            age_median = df["Age_Mois"].median()
            st.metric("👶 Âge médian", f"{int(age_median)} mois")
        elif age_median_worldpop is not None:
            st.metric("👶 Âge médian (WorldPop)", f"{int(age_median_worldpop)} mois")
        else:
            st.metric("👶 Âge médian", "N/A")
    with col4:
        if taux_letalite > 0:
            st.metric("☠️ Létalité", f"{taux_letalite:.2f}%")
        else:
            st.metric("☠️ Létalité", "N/A")
    with col5:
        st.metric("🗺️ Aires touchées", f"{n_aires_touchees}/{len(sa_gdf)}", delta=f"{pct_aires:.0f}%")

    # ── Analyse temporelle ────────────────────────────────────
    st.header("📈 Analyse Temporelle par Semaines Épidémiologiques")

    weekly_cases = df.groupby(["Annee", "Semaine_Epi", "sort_key"]).size().reset_index(name="Cas")
    weekly_cases["Semaine_Label"] = weekly_cases["Annee"].astype(str) + "-S" + weekly_cases["Semaine_Epi"].astype(str).str.zfill(2)
    weekly_cases = weekly_cases.sort_values("sort_key")

    fig_epi = go.Figure()
    fig_epi.add_trace(go.Scatter(
        x=weekly_cases["Semaine_Label"],
        y=weekly_cases["Cas"],
        mode="lines+markers",
        name="Cas observés",
        line=dict(color="#d32f2f", width=3),
        marker=dict(size=6),
        hovertemplate="<b>%{x}</b><br>Cas : %{y}<extra></extra>"
    ))

    from scipy.signal import savgol_filter
    if len(weekly_cases) > 5:
        tendance = savgol_filter(
            weekly_cases["Cas"],
            window_length=min(7, len(weekly_cases) if len(weekly_cases) % 2 == 1 else len(weekly_cases) - 1),
            polyorder=2
        )
        fig_epi.add_trace(go.Scatter(
            x=weekly_cases["Semaine_Label"],
            y=tendance,
            mode="lines",
            name="Tendance",
            line=dict(color="#1976d2", width=2, dash="dash"),
        ))

    fig_epi.add_hline(
        y=seuil_alerte_epidemique,
        line_dash="dot", line_color="orange",
        annotation_text=f"Seuil d'alerte ({seuil_alerte_epidemique} cas/sem)",
        annotation_position="right"
    )
    fig_epi.update_layout(
        title="Courbe épidémique par semaines épidémiologiques",
        xaxis_title="Semaine épidémiologique",
        yaxis_title="Nombre de cas",
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

    # ── CORRECTION 6 : Top 10 — 2 onglets (taux attaque + nb cas) ──
    st.header("🏆 Top 10 des Aires les Plus Touchées")

    top_data = sa_gdf_with_cases[["health_area", "Cas_Observes", "Taux_Attaque_10000"]].copy()
    top_data = top_data[top_data["Cas_Observes"] > 0]

    has_taux_attaque = (
        "Taux_Attaque_10000" in top_data.columns
        and top_data["Taux_Attaque_10000"].notna().sum() > 0
    )

    if has_taux_attaque:
        tab_ta, tab_cas = st.tabs(["📊 Par taux d'attaque (/10 000 hab.)", "📊 Par nombre de cas"])
        with tab_ta:
            top10_ta = top_data.nlargest(10, "Taux_Attaque_10000").sort_values("Taux_Attaque_10000")
            fig_top_ta = go.Figure(go.Bar(
                x=top10_ta["Taux_Attaque_10000"],
                y=top10_ta["health_area"],
                orientation="h",
                marker_color="#e53935",
                text=top10_ta["Taux_Attaque_10000"].apply(lambda v: f"{v:.1f}" if pd.notna(v) else "N/A"),
                textposition="outside"
            ))
            fig_top_ta.update_layout(
                title="Top 10 — Taux d'attaque (/10 000 hab.)",
                xaxis_title="Taux d'attaque (cas / 10 000 hab.)",
                height=400, template="plotly_white"
            )
            st.plotly_chart(fig_top_ta, use_container_width=True)
        with tab_cas:
            top10_cas = top_data.nlargest(10, "Cas_Observes").sort_values("Cas_Observes")
            fig_top_cas = go.Figure(go.Bar(
                x=top10_cas["Cas_Observes"],
                y=top10_cas["health_area"],
                orientation="h",
                marker_color="#c62828",
                text=top10_cas["Cas_Observes"].astype(int),
                textposition="outside"
            ))
            fig_top_cas.update_layout(
                title="Top 10 — Nombre de cas",
                xaxis_title="Nombre de cas",
                height=400, template="plotly_white"
            )
            st.plotly_chart(fig_top_cas, use_container_width=True)
    else:
        top10_cas = top_data.nlargest(10, "Cas_Observes").sort_values("Cas_Observes")
        fig_top_cas = go.Figure(go.Bar(
            x=top10_cas["Cas_Observes"],
            y=top10_cas["health_area"],
            orientation="h",
            marker_color="#c62828",
            text=top10_cas["Cas_Observes"].astype(int),
            textposition="outside"
        ))
        fig_top_cas.update_layout(
            title="Top 10 — Nombre de cas",
            xaxis_title="Nombre de cas",
            height=400, template="plotly_white"
        )
        st.plotly_chart(fig_top_cas, use_container_width=True)

    # ── Distribution par âge (CORRECTION 5 : conditionnel) ───
    st.subheader("👶 Distribution par Tranches d'Âge")

    if has_age_reel:
        df["Tranche_Age"] = pd.cut(
            df["Age_Mois"],
            bins=[0, 12, 60, 120, 180],
            labels=["0-1 an", "1-5 ans", "5-10 ans", "10-15 ans"]
        )
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
            fig_age = px.bar(
                age_stats, x="Tranche_Age", y="Nombre_Cas",
                title="Cas par tranche d'âge",
                color="Nombre_Cas", color_continuous_scale="Reds",
                text="Nombre_Cas"
            )
            fig_age.update_traces(textposition="outside")
            st.plotly_chart(fig_age, use_container_width=True)
        with col2:
            # CORRECTION 5 : vaccination dans le graphique seulement si disponible
            if has_vaccination_reel and "Pct_Non_Vaccines" in age_stats.columns and age_stats["Pct_Non_Vaccines"].sum() > 0:
                fig_vacc_age = px.bar(
                    age_stats, x="Tranche_Age", y="Pct_Non_Vaccines",
                    title="% non vaccinés par âge",
                    color="Pct_Non_Vaccines", color_continuous_scale="Oranges",
                    text="Pct_Non_Vaccines"
                )
                fig_vacc_age.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(fig_vacc_age, use_container_width=True)
            else:
                st.info("ℹ️ Données de vaccination par âge non disponibles dans ce fichier")
    else:
        st.info("ℹ️ Données d'âge non disponibles dans ce fichier")

    # ── CORRECTION 8 : Heatmap améliorée ─────────────────────
    st.header("🔥 Heatmap Épidémiologique — Cas par Aire et Semaine")

    pivot_data = df.groupby(["Aire_Sante", "Annee", "Semaine_Epi", "sort_key"]).size().reset_index(name="Cas")
    pivot_data["Label_Sem"] = pivot_data["Annee"].astype(str) + "-S" + pivot_data["Semaine_Epi"].astype(str).str.zfill(2)

    pivot_table = pivot_data.pivot_table(
        index="Aire_Sante", columns="Label_Sem", values="Cas", fill_value=0
    )

    # Tri colonnes par sort_key
    col_order = (pivot_data[["Label_Sem", "sort_key"]]
                 .drop_duplicates("Label_Sem")
                 .sort_values("sort_key")["Label_Sem"].tolist())
    col_order = [c for c in col_order if c in pivot_table.columns]
    pivot_table = pivot_table[col_order]

    # Limiter à 60 semaines max pour lisibilité
    if pivot_table.shape[1] > 60:
        pivot_table = pivot_table[col_order[-60:]]
        st.caption("ℹ️ Les 60 dernières semaines sont affichées pour la lisibilité.")

    # Trier les aires par total de cas décroissant
    pivot_table["_total"] = pivot_table.sum(axis=1)
    pivot_table = pivot_table.sort_values("_total", ascending=False).drop(columns="_total")

    heat_y = pivot_table.index.tolist()
    heat_x = pivot_table.columns.tolist()
    heat_z = pivot_table.values

    # Hauteur dynamique
    h_heat = max(400, min(1000, len(heat_y) * 22))

    fig_heat = go.Figure(go.Heatmap(
        z=heat_z,
        x=heat_x,
        y=heat_y,
        colorscale=[
            [0.0,  "rgb(255,255,255)"],
            [0.05, "rgb(255,245,220)"],
            [0.15, "rgb(255,220,130)"],
            [0.35, "rgb(255,160,60)"],
            [0.60, "rgb(220,60,30)"],
            [0.85, "rgb(180,10,10)"],
            [1.0,  "rgb(100,0,0)"],
        ],
        colorbar=dict(
            title="Cas",
            thickness=15,
            len=0.8,
            tickfont=dict(size=10)
        ),
        hoverongaps=False,
        hovertemplate="<b>%{y}</b><br>Semaine : %{x}<br>Cas : %{z}<extra></extra>",
        zsmooth=False,
    ))
    fig_heat.update_layout(
        title="Distribution hebdomadaire des cas par aire de santé",
        xaxis_title="Semaine épidémiologique",
        yaxis_title="Aire de santé",
        height=h_heat,
        template="plotly_white",
        xaxis=dict(tickangle=-60, tickfont=dict(size=9), nticks=30),
        yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
        margin=dict(l=160, r=80, t=60, b=120),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Pyramide des âges WorldPop (inchangée — conserver votre code) ──
    # [Conservez INTÉGRALEMENT votre section "Pyramide des Âges" existante ici]

    # ── Données climatiques (inchangées — conserver votre code) ──
    # [Conservez INTÉGRALEMENT votre section données climatiques ici]

# ============================================================
# TAB 2 — CARTOGRAPHIE (inchangée)
# ============================================================
with tab2:
    # [Conservez INTÉGRALEMENT votre section cartographie existante ici]
    # Carte de situation actuelle + Heatmap folium + Légende
    st.header("🗺️ Cartographie de la Situation Actuelle")

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
            colors=["#e8f5e9","#81c784","#ffeb3b","#ff9800","#f44336","#b71c1c"],
            vmin=0, vmax=max_cases,
            caption="Nombre de cas observés"
        )
        colormap.add_to(m)

    for idx, row in sa_gdf_with_cases.iterrows():
        aire_name = row["health_area"]
        cas_obs = int(row.get("Cas_Observes", 0))
        pop_enfants = row.get("Pop_Enfants", np.nan)
        taux_attaque = row.get("Taux_Attaque_10000", np.nan)
        urbanisation = row.get("Urbanisation", "N/A")
        densite = row.get("Densite_Pop", np.nan)

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

        fill_color = colormap(row["Cas_Observes"]) if max_cases > 0 else "#e0e0e0"
        line_color = "#b71c1c" if row["Cas_Observes"] >= seuil_alerte_epidemique else "black"
        line_weight = 2 if row["Cas_Observes"] >= seuil_alerte_epidemique else 0.5

        folium.GeoJson(
            row["geometry"],
            style_function=lambda x, color=fill_color, weight=line_weight, border=line_color: {
                "fillColor": color, "color": border,
                "weight": weight, "fillOpacity": 0.7
            },
            tooltip=folium.Tooltip(f"<b>{aire_name}</b><br>{cas_obs} cas", sticky=True),
            popup=folium.Popup(popup_html, max_width=400)
        ).add_to(m)

        if cas_obs > 0:
            folium.Marker(
                location=[row.geometry.centroid.y, row.geometry.centroid.x],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:7pt;color:black;font-weight:normal;'
                         f'background:none;padding:0;border:none;white-space:nowrap;">{aire_name}</div>'
                )
            ).add_to(m)

    heat_data = [
        [row.geometry.centroid.y, row.geometry.centroid.x, row["Cas_Observes"]]
        for idx, row in sa_gdf_with_cases.iterrows()
        if row["Cas_Observes"] > 0
    ]
    if heat_data:
        HeatMap(
            heat_data, radius=20, blur=25, max_zoom=13,
            gradient={0.0: "blue", 0.5: "yellow", 1.0: "red"}
        ).add_to(m)

    legend_html = f"""
    <div style="position:fixed;bottom:50px;left:50px;width:250px;
         background-color:white;border:2px solid grey;z-index:9999;
         font-size:14px;padding:10px;border-radius:5px;">
        <p style="margin:0;font-weight:bold;">📊 Légende</p>
        <p style="margin:5px 0;"><span style="background-color:#e8f5e9;padding:2px 8px;">Faible</span> 0-{max_cases//3:.0f} cas</p>
        <p style="margin:5px 0;"><span style="background-color:#ffeb3b;padding:2px 8px;">Moyen</span> {max_cases//3:.0f}-{2*max_cases//3:.0f} cas</p>
        <p style="margin:5px 0;"><span style="background-color:#f44336;padding:2px 8px;">Élevé</span> >{2*max_cases//3:.0f} cas</p>
        <p style="margin:5px 0;color:red;font-weight:bold;">⚠️ Seuil alerte : {seuil_alerte_epidemique} cas/sem</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    st_folium(m, width=1400, height=650, key="carte_principale_rougeole")
# ============================================================
# TAB 3 — MODÉLISATION PRÉDICTIVE (CORRECTION 7)
# ============================================================
with tab3:
    st.header("🔮 Modélisation Prédictive par Semaines Épidémiologiques")

    # ── CORRECTION 7 : génération des semaines FUTURES correctes ──
    def generer_semaines_futures(derniere_sem: int, derniere_an: int, n_weeks: int) -> pd.DataFrame:
        """
        Génère n_weeks semaines STRICTEMENT après la dernière semaine observée.
        Jamais dans le passé. Gère le passage d'une année à l'autre.
        """
        futures = []
        sem = derniere_sem
        an  = derniere_an
        for _ in range(n_weeks):
            sem += 1
            if sem > 52:
                sem = 1
                an += 1
            futures.append({
                "Semaine_Future": sem,
                "Annee_Future": an,
                "Semaine_Label_Future": f"{an}-S{sem:02d}",
                "sort_key_future": an * 100 + sem
            })
        return pd.DataFrame(futures)

    df_futures = generer_semaines_futures(derniere_semaine_epi, derniere_annee, n_weeks_pred)

    st.markdown(f"""
    <div style="background:#fef9f9;border-left:4px solid #e74c3c;padding:1rem;border-radius:6px;margin:.5rem 0;">
    <b>Configuration :</b><br>
    Dernière semaine de données : <b>S{derniere_semaine_epi:02d} {derniere_annee}</b><br>
    Horizon de prédiction : <b>{pred_mois} mois ({n_weeks_pred} semaines)</b><br>
    Prédictions de <b>{df_futures.iloc[0]['Semaine_Label_Future']}</b>
    à <b>{df_futures.iloc[-1]['Semaine_Label_Future']}</b>
    </div>
    """, unsafe_allow_html=True)

    # Choix du modèle ML
    def get_model(choix):
        if "GradientBoosting" in choix:
            return GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        elif "RandomForest" in choix:
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif "Ridge" in choix:
            return Ridge(alpha=1.0)
        elif "Lasso" in choix:
            return Lasso(alpha=1.0)
        elif "Decision Tree" in choix:
            return DecisionTreeRegressor(max_depth=5, random_state=42)
        return GradientBoostingRegressor(n_estimators=100, random_state=42)

    if st.button("🚀 LANCER LA MODÉLISATION", type="primary"):

        aires_mod = sa_gdf_with_cases["health_area"].unique()
        if len(aires_mod) == 0:
            st.error("❌ Aucune aire de santé disponible.")
            st.stop()

        progress_bar = st.progress(0)
        status_text = st.empty()
        resultats = []
        predictions_par_aire = []

        for i_aire, aire in enumerate(aires_mod):
            progress_bar.progress((i_aire + 1) / len(aires_mod))
            status_text.text(f"Modélisation : {aire} ({i_aire+1}/{len(aires_mod)})")

            # ── CORRECTION 7 : données historiques PAR AIRE ──────
            df_aire = df[df["Aire_Sante"] == aire].copy()
            if len(df_aire) < 5:
                continue

            # Agréger par semaine (une ligne = une semaine)
            df_agg = (df_aire
                      .groupby(["Annee", "Semaine_Epi", "sort_key"])
                      .size().reset_index(name="Cas")
                      .sort_values("sort_key"))

            if len(df_agg) < 5:
                continue

            # Features spécifiques à cette aire
            def build_features(df_src):
                X = pd.DataFrame()
                X["semaine"] = df_src["Semaine_Epi"].values
                X["annee"]   = df_src["Annee"].values
                X["sin_sem"] = np.sin(2 * np.pi * df_src["Semaine_Epi"].values / 52)
                X["cos_sem"] = np.cos(2 * np.pi * df_src["Semaine_Epi"].values / 52)
                X["trimestre"] = ((df_src["Semaine_Epi"].values - 1) // 13 + 1)
                # Enrichissements climatiques (si disponibles dans les données WorldPop/GEE)
                row_aire = sa_gdf_with_cases[sa_gdf_with_cases["health_area"] == aire]
                if len(row_aire) > 0:
                    r = row_aire.iloc[0]
                    X["pop_enfants"] = float(r.get("Pop_Enfants", 0) or 0)
                    X["densite"]     = float(r.get("Densite_Pop", 0) or 0)
                    X["humidite"]    = float(r.get("Humidite_Moy", 0) or 0)
                    X["temperature"] = float(r.get("Temp_Moy", 0) or 0)
                return X

            X_train = build_features(df_agg)
            y_train = df_agg["Cas"].values.astype(float)

            # Imputation NaN
            imputer = SimpleImputer(strategy="mean")
            X_imp = imputer.fit_transform(X_train)
            X_imp = np.nan_to_num(X_imp, nan=0.0, posinf=0.0, neginf=0.0)

            # Normalisation
            scaler = StandardScaler()
            try:
                X_scaled = scaler.fit_transform(X_imp)
                X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                continue

            # Entraînement
            try:
                model = get_model(modele_choisi)
                model.fit(X_scaled, y_train)
            except Exception:
                continue

            # ── CORRECTION 7 : Prédictions futures PAR AIRE ──────
            # df_futures contient les semaines strictement après la dernière semaine observée
            X_fut = build_features(
                pd.DataFrame({
                    "Semaine_Epi": df_futures["Semaine_Future"].values,
                    "Annee": df_futures["Annee_Future"].values,
                    "sort_key": df_futures["sort_key_future"].values
                })
            )
            X_fut_imp = imputer.transform(X_fut)
            X_fut_imp = np.nan_to_num(X_fut_imp, nan=0.0, posinf=0.0, neginf=0.0)
            try:
                X_fut_scaled = scaler.transform(X_fut_imp)
                X_fut_scaled = np.nan_to_num(X_fut_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                X_fut_scaled = np.zeros((len(df_futures), X_scaled.shape[1]))

            try:
                y_pred = model.predict(X_fut_scaled)
                y_pred = np.maximum(0, y_pred)  # pas de valeurs négatives
            except Exception:
                y_pred = np.zeros(len(df_futures))

            # Métriques
            try:
                y_pred_train = model.predict(X_scaled)
                from sklearn.metrics import mean_absolute_error, r2_score
                mae = mean_absolute_error(y_train, y_pred_train)
                r2  = r2_score(y_train, y_pred_train)
            except Exception:
                mae, r2 = np.nan, np.nan

            # Variation vs dernière valeur observée
            derniere_val_obs = float(df_agg["Cas"].iloc[-1])
            val_max_pred = float(y_pred.max()) if len(y_pred) > 0 else 0
            sem_pic_idx  = int(np.argmax(y_pred)) if len(y_pred) > 0 else 0
            sem_pic_label = df_futures.iloc[sem_pic_idx]["Semaine_Label_Future"]
            variation_pct = round((val_max_pred - derniere_val_obs) / max(derniere_val_obs, 1) * 100, 1)

            resultats.append({
                "health_area": aire,
                "Cas_Observes_Total": int(df_agg["Cas"].sum()),
                "Derniere_Val_Obs": round(derniere_val_obs, 1),
                "Cas_Predits_Total": int(y_pred.sum()),
                "Cas_Predits_Max": round(val_max_pred, 1),
                "Semaine_Pic": sem_pic_label,
                "Variation_Pct": variation_pct,
                "MAE": round(mae, 2) if not np.isnan(mae) else np.nan,
                "R2": round(r2, 3) if not np.isnan(r2) else np.nan,
            })

            for j, row_f in df_futures.iterrows():
                predictions_par_aire.append({
                    "health_area": aire,
                    "Semaine_Pred": row_f["Semaine_Future"],
                    "Annee_Pred": row_f["Annee_Future"],
                    "Semaine_Label": row_f["Semaine_Label_Future"],
                    "sort_key_pred": row_f["sort_key_future"],
                    "Cas_Predits": round(float(y_pred[j]), 1),
                })

        progress_bar.empty()
        status_text.empty()

        if not resultats:
            st.warning("⚠️ Aucun résultat. Vérifiez que les données ont suffisamment de semaines par aire.")
            st.stop()

        risk_df = pd.DataFrame(resultats)
        pred_df = pd.DataFrame(predictions_par_aire)

        # Catégorisation
        seuil_hausse = 30
        seuil_baisse = 20

        def categoriser(v):
            if pd.isna(v):             return "Données insuffisantes"
            if v >= seuil_hausse:      return "Forte hausse"
            elif v >= 10:              return "Hausse modérée"
            elif v <= -seuil_baisse:   return "Forte baisse"
            elif v <= -10:             return "Baisse modérée"
            else:                      return "Stable"

        risk_df["Categorie_Variation"] = risk_df["Variation_Pct"].apply(categoriser)
        risk_df["Variation_Pct"] = risk_df["Variation_Pct"].astype(str)

        # Sauvegarder en session state
        st.session_state["risk_df"]  = risk_df
        st.session_state["pred_df"]  = pred_df
        st.session_state["model_trained"] = True
        st.success(f"✅ Modélisation terminée pour {len(resultats)} aires de santé.")

    # ── Affichage résultats ───────────────────────────────────
    if st.session_state.get("model_trained") and st.session_state.get("risk_df") is not None:
        risk_df = st.session_state["risk_df"]
        pred_df = st.session_state["pred_df"]

        st.subheader("📊 Résultats — Variation prédite par aire")
        st.dataframe(
            risk_df.sort_values("Variation_Pct", ascending=False),
            use_container_width=True, height=350
        )

        # ── Graphique prédiction par aire ─────────────────────
        st.subheader("📈 Prédictions par aire de santé")
        aires_pred = sorted(pred_df["health_area"].unique().tolist())
        aire_sel = st.selectbox("Sélectionner une aire", aires_pred)

        df_pred_aire = pred_df[pred_df["health_area"] == aire_sel].sort_values("sort_key_pred")
        df_hist_aire = (df[df["Aire_Sante"] == aire_sel]
                        .groupby(["Annee","Semaine_Epi","sort_key","Semaine_Annee"])
                        .size().reset_index(name="Cas")
                        .sort_values("sort_key"))

        fig_pred = go.Figure()
        if len(df_hist_aire) > 0:
            fig_pred.add_trace(go.Scatter(
                x=df_hist_aire["Semaine_Annee"], y=df_hist_aire["Cas"],
                mode="lines+markers", name="Données historiques",
                line=dict(color="#2c3e50", width=2), marker=dict(size=5)
            ))
        if len(df_pred_aire) > 0:
            fig_pred.add_trace(go.Scatter(
                x=df_pred_aire["Semaine_Label"], y=df_pred_aire["Cas_Predits"],
                mode="lines+markers", name="Prédictions",
                line=dict(color="#e74c3c", width=2, dash="dash"),
                marker=dict(size=6, symbol="diamond"),
                fill="tozeroy", fillcolor="rgba(231,76,60,0.08)"
            ))
            # Ligne de séparation historique/prédiction
            if len(df_hist_aire) > 0:
                fig_pred.add_vline(
                    x=df_hist_aire["Semaine_Annee"].iloc[-1],
                    line_dash="dot", line_color="grey", line_width=1.5,
                    annotation_text="Fin données", annotation_position="top"
                )
        fig_pred.update_layout(
            title=f"Historique et prédictions — {aire_sel}",
            xaxis_title="Semaine épidémiologique",
            yaxis_title="Nombre de cas",
            height=420, template="plotly_white",
            xaxis=dict(tickangle=-45, nticks=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        # ── Carte des risques prédits ─────────────────────────
        st.subheader("🗺️ Carte des risques prédits")

        gdf_risk = sa_gdf_with_cases.merge(
            risk_df[["health_area","Variation_Pct","Categorie_Variation","Cas_Predits_Total","Semaine_Pic"]],
            on="health_area", how="left"
        )

        m2 = folium.Map(
            location=[gdf_risk.geometry.centroid.y.mean(), gdf_risk.geometry.centroid.x.mean()],
            zoom_start=6, tiles="CartoDB positron"
        )

        def couleur_risque(cat):
            c = {
                "Forte hausse":     "#c0392b",
                "Hausse modérée":   "#e67e22",
                "Stable":           "#f1c40f",
                "Baisse modérée":   "#2ecc71",
                "Forte baisse":     "#27ae60",
                "Données insuffisantes": "#aaaaaa"
            }
            return c.get(str(cat), "#aaaaaa")

        for idx, row in gdf_risk.iterrows():
            try:
                if row["geometry"] is None or row["geometry"].is_empty:
                    continue
                cat  = row.get("Categorie_Variation", "Données insuffisantes")
                var  = row.get("Variation_Pct", "N/A")
                aire = row.get("health_area", "N/A")
                pred_total = row.get("Cas_Predits_Total", "N/A")
                sem_pic = row.get("Semaine_Pic", "N/A")

                popup_pred = f"""
                <div style="font-family:Arial;font-size:12px;min-width:200px;">
                  <h4 style="margin:0 0 8px;color:#c0392b;">🔮 {aire}</h4>
                  <table style="width:100%;">
                    <tr><td><b>Cas prédits (total)</b></td><td>{pred_total}</td></tr>
                    <tr><td><b>Semaine pic</b></td><td>{sem_pic}</td></tr>
                    <tr><td><b>Variation</b></td><td>{var}%</td></tr>
                    <tr><td><b>Catégorie</b></td><td>{cat}</td></tr>
                  </table>
                </div>"""

                folium.GeoJson(
                    row["geometry"],
                    style_function=lambda x, c=couleur_risque(cat): {
                        "fillColor": c, "color": "#333",
                        "weight": 0.8, "fillOpacity": 0.75
                    },
                    tooltip=folium.Tooltip(f"<b>{aire}</b><br>{cat}<br>Var: {var}%", sticky=True),
                    popup=folium.Popup(popup_pred, max_width=280)
                ).add_to(m2)
            except Exception:
                continue

        legend_pred = """
        <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
             background:white;padding:12px;border-radius:8px;
             box-shadow:0 2px 8px rgba(0,0,0,.2);font-size:12px;">
        <b>🔮 Tendance prédite</b><br>
        <span style="color:#c0392b">■</span> Forte hausse (&ge;+30%)<br>
        <span style="color:#e67e22">■</span> Hausse modérée (+10 à +30%)<br>
        <span style="color:#f1c40f">■</span> Stable (-10% à +10%)<br>
        <span style="color:#2ecc71">■</span> Baisse modérée (-10 à -20%)<br>
        <span style="color:#27ae60">■</span> Forte baisse (&le;-20%)<br>
        <span style="color:#aaaaaa">■</span> Données insuffisantes
        </div>"""
        m2.get_root().html.add_child(folium.Element(legend_pred))
        st_folium(m2, width=1400, height=600, key="carte_predictions_rougeole")

        # ── Export ────────────────────────────────────────────
        st.subheader("📥 Export des résultats")
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            csv_risk = risk_df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                "📥 Résultats synthèse (CSV)",
                data=csv_risk,
                file_name=f"predictions_rougeole_S{derniere_semaine_epi:02d}_{derniere_annee}.csv",
                mime="text/csv"
            )
        with col_exp2:
            csv_pred = pred_df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                "📥 Prédictions détaillées (CSV)",
                data=csv_pred,
                file_name=f"pred_detail_S{derniere_semaine_epi:02d}_{derniere_annee}.csv",
                mime="text/csv"
            )
