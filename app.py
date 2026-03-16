from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Centro de Soporte TI",
    page_icon=":ticket:",
    layout="wide",
    initial_sidebar_state="expanded",
)


BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "datasets"
FALLBACK_DATASET_DIR = BASE_DIR.parent / "datasets"
BRANDING_DIR = BASE_DIR / "branding"
BRANDING_ASSETS = {
    "logo": BRANDING_DIR / "logo.png",
    "fondo_app": BRANDING_DIR / "fondo-app.jpg",
    "fondo_sidebar": BRANDING_DIR / "fondo-sidebar.jpg",
    "fondo_hero": BRANDING_DIR / "fondo-hero.jpg",
}
FILTRO_KEYS = {
    "departamento": "filtro_departamento",
    "tema": "filtro_tema",
    "fuente": "filtro_fuente",
    "estado": "filtro_estado",
    "agente": "filtro_agente",
    "mes": "filtro_mes",
}
PENDING_KEYS = {
    "departamento": "pending_departamento",
    "agente": "pending_agente",
}


def encontrar_dataset_tickets() -> Path | None:
    coincidencias_locales = sorted(DATASET_DIR.glob("*Tickets*.csv"))
    if coincidencias_locales:
        return coincidencias_locales[0]
    coincidencias_fallback = sorted(FALLBACK_DATASET_DIR.glob("*Tickets*.csv"))
    return coincidencias_fallback[0] if coincidencias_fallback else None


def imagen_a_data_uri(ruta: Path | None) -> str | None:
    if ruta is None or not ruta.exists():
        return None
    mime_type, _ = mimetypes.guess_type(ruta.name)
    if mime_type is None:
        mime_type = "image/png"
    contenido = base64.b64encode(ruta.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{contenido}"


def valor_bool(serie: pd.Series) -> pd.Series:
    return serie.fillna("").astype(str).str.strip().str.lower().isin(["si", "sí", "yes", "true", "1"])


@st.cache_data
def cargar_tickets() -> tuple[pd.DataFrame, Path | None]:
    ruta = encontrar_dataset_tickets()
    if ruta is None:
        return pd.DataFrame(), None

    df = pd.read_csv(ruta)

    fechas = [
        "Fecha de creación",
        "Última actualización",
        "Fecha de expiración de SLA",
        "Fecha de Vencimiento",
        "Fecha de cierre",
    ]
    for columna in fechas:
        if columna in df.columns:
            df[columna] = pd.to_datetime(df[columna], errors="coerce")

    numeros = ["Cuenta de hilos", "Reabrir contador", "Recuento de datos adjuntos", "Recuento de tarea"]
    for columna in numeros:
        if columna in df.columns:
            df[columna] = pd.to_numeric(df[columna], errors="coerce").fillna(0).astype(int)

    df["Agente asignado"] = df["Agente asignado"].fillna("Sin asignar").replace("", "Sin asignar")
    df["Equipo asignado"] = df["Equipo asignado"].fillna("Sin equipo").replace("", "Sin equipo")
    df["Departamento"] = df["Departamento"].fillna("Sin departamento").replace("", "Sin departamento")
    df["Temas de ayuda"] = df["Temas de ayuda"].fillna("Sin clasificar").replace("", "Sin clasificar")
    df["Fuente"] = df["Fuente"].fillna("Sin fuente").replace("", "Sin fuente")
    df["Estado actual"] = df["Estado actual"].fillna("Sin estado").replace("", "Sin estado")
    df["Prioridad"] = df["Prioridad"].fillna("Sin prioridad").replace("", "Sin prioridad")

    df["atrasado_bool"] = valor_bool(df["Atrasado"])
    df["respondio_bool"] = valor_bool(df["Respondió"])
    df["enlazado_bool"] = valor_bool(df["Enlazado"])
    df["unido_bool"] = valor_bool(df["Unido"])
    df["cerrado_bool"] = df["Estado actual"].astype(str).str.lower().eq("cerrado")
    df["abierto_bool"] = df["Estado actual"].astype(str).str.lower().isin(["abierto", "en progreso"])

    df["mes"] = df["Fecha de creación"].dt.to_period("M").astype(str)
    df["dia"] = df["Fecha de creación"].dt.date
    df["semana"] = df["Fecha de creación"].dt.to_period("W").astype(str)
    df["hora_creacion"] = df["Fecha de creación"].dt.hour
    df["tiempo_resolucion_horas"] = (
        (df["Fecha de cierre"] - df["Fecha de creación"]).dt.total_seconds() / 3600
    ).where(df["Fecha de cierre"].notna())
    df["tiempo_ultima_actualizacion_horas"] = (
        (df["Última actualización"] - df["Fecha de creación"]).dt.total_seconds() / 3600
    ).where(df["Última actualización"].notna())
    df["edad_ticket_horas"] = (
        (pd.Timestamp.now().normalize() - df["Fecha de creación"]).dt.total_seconds() / 3600
    ).clip(lower=0)

    df["cumple_sla"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    mascara_sla = df["Fecha de expiración de SLA"].notna()
    df.loc[mascara_sla & df["Fecha de cierre"].notna(), "cumple_sla"] = (
        df.loc[mascara_sla & df["Fecha de cierre"].notna(), "Fecha de cierre"]
        <= df.loc[mascara_sla & df["Fecha de cierre"].notna(), "Fecha de expiración de SLA"]
    )
    df.loc[mascara_sla & df["Fecha de cierre"].isna(), "cumple_sla"] = (
        pd.Timestamp.now() <= df.loc[mascara_sla & df["Fecha de cierre"].isna(), "Fecha de expiración de SLA"]
    )

    df["bucket_edad"] = pd.cut(
        df["edad_ticket_horas"],
        bins=[-1, 8, 24, 48, 96, 999999],
        labels=["0-8 h", "8-24 h", "1-2 dias", "2-4 dias", "4+ dias"],
    ).astype(str)

    return df, ruta


def aplicar_estilos(logo_uri: str | None, fondo_app_uri: str | None, fondo_sidebar_uri: str | None, fondo_hero_uri: str | None) -> None:
    fondo_app_css = (
        f"background-image: linear-gradient(rgba(247,249,252,0.92), rgba(237,242,247,0.95)), url('{fondo_app_uri}');"
        if fondo_app_uri
        else "background: linear-gradient(180deg, #f7f9fc 0%, #edf2f7 100%);"
    )
    fondo_sidebar_css = (
        f"background-image: linear-gradient(rgba(255,255,255,0.96), rgba(255,255,255,0.96)), url('{fondo_sidebar_uri}'); background-size: cover; background-position: center;"
        if fondo_sidebar_uri
        else "background: #ffffff;"
    )
    fondo_hero_css = (
        f"background-image: linear-gradient(rgba(255,255,255,0.88), rgba(248,251,255,0.94)), url('{fondo_hero_uri}'); background-size: cover; background-position: center;"
        if fondo_hero_uri
        else "background: linear-gradient(180deg, #ffffff, #f8fbff);"
    )

    st.markdown(
        f"""
        <style>
        .stApp {{ {fondo_app_css} background-size: cover; background-attachment: fixed; }}
        header[data-testid="stHeader"] {{ display: none; }}
        div[data-testid="stToolbar"] {{ display: none; }}
        .block-container {{ padding-top: 0.45rem; padding-bottom: 1rem; }}
        [data-testid="stSidebar"] {{ {fondo_sidebar_css} border-right: 1px solid #e5e7eb; }}
        [data-testid="stSidebar"] .stSelectbox label {{ font-weight: 700; color: #334155; }}
        [data-testid="stSidebar"] .stSelectbox > div > div {{
            background: #f8fafc;
            border-radius: 12px;
        }}
        [data-testid="stSidebar"] .stButton button {{
            border-radius: 12px;
            border: 1px solid #cbd5e1;
            background: linear-gradient(180deg, #ffffff, #eff6ff);
            font-weight: 700;
            min-height: 42px;
        }}
        .app-topbar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.8rem 1rem;
            margin-bottom: 0.8rem;
            border-radius: 18px;
            border: 1px solid #dbe7f3;
            background: rgba(255,255,255,0.92);
            backdrop-filter: blur(10px);
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }}
        .app-topbar-left {{
            display: flex;
            align-items: center;
            gap: 0.85rem;
            min-width: 0;
        }}
        .app-topbar-logo {{
            width: 44px;
            height: 44px;
            border-radius: 12px;
            background: linear-gradient(180deg, #eff6ff, #dbeafe);
            display: grid;
            place-items: center;
            overflow: hidden;
            flex: 0 0 auto;
        }}
        .app-topbar-logo img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }}
        .app-topbar-copy {{
            min-width: 0;
        }}
        .app-topbar-title {{
            font-size: 1rem;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.1;
        }}
        .app-topbar-subtitle {{
            color: #64748b;
            font-size: 0.82rem;
            margin-top: 0.15rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .app-topbar-right {{
            color: #1d4ed8;
            font-size: 0.78rem;
            font-weight: 700;
            border-radius: 999px;
            padding: 0.4rem 0.7rem;
            border: 1px solid rgba(37, 99, 235, 0.18);
            background: rgba(239,246,255,0.96);
            flex: 0 0 auto;
        }}
        .hero-card {{
            {fondo_hero_css}
            border: 1px solid #dbe7f3;
            border-radius: 22px;
            padding: 1.3rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.06);
        }}
        .hero-layout {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
        }}
        .hero-tag {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.12em; color: #2563eb; font-weight: 800; }}
        .hero-title {{ font-size: clamp(1.8rem, 3vw, 2.35rem); font-weight: 800; color: #0f172a; margin-top: 0.35rem; line-height: 1.04; }}
        .hero-subtitle {{ margin-top: 0.55rem; color: #475569; max-width: 68ch; }}
        .hero-brand {{
            min-width: 150px;
            max-width: 180px;
            display: flex;
            justify-content: center;
            align-items: center;
        }}
        .hero-brand img {{
            max-width: 100%;
            max-height: 64px;
            object-fit: contain;
            filter: drop-shadow(0 12px 24px rgba(15, 23, 42, 0.12));
        }}
        .hero-brand-placeholder {{
            width: 160px;
            min-height: 76px;
            display: grid;
            place-items: center;
            border-radius: 18px;
            border: 1px dashed #93c5fd;
            background: rgba(255,255,255,0.76);
            color: #2563eb;
            font-size: 0.82rem;
            font-weight: 700;
            text-align: center;
            padding: 0.8rem;
        }}
        .sidebar-brand {{
            margin-bottom: 1rem;
            padding: 1rem;
            border-radius: 18px;
            border: 1px solid #dbeafe;
            background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(239,246,255,0.96));
            text-align: center;
        }}
        .sidebar-brand img {{
            max-width: 100%;
            max-height: 52px;
            object-fit: contain;
            margin-bottom: 0.5rem;
        }}
        .sidebar-brand-title {{
            color: #1e3a8a;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-weight: 800;
        }}
        .sidebar-brand-note {{
            color: #64748b;
            font-size: 0.8rem;
            margin-top: 0.3rem;
        }}
        .mini-card {{
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 1.05rem 1.1rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.05);
            min-height: 130px;
        }}
        .mini-card-title {{
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #64748b;
            font-weight: 700;
        }}
        .mini-card-value {{
            font-size: clamp(1.2rem, 2vw, 1.6rem);
            font-weight: 800;
            color: #0f172a;
            margin-top: 0.35rem;
            line-height: 1.05;
        }}
        .mini-card-note {{
            margin-top: 0.55rem;
            color: #64748b;
            font-size: 0.82rem;
            line-height: 1.45;
        }}
        .focus-strip {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin-bottom: 1rem;
        }}
        .focus-chip {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            border: 1px solid #dbeafe;
            border-radius: 999px;
            padding: 0.45rem 0.7rem;
            background: rgba(239,246,255,0.95);
            color: #1d4ed8;
            font-size: 0.78rem;
            font-weight: 700;
        }}
        .section-title {{ font-size: 1rem; font-weight: 800; color: #0f172a; margin-bottom: 0.55rem; }}
        .section-note {{ color: #64748b; font-size: 0.85rem; margin-bottom: 0.5rem; line-height: 1.45; }}
        .section-block {{
            margin-bottom: 1.35rem;
        }}
        .support-table-wrap {{
            background: rgba(255,255,255,0.78);
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 0.35rem;
            margin-top: 0.65rem;
        }}
        .dataset-tag {{
            display: inline-block;
            margin-top: 0.85rem;
            border-radius: 999px;
            background: rgba(37, 99, 235, 0.08);
            border: 1px solid rgba(37, 99, 235, 0.18);
            color: #1d4ed8;
            padding: 0.35rem 0.7rem;
            font-size: 0.76rem;
            font-weight: 700;
        }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 10px; flex-wrap: wrap; }}
        .stTabs [data-baseweb="tab"] {{
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 0.55rem 1rem;
            box-shadow: 0 4px 10px rgba(15, 23, 42, 0.04);
        }}
        .stTabs [aria-selected="true"] {{ color: #2563eb; border-color: #bfdbfe; background: #eff6ff; }}
        div[data-testid="stPlotlyChart"], div[data-testid="stDataFrame"] {{
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 20px;
            padding: 0.55rem;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.05);
        }}
        div[data-testid="stDataFrame"] {{ overflow-x: auto; }}
        [data-testid="column"] {{ min-width: 0; }}
        @media (max-width: 1200px) {{
            .block-container {{ padding-left: 1.05rem; padding-right: 1.05rem; }}
            .hero-card {{ padding: 1.15rem 1.2rem; }}
        }}
        @media (max-width: 900px) {{
            .app-topbar {{ flex-direction: column; align-items: flex-start; }}
            .app-topbar-right {{ width: 100%; text-align: center; }}
            .hero-layout {{ flex-direction: column; align-items: flex-start; }}
            .hero-brand {{ width: 100%; justify-content: flex-start; }}
            .mini-card {{ min-height: auto; }}
            .stTabs [data-baseweb="tab"] {{ width: calc(50% - 6px); justify-content: center; }}
        }}
        @media (max-width: 640px) {{
            .block-container {{ padding-top: 0.6rem; padding-left: 0.7rem; padding-right: 0.7rem; }}
            .app-topbar {{
                padding: 0.7rem 0.8rem;
                border-radius: 14px;
            }}
            .app-topbar-title {{ font-size: 0.94rem; }}
            .app-topbar-subtitle {{ font-size: 0.75rem; white-space: normal; }}
            .hero-title {{ font-size: 1.55rem; }}
            .hero-subtitle {{ font-size: 0.92rem; }}
            .hero-brand-placeholder {{ width: 100%; min-height: 62px; font-size: 0.76rem; }}
            .sidebar-brand {{ padding: 0.8rem; border-radius: 14px; }}
            .mini-card-note {{ font-size: 0.76rem; }}
            .stTabs [data-baseweb="tab"] {{ width: 100%; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def formato_entero(valor: float) -> str:
    return f"{int(valor):,}".replace(",", ".")


def formato_horas(valor: float | int | None) -> str:
    if valor is None or pd.isna(valor):
        return "-"
    return f"{valor:,.1f} h".replace(",", ".")


def render_mini_card(titulo: str, valor: str, nota: str) -> None:
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="mini-card-title">{titulo}</div>
            <div class="mini-card-value">{valor}</div>
            <div class="mini-card-note">{nota}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def estilizar_figura(fig: go.Figure, height: int = 360, showlegend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=18, b=18),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=showlegend,
        hoverlabel=dict(
            bgcolor="#0f172a",
            bordercolor="#2563eb",
            font=dict(color="#f8fafc", size=13, family="Segoe UI"),
            namelength=-1,
        ),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.18)", zeroline=False)
    return fig


def render_support_table(df: pd.DataFrame, columns: list[str], title: str) -> None:
    st.markdown(f'<div class="section-note">{title}</div>', unsafe_allow_html=True)
    st.markdown('<div class="support-table-wrap">', unsafe_allow_html=True)
    st.dataframe(df[columns], use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def set_vista_kpi(valor: str) -> None:
    st.session_state["vista_kpi"] = valor


def limpiar_filtros() -> None:
    st.session_state[FILTRO_KEYS["departamento"]] = "Todos"
    st.session_state[FILTRO_KEYS["tema"]] = "Todos"
    st.session_state[FILTRO_KEYS["fuente"]] = "Todos"
    st.session_state[FILTRO_KEYS["estado"]] = "Todos"
    st.session_state[FILTRO_KEYS["agente"]] = "Todos"
    st.session_state[FILTRO_KEYS["mes"]] = "Todos"
    st.session_state["vista_kpi"] = "Todos"


def asegurar_estado() -> None:
    for key in FILTRO_KEYS.values():
        if key not in st.session_state:
            st.session_state[key] = "Todos"
    if "vista_kpi" not in st.session_state:
        st.session_state["vista_kpi"] = "Todos"
    for key in PENDING_KEYS.values():
        if key not in st.session_state:
            st.session_state[key] = None


def aplicar_filtros_pendientes() -> None:
    if st.session_state[PENDING_KEYS["departamento"]]:
        st.session_state[FILTRO_KEYS["departamento"]] = st.session_state[PENDING_KEYS["departamento"]]
        st.session_state[PENDING_KEYS["departamento"]] = None
    if st.session_state[PENDING_KEYS["agente"]]:
        st.session_state[FILTRO_KEYS["agente"]] = st.session_state[PENDING_KEYS["agente"]]
        st.session_state[PENDING_KEYS["agente"]] = None


def filtrar_tickets(
    df: pd.DataFrame,
    departamento: str,
    tema: str,
    fuente: str,
    estado: str,
    agente: str,
    mes: str,
) -> pd.DataFrame:
    mascara = pd.Series(True, index=df.index)
    if departamento != "Todos":
        mascara &= df["Departamento"] == departamento
    if tema != "Todos":
        mascara &= df["Temas de ayuda"] == tema
    if fuente != "Todos":
        mascara &= df["Fuente"] == fuente
    if estado != "Todos":
        mascara &= df["Estado actual"] == estado
    if agente != "Todos":
        mascara &= df["Agente asignado"] == agente
    if mes != "Todos":
        mascara &= df["mes"] == mes
    return df.loc[mascara].copy()


def aplicar_vista(df: pd.DataFrame, vista: str) -> pd.DataFrame:
    if vista == "Abiertos":
        return df[df["abierto_bool"]].copy()
    if vista == "Cerrados":
        return df[df["cerrado_bool"]].copy()
    if vista == "Atrasados":
        return df[df["atrasado_bool"]].copy()
    if vista == "Respuesta":
        return df[df["respondio_bool"]].copy()
    if vista == "MTTR":
        return df[df["tiempo_resolucion_horas"].notna()].copy()
    return df.copy()


tickets_df, dataset_path = cargar_tickets()
logo_uri = imagen_a_data_uri(BRANDING_ASSETS["logo"])
fondo_app_uri = imagen_a_data_uri(BRANDING_ASSETS["fondo_app"])
fondo_sidebar_uri = imagen_a_data_uri(BRANDING_ASSETS["fondo_sidebar"])
fondo_hero_uri = imagen_a_data_uri(BRANDING_ASSETS["fondo_hero"])
aplicar_estilos(logo_uri, fondo_app_uri, fondo_sidebar_uri, fondo_hero_uri)
asegurar_estado()
aplicar_filtros_pendientes()

if tickets_df.empty:
    st.error("No se encontro un archivo CSV de tickets dentro de la carpeta datasets.")
    st.stop()

st.markdown(
    f"""
    <div class="app-topbar">
        <div class="app-topbar-left">
            <div class="app-topbar-logo">
                {f'<img src="{logo_uri}" alt="Logo corporativo" />' if logo_uri else '<span style="font-weight:800;color:#2563eb;">TI</span>'}
            </div>
            <div class="app-topbar-copy">
                <div class="app-topbar-title">Centro de soporte tecnologico</div>
                <div class="app-topbar-subtitle">Operacion de tickets, SLA, agentes y backlog del area TI</div>
            </div>
        </div>
        <div class="app-topbar-right">{dataset_path.name if dataset_path else "Sin dataset"}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-brand">
            {f'<img src="{logo_uri}" alt="Logo corporativo" />' if logo_uri else ""}
            </div>
        """,
        unsafe_allow_html=True,
    )

    st.header("Filtros")

    deptos = ["Todos"] + sorted(tickets_df["Departamento"].dropna().unique().tolist())
    if st.session_state[FILTRO_KEYS["departamento"]] not in deptos:
        st.session_state[FILTRO_KEYS["departamento"]] = "Todos"
    departamento_sel = st.selectbox("Departamento", deptos, key=FILTRO_KEYS["departamento"])

    base_tema = tickets_df if departamento_sel == "Todos" else tickets_df[tickets_df["Departamento"] == departamento_sel]
    temas = ["Todos"] + sorted(base_tema["Temas de ayuda"].dropna().unique().tolist())
    if st.session_state[FILTRO_KEYS["tema"]] not in temas:
        st.session_state[FILTRO_KEYS["tema"]] = "Todos"
    tema_sel = st.selectbox("Tema de ayuda", temas, key=FILTRO_KEYS["tema"])

    fuentes = ["Todos"] + sorted(tickets_df["Fuente"].dropna().unique().tolist())
    if st.session_state[FILTRO_KEYS["fuente"]] not in fuentes:
        st.session_state[FILTRO_KEYS["fuente"]] = "Todos"
    fuente_sel = st.selectbox("Fuente", fuentes, key=FILTRO_KEYS["fuente"])

    estados = ["Todos"] + sorted(tickets_df["Estado actual"].dropna().unique().tolist())
    if st.session_state[FILTRO_KEYS["estado"]] not in estados:
        st.session_state[FILTRO_KEYS["estado"]] = "Todos"
    estado_sel = st.selectbox("Estado", estados, key=FILTRO_KEYS["estado"])

    agentes = ["Todos"] + sorted(tickets_df["Agente asignado"].dropna().unique().tolist())
    if st.session_state[FILTRO_KEYS["agente"]] not in agentes:
        st.session_state[FILTRO_KEYS["agente"]] = "Todos"
    agente_sel = st.selectbox("Agente asignado", agentes, key=FILTRO_KEYS["agente"])

    meses = ["Todos"] + sorted(tickets_df["mes"].dropna().unique().tolist())
    if st.session_state[FILTRO_KEYS["mes"]] not in meses:
        st.session_state[FILTRO_KEYS["mes"]] = "Todos"
    mes_sel = st.selectbox("Mes de creacion", meses, key=FILTRO_KEYS["mes"])

    st.button("Limpiar filtros", use_container_width=True, on_click=limpiar_filtros)


filtrado = filtrar_tickets(tickets_df, departamento_sel, tema_sel, fuente_sel, estado_sel, agente_sel, mes_sel)
vista_actual = st.session_state.get("vista_kpi", "Todos")
vista_df = aplicar_vista(filtrado, vista_actual)
analisis_df = vista_df

total_tickets = len(analisis_df)
abiertos = int(analisis_df["abierto_bool"].sum())
cerrados = int(analisis_df["cerrado_bool"].sum())
atrasados = int(analisis_df["atrasado_bool"].sum())
respondidos = int(analisis_df["respondio_bool"].sum())
tasa_respuesta = (respondidos / total_tickets * 100) if total_tickets else 0
cumple_sla = analisis_df["cumple_sla"].dropna()
tasa_sla = (cumple_sla.mean() * 100) if not cumple_sla.empty else 0
mttr = analisis_df["tiempo_resolucion_horas"].dropna().mean()
tiempo_primera_actividad = analisis_df["tiempo_ultima_actualizacion_horas"].dropna().mean()

serie_dia = (
    analisis_df.groupby(["dia", "Estado actual"], as_index=False)["Número de Ticket"]
    .count()
    .rename(columns={"Número de Ticket": "tickets"})
)
serie_mes = (
    analisis_df.groupby("mes", as_index=False)["Número de Ticket"]
    .count()
    .rename(columns={"Número de Ticket": "tickets"})
    .sort_values("mes")
)
creados_mes = (
    filtrado.groupby("mes", as_index=False)["Número de Ticket"]
    .count()
    .rename(columns={"Número de Ticket": "creados"})
)
cerrados_mes = (
    filtrado[filtrado["Fecha de cierre"].notna()]
    .assign(mes_cierre=lambda df: df["Fecha de cierre"].dt.to_period("M").astype(str))
    .groupby("mes_cierre", as_index=False)["Número de Ticket"]
    .count()
    .rename(columns={"mes_cierre": "mes", "Número de Ticket": "cerrados"})
)
cumplimiento_mes = creados_mes.merge(cerrados_mes, on="mes", how="outer").fillna(0).sort_values("mes")
cumplimiento_mes["creados"] = cumplimiento_mes["creados"].astype(int)
cumplimiento_mes["cerrados"] = cumplimiento_mes["cerrados"].astype(int)
cumplimiento_mes["tasa_cierre"] = (
    (cumplimiento_mes["cerrados"] / cumplimiento_mes["creados"].replace(0, pd.NA)) * 100
).fillna(0).round(1)

serie_dia_semana = (
    analisis_df.assign(dia_semana=analisis_df["Fecha de creación"].dt.day_name())
    .groupby("dia_semana", as_index=False)["Número de Ticket"]
    .count()
    .rename(columns={"Número de Ticket": "tickets"})
)
orden_semana = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
serie_dia_semana["orden"] = serie_dia_semana["dia_semana"].map({dia: i for i, dia in enumerate(orden_semana)})
serie_dia_semana = serie_dia_semana.sort_values("orden")
serie_solicitante = (
    analisis_df.groupby("De", as_index=False)["Número de Ticket"]
    .count()
    .rename(columns={"Número de Ticket": "tickets"})
    .sort_values("tickets", ascending=False)
    .head(15)
)
serie_prioridad = (
    analisis_df.groupby("Prioridad", as_index=False)["Número de Ticket"]
    .count()
    .rename(columns={"Número de Ticket": "tickets"})
    .sort_values("tickets", ascending=False)
)
serie_departamento = (
    analisis_df.groupby("Departamento", as_index=False)["Número de Ticket"]
    .count()
    .rename(columns={"Número de Ticket": "tickets"})
    .sort_values("tickets", ascending=False)
)
serie_fuente = (
    analisis_df.groupby("Fuente", as_index=False)["Número de Ticket"]
    .count()
    .rename(columns={"Número de Ticket": "tickets"})
    .sort_values("tickets", ascending=False)
)
serie_estado = (
    analisis_df.groupby("Estado actual", as_index=False)["Número de Ticket"]
    .count()
    .rename(columns={"Número de Ticket": "tickets"})
    .sort_values("tickets", ascending=False)
)
serie_tema = (
    analisis_df.groupby("Temas de ayuda", as_index=False)["Número de Ticket"]
    .count()
    .rename(columns={"Número de Ticket": "tickets"})
    .sort_values("tickets", ascending=False)
    .head(10)
)
serie_sla_depto = (
    analisis_df.assign(
        estado_sla=analisis_df["cumple_sla"].map({True: "Cumple SLA", False: "Incumple SLA"}).fillna("Sin dato SLA")
    )
    .groupby(["Departamento", "estado_sla"], as_index=False)["Número de Ticket"]
    .count()
    .rename(columns={"Número de Ticket": "tickets"})
)
serie_edad = (
    analisis_df.groupby("bucket_edad", as_index=False)["Número de Ticket"]
    .count()
    .rename(columns={"Número de Ticket": "tickets"})
)

agentes_df = (
    analisis_df.groupby("Agente asignado", as_index=False)
    .agg(
        tickets=("Número de Ticket", "count"),
        abiertos=("abierto_bool", "sum"),
        cerrados=("cerrado_bool", "sum"),
        atrasados=("atrasado_bool", "sum"),
        mttr=("tiempo_resolucion_horas", "mean"),
        respuesta=("respondio_bool", "mean"),
    )
    .sort_values("tickets", ascending=False)
)

contexto_agente = (
    analisis_df.groupby("Agente asignado", as_index=False)["Número de Ticket"]
    .count()
    .rename(columns={"Número de Ticket": "tickets"})
    .sort_values("tickets", ascending=False)
    .head(10)
)
contexto_dia = (
    analisis_df.groupby("dia", as_index=False)["Número de Ticket"]
    .count()
    .rename(columns={"Número de Ticket": "tickets"})
)
contexto_tema = (
    analisis_df.groupby("Temas de ayuda", as_index=False)["Número de Ticket"]
    .count()
    .rename(columns={"Número de Ticket": "tickets"})
    .sort_values("tickets", ascending=False)
    .head(8)
)
contexto_observacion = analisis_df["Asunto"].fillna("").astype(str).str.upper().value_counts().head(8).reset_index()
contexto_observacion.columns = ["Asunto", "tickets"]

tabs = st.tabs(["Resumen Ejecutivo", "Tendencias", "Operacion y SLA", "Agentes y Solicitantes", "Exploracion", "Tablas", "Detalle"])

with tabs[0]:
    vista_cols = st.columns(6)
    vistas = [
        ("Todos", f"Todos ({formato_entero(len(filtrado))})"),
        ("Abiertos", f"Abiertos ({formato_entero(int(filtrado['abierto_bool'].sum()))})"),
        ("Cerrados", f"Cerrados ({formato_entero(int(filtrado['cerrado_bool'].sum()))})"),
        ("Atrasados", f"Atrasados ({formato_entero(int(filtrado['atrasado_bool'].sum()))})"),
        ("Respuesta", f"Con respuesta ({formato_entero(int(filtrado['respondio_bool'].sum()))})"),
        ("MTTR", "Con cierre"),
    ]
    for col, (clave_vista, label) in zip(vista_cols, vistas):
        with col:
            st.button(label, key=f"vista_btn_{clave_vista}", use_container_width=True, on_click=set_vista_kpi, args=(clave_vista,))

    st.markdown(
        f"""
        <div class="focus-strip">
            <div class="focus-chip">Vista activa: {vista_actual}</div>
            <div class="focus-chip">Tickets mostrados: {formato_entero(total_tickets)}</div>
            <div class="focus-chip">Departamento: {departamento_sel}</div>
            <div class="focus-chip">Agente: {agente_sel}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_row = st.columns([1.55, 1])
    with top_row[0]:
        st.markdown('<div class="section-title">Flujo diario de tickets</div><div class="section-note">Comportamiento de entradas y estados a lo largo del tiempo.</div>', unsafe_allow_html=True)
        fig_dia = px.area(
            serie_dia,
            x="dia",
            y="tickets",
            color="Estado actual",
            color_discrete_map={"Cerrado": "#0f766e", "Abierto": "#2563eb", "En Progreso": "#f97316"},
        )
        estilizar_figura(fig_dia, height=420, showlegend=True)
        fig_dia.update_layout(legend_title_text="", clickmode="event+select")
        st.plotly_chart(fig_dia, use_container_width=True)
    with top_row[1]:
        st.markdown('<div class="section-title">Canales de ingreso</div><div class="section-note">Distribucion de tickets por fuente de contacto.</div>', unsafe_allow_html=True)
        fig_fuente = px.pie(
            serie_fuente,
            names="Fuente",
            values="tickets",
            hole=0.62,
            color_discrete_sequence=["#2563eb", "#0f766e", "#f97316", "#7c3aed"],
        )
        estilizar_figura(fig_fuente, height=420, showlegend=True)
        st.plotly_chart(fig_fuente, use_container_width=True)

    mid_row = st.columns([1.1, 1])
    with mid_row[0]:
        st.markdown('<div class="section-title">Volumen por departamento</div><div class="section-note">Carga operativa por unidad de tecnologia.</div>', unsafe_allow_html=True)
        fig_depto = px.bar(
            serie_departamento,
            x="Departamento",
            y="tickets",
            color="Departamento",
            color_discrete_sequence=["#0f766e", "#2563eb", "#f97316", "#7c3aed"],
            text_auto=True,
        )
        estilizar_figura(fig_depto, height=390, showlegend=False)
        fig_depto.update_layout(clickmode="event+select")
        evento_depto = st.plotly_chart(fig_depto, use_container_width=True, key="chart_departamento", on_select="rerun")
        if evento_depto and evento_depto.selection and evento_depto.selection.get("points"):
            punto = evento_depto.selection["points"][0]
            if "x" in punto:
                st.session_state[PENDING_KEYS["departamento"]] = punto["x"]
                st.rerun()
    with mid_row[1]:
        st.markdown('<div class="section-title">Tickets por agente</div><div class="section-note">Selecciona un agente para filtrar la vista completa.</div>', unsafe_allow_html=True)
        fig_ctx_agente = px.bar(
            contexto_agente,
            x="Agente asignado",
            y="tickets",
            color="tickets",
            color_continuous_scale="Blues",
            text_auto=True,
        )
        estilizar_figura(fig_ctx_agente, height=390, showlegend=False)
        fig_ctx_agente.update_layout(coloraxis_showscale=False)
        evento_agente = st.plotly_chart(fig_ctx_agente, use_container_width=True, key="chart_ctx_agente", on_select="rerun")
        if evento_agente and evento_agente.selection and evento_agente.selection.get("points"):
            punto = evento_agente.selection["points"][0]
            if "x" in punto:
                st.session_state[PENDING_KEYS["agente"]] = punto["x"]
                st.rerun()

    bottom_row = st.columns([1.05, 1])
    with bottom_row[0]:
        st.markdown('<div class="section-title">Ritmo diario de la vista actual</div><div class="section-note">Seguimiento del subconjunto activo tras aplicar filtros y vista.</div>', unsafe_allow_html=True)
        fig_ctx_dia = px.line(contexto_dia, x="dia", y="tickets", markers=True)
        estilizar_figura(fig_ctx_dia, height=360, showlegend=False)
        st.plotly_chart(fig_ctx_dia, use_container_width=True)
    with bottom_row[1]:
        st.markdown('<div class="section-title">Temas dominantes</div><div class="section-note">Concentracion de tickets por tema de ayuda en la vista actual.</div>', unsafe_allow_html=True)
        fig_ctx_tema = px.bar(
            contexto_tema,
            x="tickets",
            y="Temas de ayuda",
            orientation="h",
            color="tickets",
            color_continuous_scale="Oranges",
            text_auto=True,
        )
        estilizar_figura(fig_ctx_tema, height=360, showlegend=False)
        fig_ctx_tema.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_ctx_tema, use_container_width=True)
with tabs[1]:
    st.markdown('<div class="section-title">Tendencias y estacionalidad</div><div class="section-note">Explora la evolucion temporal, picos de demanda y periodos con mayor carga de soporte.</div>', unsafe_allow_html=True)
    trend_tabs = st.tabs(["Por mes", "Por dia", "Por estado", "Cumplimiento"])
    with trend_tabs[0]:
        fig_mes = px.line(serie_mes, x="mes", y="tickets", markers=True)
        estilizar_figura(fig_mes, height=380, showlegend=False)
        st.plotly_chart(fig_mes, use_container_width=True)
    with trend_tabs[1]:
        fig_semana = px.bar(serie_dia_semana, x="dia_semana", y="tickets", color="tickets", color_continuous_scale="Blues", text_auto=True)
        estilizar_figura(fig_semana, height=380, showlegend=False)
        fig_semana.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_semana, use_container_width=True)
    with trend_tabs[2]:
        estado_mes_df = analisis_df.groupby(["mes", "Estado actual"], as_index=False)["Número de Ticket"].count().rename(columns={"Número de Ticket": "tickets"})
        fig_estado_mes = px.bar(
            estado_mes_df,
            x="mes",
            y="tickets",
            color="Estado actual",
            barmode="stack",
            color_discrete_map={"Cerrado": "#0f766e", "Abierto": "#2563eb", "En Progreso": "#f97316"},
        )
        estilizar_figura(fig_estado_mes, height=390, showlegend=True)
        st.plotly_chart(fig_estado_mes, use_container_width=True)
    with trend_tabs[3]:
        fig_cumplimiento = go.Figure()
        fig_cumplimiento.add_bar(name="Creados", x=cumplimiento_mes["mes"], y=cumplimiento_mes["creados"], marker_color="#2563eb")
        fig_cumplimiento.add_bar(name="Cerrados", x=cumplimiento_mes["mes"], y=cumplimiento_mes["cerrados"], marker_color="#0f766e")
        fig_cumplimiento.add_trace(
            go.Scatter(
                name="Tasa de cierre %",
                x=cumplimiento_mes["mes"],
                y=cumplimiento_mes["tasa_cierre"],
                yaxis="y2",
                mode="lines+markers",
                line=dict(color="#f97316", width=3),
                marker=dict(size=8),
            )
        )
        estilizar_figura(fig_cumplimiento, height=400, showlegend=True)
        fig_cumplimiento.update_layout(
            barmode="group",
            legend_title_text="",
            yaxis_title="Tickets",
            yaxis2=dict(title="Tasa %", overlaying="y", side="right", showgrid=False, rangemode="tozero"),
        )
        st.plotly_chart(fig_cumplimiento, use_container_width=True)

        tabla_cumplimiento_mes = cumplimiento_mes.copy()
        tabla_cumplimiento_mes["brecha"] = tabla_cumplimiento_mes["creados"] - tabla_cumplimiento_mes["cerrados"]
        tabla_cumplimiento_mes["tasa_cierre_label"] = tabla_cumplimiento_mes["tasa_cierre"].map(lambda v: f"{v:.1f}%")
        render_support_table(
            tabla_cumplimiento_mes.sort_values("mes", ascending=False),
            ["mes", "creados", "cerrados", "brecha", "tasa_cierre_label"],
            "Cumplimiento por mes",
        )

        creados_total_periodo = int(cumplimiento_mes["creados"].sum()) if not cumplimiento_mes.empty else 0
        cerrados_total_periodo = int(cumplimiento_mes["cerrados"].sum()) if not cumplimiento_mes.empty else 0
        tasa_global_cierre = (cerrados_total_periodo / creados_total_periodo * 100) if creados_total_periodo else 0
        brecha_total = creados_total_periodo - cerrados_total_periodo
        mejor_mes = (
            cumplimiento_mes.sort_values(["tasa_cierre", "cerrados"], ascending=[False, False]).iloc[0]
            if not cumplimiento_mes.empty
            else None
        )
        peor_mes = (
            cumplimiento_mes.sort_values(["tasa_cierre", "creados"], ascending=[True, False]).iloc[0]
            if not cumplimiento_mes.empty
            else None
        )

        resumen_cols = st.columns(4)
        with resumen_cols[0]:
            render_mini_card("Creados periodo", formato_entero(creados_total_periodo), "Total de tickets generados en los meses visibles")
        with resumen_cols[1]:
            render_mini_card("Cerrados periodo", formato_entero(cerrados_total_periodo), "Total de tickets cerrados en el mismo periodo")
        with resumen_cols[2]:
            render_mini_card("Tasa global", f"{tasa_global_cierre:.1f}%", "Relacion entre tickets cerrados y creados")
        with resumen_cols[3]:
            render_mini_card("Brecha neta", formato_entero(brecha_total), "Diferencia entre entradas y cierres")

        if mejor_mes is not None and peor_mes is not None:
            st.markdown(
                f"""
                <div class="section-block">
                    <div class="section-title">Lectura del cumplimiento mensual</div>
                    <div class="section-note">
                        En el periodo filtrado se crearon <b>{formato_entero(creados_total_periodo)}</b> tickets y se cerraron <b>{formato_entero(cerrados_total_periodo)}</b>,
                        lo que deja una tasa global de cierre de <b>{tasa_global_cierre:.1f}%</b> y una brecha neta de <b>{formato_entero(brecha_total)}</b> tickets.
                        El mejor mes fue <b>{mejor_mes["mes"]}</b> con una tasa de <b>{mejor_mes["tasa_cierre"]:.1f}%</b>,
                        mientras que el mes con menor cobertura fue <b>{peor_mes["mes"]}</b> con <b>{peor_mes["tasa_cierre"]:.1f}%</b>.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

with tabs[2]:
    cols = st.columns(4)
    with cols[0]:
        render_mini_card("Cumplimiento SLA", f"{tasa_sla:.1f}%", "Tickets con fecha SLA dentro del objetivo")
    with cols[1]:
        render_mini_card("Primera actividad", formato_horas(tiempo_primera_actividad), "Tiempo medio hasta la ultima actualizacion inicial")
    with cols[2]:
        render_mini_card("Temas distintos", formato_entero(analisis_df["Temas de ayuda"].nunique()), "Catalogo activo en el filtro")
    with cols[3]:
        render_mini_card("Pendientes", formato_entero(abiertos + atrasados), "Backlog critico + vencidos")

    ops_tabs = st.tabs(["Temas", "Antiguedad", "SLA"])
    with ops_tabs[0]:
        st.markdown('<div class="section-title">Temas de ayuda mas recurrentes</div><div class="section-note">Incidencias con mayor repeticion para priorizar automatizacion o correccion raiz.</div>', unsafe_allow_html=True)
        fig_tema = px.bar(
            serie_tema,
            x="tickets",
            y="Temas de ayuda",
            orientation="h",
            color="tickets",
            color_continuous_scale="Blues",
            text_auto=True,
        )
        estilizar_figura(fig_tema, height=400, showlegend=False)
        fig_tema.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_tema, use_container_width=True)
    with ops_tabs[1]:
        st.markdown('<div class="section-title">Backlog por antiguedad</div><div class="section-note">Edad operativa de los tickets dentro del universo filtrado.</div>', unsafe_allow_html=True)
        fig_edad = px.bar(
            serie_edad,
            x="bucket_edad",
            y="tickets",
            color="bucket_edad",
            color_discrete_sequence=["#bfdbfe", "#60a5fa", "#2563eb", "#f97316", "#dc2626"],
            text_auto=True,
        )
        estilizar_figura(fig_edad, height=400, showlegend=False)
        st.plotly_chart(fig_edad, use_container_width=True)
    with ops_tabs[2]:
        st.markdown('<div class="section-title">SLA por departamento</div><div class="section-note">Comparativo entre cumplimiento, incumplimiento y casos sin dato de SLA.</div>', unsafe_allow_html=True)
        fig_sla = px.bar(
            serie_sla_depto,
            x="Departamento",
            y="tickets",
            color="estado_sla",
            barmode="stack",
            color_discrete_map={"Cumple SLA": "#0f766e", "Incumple SLA": "#dc2626", "Sin dato SLA": "#94a3b8"},
            text_auto=True,
        )
        estilizar_figura(fig_sla, height=390, showlegend=True)
        st.plotly_chart(fig_sla, use_container_width=True)

with tabs[3]:
    cols = st.columns(3)
    with cols[0]:
        render_mini_card("Agentes activos", formato_entero(agentes_df["Agente asignado"].nunique()), "Personas con tickets en el filtro actual")
    with cols[1]:
        render_mini_card("Top agente", agentes_df.iloc[0]["Agente asignado"] if not agentes_df.empty else "-", "Mayor volumen de tickets")
    with cols[2]:
        render_mini_card("Respuesta media", f"{(agentes_df['respuesta'].mean() * 100 if not agentes_df.empty else 0):.1f}%", "Promedio de tickets respondidos por agente")

    people_tabs = st.tabs(["Agentes", "Solicitantes", "Balance"])
    with people_tabs[0]:
        st.markdown('<div class="section-title">Carga operativa por agente</div><div class="section-note">Volumen de tickets asignados para identificar concentracion de demanda.</div>', unsafe_allow_html=True)
        fig_agente = px.bar(
            agentes_df.head(12),
            x="Agente asignado",
            y="tickets",
            color="atrasados",
            color_continuous_scale="Oranges",
            text_auto=True,
        )
        estilizar_figura(fig_agente, height=390, showlegend=False)
        fig_agente.update_layout(coloraxis_colorbar_title="Atrasados")
        st.plotly_chart(fig_agente, use_container_width=True)
        st.markdown('<div class="section-title">Resolucion vs backlog</div><div class="section-note">Comparativo de tickets cerrados y abiertos por agente.</div>', unsafe_allow_html=True)
        fig_balance = go.Figure()
        fig_balance.add_bar(name="Cerrados", x=agentes_df["Agente asignado"], y=agentes_df["cerrados"], marker_color="#0f766e")
        fig_balance.add_bar(name="Abiertos", x=agentes_df["Agente asignado"], y=agentes_df["abiertos"], marker_color="#2563eb")
        estilizar_figura(fig_balance, height=390, showlegend=True)
        fig_balance.update_layout(barmode="group", legend_title_text="")
        st.plotly_chart(fig_balance, use_container_width=True)
    with people_tabs[1]:
        st.markdown('<div class="section-title">Solicitantes con mayor volumen</div><div class="section-note">Usuarios o areas que generan mas tickets y concentran la demanda de soporte.</div>', unsafe_allow_html=True)
        fig_solicitante = px.bar(
            serie_solicitante,
            x="De",
            y="tickets",
            color="tickets",
            color_continuous_scale="Tealgrn",
            text_auto=True,
        )
        estilizar_figura(fig_solicitante, height=400, showlegend=False)
        fig_solicitante.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_solicitante, use_container_width=True)
    with people_tabs[2]:
        st.markdown('<div class="section-title">Balance operativo de agentes</div><div class="section-note">Cruce entre tickets cerrados, abiertos y atrasados para detectar sobrecarga o baja resolucion.</div>', unsafe_allow_html=True)
        fig_balance_scatter = px.scatter(
            agentes_df.head(15),
            x="cerrados",
            y="abiertos",
            size="tickets",
            color="atrasados",
            hover_name="Agente asignado",
            color_continuous_scale="Oranges",
        )
        estilizar_figura(fig_balance_scatter, height=390, showlegend=False)
        fig_balance_scatter.update_layout(coloraxis_colorbar_title="Atrasados")
        st.plotly_chart(fig_balance_scatter, use_container_width=True)

with tabs[4]:
    st.markdown('<div class="section-title">Exploracion profunda</div><div class="section-note">Analisis cruzado de departamentos, fuentes, prioridades, temas y observaciones del contexto actual.</div>', unsafe_allow_html=True)
    deep_tabs = st.tabs(["Estados y prioridad", "Temas y asuntos", "Tablas"])
    with deep_tabs[0]:
        ex1, ex2 = st.columns(2)
        with ex1:
            fig_estado = px.bar(serie_estado, x="Estado actual", y="tickets", color="Estado actual", text_auto=True, color_discrete_map={"Cerrado": "#0f766e", "Abierto": "#2563eb", "En Progreso": "#f97316"})
            estilizar_figura(fig_estado, height=360, showlegend=False)
            st.plotly_chart(fig_estado, use_container_width=True)
        with ex2:
            fig_prioridad = px.bar(serie_prioridad, x="Prioridad", y="tickets", color="Prioridad", text_auto=True)
            estilizar_figura(fig_prioridad, height=360, showlegend=False)
            st.plotly_chart(fig_prioridad, use_container_width=True)
    with deep_tabs[1]:
        ex3, ex4 = st.columns([1.05, 1])
        with ex3:
            fig_tema_ctx = px.bar(contexto_tema, x="tickets", y="Temas de ayuda", orientation="h", color="tickets", color_continuous_scale="Oranges", text_auto=True)
            estilizar_figura(fig_tema_ctx, height=380, showlegend=False)
            fig_tema_ctx.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_tema_ctx, use_container_width=True)
        with ex4:
            fig_obs = px.bar(contexto_observacion, x="tickets", y="Asunto", orientation="h", color="tickets", color_continuous_scale="Purples", text_auto=True)
            estilizar_figura(fig_obs, height=380, showlegend=False)
            fig_obs.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_obs, use_container_width=True)
    with deep_tabs[2]:
        st.info("Las tablas de apoyo se movieron a la pestaña Tablas.")

with tabs[5]:
    st.markdown('<div class="section-title">Tablas</div><div class="section-note">Todas las tablas de soporte del analisis consolidadas en un solo lugar.</div>', unsafe_allow_html=True)
    table_tabs = st.tabs(["Resumen", "Tendencias", "SLA", "Personas", "Temas"])
    with table_tabs[0]:
        detalle_contexto = analisis_df.copy().sort_values("Fecha de creación", ascending=False).head(20)
        detalle_contexto["Fecha de creación"] = detalle_contexto["Fecha de creación"].dt.strftime("%Y-%m-%d %H:%M")
        detalle_contexto["Última actualización"] = detalle_contexto["Última actualización"].dt.strftime("%Y-%m-%d %H:%M")
        render_support_table(serie_fuente, ["Fuente", "tickets"], "Entradas por canal")
        render_support_table(serie_estado, ["Estado actual", "tickets"], "Resumen por estado")
        render_support_table(detalle_contexto, ["Número de Ticket", "Fecha de creación", "Asunto", "Departamento", "Temas de ayuda", "Estado actual", "Agente asignado", "Atrasado"], "Ultimos tickets de la vista actual")
    with table_tabs[1]:
        estado_mes_df = analisis_df.groupby(["mes", "Estado actual"], as_index=False)["Número de Ticket"].count().rename(columns={"Número de Ticket": "tickets"})
        render_support_table(serie_mes.sort_values("mes", ascending=False), ["mes", "tickets"], "Tickets por mes")
        render_support_table(cumplimiento_mes.sort_values("mes", ascending=False), ["mes", "creados", "cerrados", "tasa_cierre"], "Creados vs cerrados por mes")
        render_support_table(serie_dia_semana[["dia_semana", "tickets"]], ["dia_semana", "tickets"], "Tickets por dia")
        render_support_table(estado_mes_df.sort_values(["mes", "tickets"], ascending=[False, False]), ["mes", "Estado actual", "tickets"], "Tickets por mes y estado")
    with table_tabs[2]:
        render_support_table(serie_edad, ["bucket_edad", "tickets"], "Backlog por antiguedad")
        render_support_table(serie_sla_depto.sort_values("tickets", ascending=False), ["Departamento", "estado_sla", "tickets"], "Detalle SLA por departamento")
    with table_tabs[3]:
        detalle_agentes = agentes_df.copy()
        detalle_agentes["respuesta"] = (detalle_agentes["respuesta"] * 100).round(1).astype(str) + "%"
        detalle_agentes["mttr"] = detalle_agentes["mttr"].apply(formato_horas)
        render_support_table(detalle_agentes, ["Agente asignado", "tickets", "abiertos", "cerrados", "atrasados", "mttr", "respuesta"], "Desempeno por agente")
        render_support_table(serie_solicitante, ["De", "tickets"], "Top solicitantes")
    with table_tabs[4]:
        render_support_table(serie_tema, ["Temas de ayuda", "tickets"], "Top de temas")
        render_support_table(contexto_tema, ["Temas de ayuda", "tickets"], "Temas en la vista actual")
        render_support_table(contexto_observacion, ["Asunto", "tickets"], "Asuntos frecuentes")

with tabs[6]:
    st.markdown('<div class="section-title">Detalle de tickets</div><div class="section-note">Universo completo de incidencias filtradas para revision operativa.</div>', unsafe_allow_html=True)

    top_tema = serie_tema.iloc[0]["Temas de ayuda"] if not serie_tema.empty else "-"
    top_tema_tickets = int(serie_tema.iloc[0]["tickets"]) if not serie_tema.empty else 0
    top_solicitante = serie_solicitante.iloc[0]["De"] if not serie_solicitante.empty else "-"
    top_solicitante_tickets = int(serie_solicitante.iloc[0]["tickets"]) if not serie_solicitante.empty else 0
    top_departamento = serie_departamento.iloc[0]["Departamento"] if not serie_departamento.empty else "-"
    top_departamento_tickets = int(serie_departamento.iloc[0]["tickets"]) if not serie_departamento.empty else 0

    st.markdown(
        f"""
        <div class="section-block">
            <div class="section-title">Lectura ejecutiva</div>
            <div class="section-note">
                La vista actual contiene <b>{formato_entero(total_tickets)}</b> tickets. El mayor volumen se concentra en <b>{top_departamento}</b> con <b>{formato_entero(top_departamento_tickets)}</b> casos.
                El tema mas frecuente es <b>{top_tema}</b> con <b>{formato_entero(top_tema_tickets)}</b> tickets y el solicitante con mayor actividad es <b>{top_solicitante}</b> con <b>{formato_entero(top_solicitante_tickets)}</b> requerimientos.
                En esta misma vista existen <b>{formato_entero(abiertos)}</b> tickets abiertos, <b>{formato_entero(atrasados)}</b> atrasados y un cumplimiento SLA aproximado de <b>{tasa_sla:.1f}%</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    export_cols = st.columns(2)
    with export_cols[0]:
        st.download_button(
            "Descargar tickets filtrados",
            data=analisis_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="tickets_filtrados.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with export_cols[1]:
        st.download_button(
            "Descargar contexto interactivo",
            data=analisis_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="tickets_contexto.csv",
            mime="text/csv",
            use_container_width=True,
        )

    detalle = analisis_df.copy()
    detalle["Fecha de creación"] = detalle["Fecha de creación"].dt.strftime("%Y-%m-%d %H:%M")
    detalle["Última actualización"] = detalle["Última actualización"].dt.strftime("%Y-%m-%d %H:%M")
    detalle["Fecha de cierre"] = detalle["Fecha de cierre"].dt.strftime("%Y-%m-%d %H:%M")
    detalle["Fecha de expiración de SLA"] = detalle["Fecha de expiración de SLA"].dt.strftime("%Y-%m-%d %H:%M")
    detalle["tiempo_resolucion_horas"] = detalle["tiempo_resolucion_horas"].apply(formato_horas)

    columnas_detalle = [
        "Número de Ticket",
        "Fecha de creación",
        "Asunto",
        "Prioridad",
        "Departamento",
        "Temas de ayuda",
        "Fuente",
        "Estado actual",
        "Atrasado",
        "Respondió",
        "Agente asignado",
        "Fecha de expiración de SLA",
        "Fecha de cierre",
        "tiempo_resolucion_horas",
        "Cuenta de hilos",
        "Reabrir contador",
    ]
    st.dataframe(detalle[columnas_detalle], use_container_width=True, hide_index=True)
