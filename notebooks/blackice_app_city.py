# -*- coding: utf-8 -*-
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from joblib import load
import os
import osmnx as ox
from shapely.geometry import LineString, MultiLineString
import altair as alt
import requests
from datetime import datetime

# =========================
# 페이지 설정
# =========================
st.set_page_config(page_title="Black Ice Safety Dashboard", page_icon="❄️", layout="wide")

# =========================
# 모델 로드
# =========================
MODEL_URL = "https://github.com/dlekgp1183/SAI-blackice/releases/download/v1.0/blackice_model.joblib"
MODEL_FILENAME = "blackice_model.joblib"
CACHE_DIR = "model_cache"
MODEL_PATH = os.path.join(CACHE_DIR, MODEL_FILENAME)

@st.cache_data(show_spinner="⏳ 모델 로드 중...")
def load_model():
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH):
        return load(MODEL_PATH)
    res = requests.get(MODEL_URL, stream=True, timeout=300)
    res.raise_for_status()
    with open(MODEL_PATH, 'wb') as f:
        for chunk in res.iter_content(8192):
            f.write(chunk)
    return load(MODEL_PATH)

model = load_model()

# =========================
# CSS 커스텀
# =========================
st.markdown("""
<style>
.subheader-box{
    background: linear-gradient(90deg, #CBF7F7, #A9CCCC);
    color: #004D40;
    padding: 6px 20px;
    border-radius: 12px;
    font-weight: 900;
    margin-bottom: 16px;
    font-size: 23px;
    text-align: left;
    font-family: 'LeeSunSinDotum', sans-serif;
}
@font-face {
    font-family: 'Cafe24Surround';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_2105_2@1.0/Cafe24Ssurround.woff') format('woff');
    font-weight: normal;
    font-display: swap;
}
@font-face {
    font-family: 'LeeSunSinDotum';
    src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_two@1.0/YiSunShinDotumM.woff') format('woff');
    font-weight: normal;
    font-display: swap;
}
h1.title-font { 
    font-family: 'Cafe24Surround', sans-serif !important; 
}
body, p, h2, h3, h4, h5, h6, 
.stMetric-value, .stMetric-delta, .stDataFrame, .stMarkdown p, .stText, .stButton > button, 
[class*="st-emotion-"] {
    font-family: 'LeeSunSinDotum', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title-font">❄️ 블랙아이스 위험도 모니터링</h1>', unsafe_allow_html=True)

# =========================
# test_data.csv 로드
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "test_data.csv")

try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    st.error(f"❌ '{CSV_PATH}' 파일을 찾을 수 없습니다.")
    st.stop()

# =========================
# 예측 관련 함수
# =========================
def predict_road_state(model, atmp_tmpr, road_tmpr, rltv_hmdt, hour):
    slot = "midnight" if hour<6 else "morning" if hour<12 else "afternoon" if hour<18 else "evening"
    input_data = {
        "atmp_tmpr": [atmp_tmpr],
        "road_tmpr": [road_tmpr],
        "rltv_hmdt": [rltv_hmdt],
        "time_slot_midnight": [1 if slot=="midnight" else 0],
        "time_slot_morning": [1 if slot=="morning" else 0],
        "time_slot_afternoon": [1 if slot=="afternoon" else 0],
        "time_slot_evening": [1 if slot=="evening" else 0],
    }
    input_df = pd.DataFrame(input_data)
    input_df = input_df[model.feature_names_in_]
    pred_class = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    return pred_class, {cls: p for cls,p in zip(model.classes_, proba)}, slot

def calculate_risk_limited(proba_dict, atmp_tmpr, road_tmpr):
    class_weights = {"DRY":0,"IC1":0.9,"IC2":1,"SN1":0.4,"SN2":0.4,"WT1":0.5,"WT2":0.5,"WT3":0.5}
    risk = 0
    for cls, prob in proba_dict.items():
        weight = class_weights.get(cls, 0)
        if cls in ["WT1","WT2","WT3"]:
            if atmp_tmpr < 0 or road_tmpr < 0:
                weight = min(weight + max(0,-min(atmp_tmpr,road_tmpr))/10, 1)
        risk += prob*weight
    return round(risk*100,1)

# =========================
# 좌표 캐시
# =========================
@st.cache_data
def load_or_cache_coords(highway_name, city_name):
    os.makedirs("coords_cache", exist_ok=True)
    filename = f"coords_cache/{highway_name}_{city_name}.csv"
    if os.path.exists(filename):
        return pd.read_csv(filename)
    try:
        G = ox.graph_from_place(f"{city_name}, South Korea", network_type='drive')
        nodes, edges = ox.graph_to_gdfs(G)
        edges = edges[edges['name'].str.contains(highway_name, na=False)]
        coords = []
        for _, row in edges.iterrows():
            geom = row['geometry']
            lines = [geom] if isinstance(geom, LineString) else list(geom.geoms) if isinstance(geom, MultiLineString) else []
            for line in lines:
                xs = np.linspace(line.coords[0][0], line.coords[-1][0], 10)
                ys = np.linspace(line.coords[0][1], line.coords[-1][1], 10)
                coords.extend(list(zip(xs, ys)))
        df_coords = pd.DataFrame(coords, columns=['lon','lat'])
        df_coords.to_csv(filename, index=False)
        return df_coords
    except Exception as e:
        print(f"⚠️ {city_name} OSMnx 로드 실패:", e)
        return pd.DataFrame(columns=['lon','lat'])

# =========================
# 고속도로/도시
# =========================
highways = ["경부고속도로", "호남고속도로", "경인고속도로"]
cities_dict = {
    "경부고속도로": ["대전", "서울", "부산"],
    "호남고속도로": ["광주", "대전"],
    "경인고속도로": ["서울", "인천"]
}

# =========================
# 사이드바
# =========================
with st.sidebar:
    highway_choice = option_menu(
        "MENU",
        highways,
        icons=['map', 'map', 'map'],
        menu_icon="arrow",
        default_index=0
    )

selected_city = st.selectbox(f"{highway_choice} 주요 도시 선택", cities_dict[highway_choice])

# =========================
# 세션 초기화
# =========================
if 'highway_data' not in st.session_state:
    st.session_state['highway_data'] = {}
if 'all_coords' not in st.session_state:
    st.session_state['all_coords'] = {}

key_combo = f"{highway_choice}_{selected_city}"
st.session_state['all_coords'].setdefault(highway_choice, {})
st.session_state['all_coords'][highway_choice].setdefault(selected_city, load_or_cache_coords(highway_choice, selected_city))

st.session_state['highway_data'].setdefault(highway_choice, {})
st.session_state['highway_data'][highway_choice].setdefault(key_combo, pd.DataFrame(
    columns=["lon","lat","road_tmpr","atmp_tmpr","rltv_hmdt","hour","time_slot","risk"]
))

road_df = st.session_state['all_coords'][highway_choice][selected_city]
df_points = st.session_state['highway_data'][highway_choice][key_combo]

# =========================
# 자동 데이터 추가
# =========================
def add_new_data(df_points, road_df, n=1):
    if len(df_points) >= 50:
        return df_points
    new_rows = []
    for _ in range(n):
        sample = df.sample(1).iloc[0]
        coord = road_df.sample(1).iloc[0]
        atmp_tmpr = sample.get("atmp_tmpr", np.random.uniform(-5,10))
        road_tmpr = sample.get("road_tmpr", np.random.uniform(-5,15))
        rltv_hmdt = sample.get("rltv_hmdt", np.random.uniform(30,100))
        hour = int(sample.get("hour", np.random.randint(0,24)))
        _, proba, slot = predict_road_state(model, atmp_tmpr, road_tmpr, rltv_hmdt, hour)
        risk = calculate_risk_limited(proba, atmp_tmpr, road_tmpr)
        new_rows.append({
            "lon": coord["lon"], "lat": coord["lat"],
            "road_tmpr": road_tmpr, "atmp_tmpr": atmp_tmpr,
            "rltv_hmdt": rltv_hmdt, "hour": hour,
            "time_slot": slot, "risk": risk
        })
    return pd.concat([df_points, pd.DataFrame(new_rows)], ignore_index=True)

if len(df_points) < 50:
    st.session_state['highway_data'][highway_choice][key_combo] = add_new_data(df_points, road_df, n=5)
    df_points = st.session_state['highway_data'][highway_choice][key_combo]

# =========================
# Heatmap & 수치표 & 파이차트
# =========================
left_col, right_col = st.columns([1.5, 2])

with left_col.container():
    st.markdown(f'<div class="subheader-box">위험도 Heatmap - {selected_city}</div>', unsafe_allow_html=True)
    if df_points.empty:
        m = folium.Map(location=[37.5665, 126.9780], zoom_start=12)
    else:
        lat_mean = df_points['lat'].mean()
        lon_mean = df_points['lon'].mean()
        m = folium.Map(location=[lat_mean, lon_mean], zoom_start=13)
        HeatMap(df_points[['lat','lon','risk']].values, radius=18, blur=10, min_opacity=0.5).add_to(m)
    st_folium(m, width=700, height=500)

with right_col.container():
    st.markdown(f'<div class="subheader-box">모델 예측 데이터 수치표 - {selected_city}</div>', unsafe_allow_html=True)
    def highlight_risk(row):
        return ['background-color: #FFCCCC' if row['risk'] >= 70 else '' for _ in row]
    if not df_points.empty:
        styled_df = df_points[["lat","lon","road_tmpr","atmp_tmpr","rltv_hmdt","hour","time_slot","risk"]].sort_values(by="risk", ascending=False).reset_index(drop=True).style.apply(highlight_risk, axis=1)
        st.dataframe(styled_df, height=400)
    else:
        st.info("데이터를 추가해 주세요.")

with right_col.container():
    st.markdown(f'<div class="subheader-box">안전/주의/위험 구간 비율 - {selected_city}</div>', unsafe_allow_html=True)
    if not df_points.empty:
        bins = pd.cut(df_points['risk'], bins=[0,30,60,100], labels=['안전','주의','위험'])
        count = bins.value_counts().reindex(['안전','주의','위험']).reset_index()
        count.columns = ['category','count']
        pie_chart = alt.Chart(count).mark_arc(innerRadius=30).encode(
            theta=alt.Theta(field="count", type="quantitative"),
            color=alt.Color(field="category", type="nominal",
                            scale=alt.Scale(domain=['안전','주의','위험'],
                                            range=['#3CB371','#FFD700','#FF6347'])),
            tooltip=['category','count']
        ).properties(width=300, height=250)
        st.altair_chart(pie_chart, use_container_width=True)
    else:
        st.info("데이터를 추가해 주세요.")
 