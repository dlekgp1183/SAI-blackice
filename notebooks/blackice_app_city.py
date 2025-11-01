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

st.set_page_config(page_title="Black Ice Safety Dashboard", page_icon="❄️", layout="wide")

# =========================
# 모델 다운로드 URL
# =========================
MODEL_URL = "https://github.com/dlekgp1183/SAI-blackice/releases/download/v1.0/blackice_model.joblib"
MODEL_FILENAME = "blackice_model.joblib"
CACHE_DIR = "model_cache"
MODEL_PATH = os.path.join(CACHE_DIR, MODEL_FILENAME)

# =========================
# 모델 다운로드 및 로드
# =========================
@st.cache_data(show_spinner="⏳ 모델 파일 다운로드 및 로드 중...")
def load_model_from_github():
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH):
        return load(MODEL_PATH)
    try:
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return load(MODEL_PATH)
    except requests.exceptions.RequestException as e:
        st.error(f"❌ GitHub에서 모델 다운로드 실패: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ 모델 로드 중 오류 발생: {e}")
        st.stop()

# =========================
# CSS: 폰트 + subheader
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

# =========================
# 제목
# =========================
st.markdown('<h1 class="title-font">❄️ 블랙아이스 위험도 모니터링</h1>', unsafe_allow_html=True)

# =========================
# 모델 로드
# =========================
try:
    model = load_model_from_github()
except Exception:
    st.error("모델 로드에 실패하여 앱 실행을 중단합니다.")
    st.stop()

# =========================
# 예측 함수
# =========================
def predict_road_state(model, atmp_tmpr, road_tmpr, rltv_hmdt, hour):
    def time_slot(hour):
        if 0 <= hour < 6: return "midnight"
        elif 6 <= hour < 12: return "morning"
        elif 12 <= hour < 18: return "afternoon"
        else: return "evening"
    slot = time_slot(hour)
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
    proba_dict = {cls: prob for cls, prob in zip(model.classes_, proba)}
    return pred_class, proba_dict, slot

def calculate_risk_limited(proba_dict, atmp_tmpr, road_tmpr):
    class_weights = {"DRY":0,"IC1":0.9,"IC2":1,"SN1":0.4,"SN2":0.4,"WT1":0.5,"WT2":0.5,"WT3":0.5}
    risk = 0
    for cls, prob in proba_dict.items():
        weight = class_weights.get(cls, 0)
        if cls in ["WT1","WT2","WT3"]:
            if atmp_tmpr < 0 or road_tmpr < 0:
                temp_factor = max(0,-min(atmp_tmpr,road_tmpr))/10
                weight = min(weight+temp_factor,1)
        risk += prob*weight
    return round(risk*100,1)

# =========================
# 데이터 로드
# =========================
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "test_data.csv"))

# =========================
# 좌표 캐시 함수 (레포지토리 파일 우선)
# =========================
@st.cache_data
def load_or_cache_coords(highway_name, city_name):
    import os
    BASE_DIR = os.path.dirname(__file__)  # 현재 파일 기준 경로
    filename = os.path.join(BASE_DIR, f"coords_{highway_name}_{city_name}.csv")
    return pd.read_csv(filename)




# =========================
# 고속도로, 도시 목록
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
        default_index=0,
        styles={
            "container": {"padding": "4!important", "background-color": "#f7f9f9"},
            "icon": {"color": "#03645a", "font-size": "18px"},
            "nav-link": {"font-size": "14px", "text-align": "left", "margin": "2px", "--hover-color": "#e6f7f7"},
            "nav-link-selected": {"background-color": "#8ac0ba", "color": "white"},
        }
    )

# =========================
# 메트릭
# =========================
cols = st.columns(6, gap="small")
cols[0].metric("Max temperature", "35.0°C", delta="-0.6°C")
cols[1].metric("Min temperature", "-3.8°C", delta="2.2°C")
cols[2].metric("Max precipitation", "55.9°C", delta="9.2°C")
cols[3].metric("Min precipitation", "0.0°C", delta="0.0°C")
cols[4].metric("Max wind", "8.0 m/s", delta="-0.8 m/s")
cols[5].metric("Min wind", "0.5 m/s", delta="-0.1 m/s")

# =========================
# 세션 상태 초기화
# =========================
if 'highway_data' not in st.session_state:
    st.session_state['highway_data'] = {}
if 'all_coords' not in st.session_state:
    st.session_state['all_coords'] = {}

# =========================
# 좌표 로드
# =========================
for highway in highways:
    if highway not in st.session_state['all_coords']:
        st.session_state['all_coords'][highway] = {}
    for city in cities_dict[highway]:
        if city not in st.session_state['all_coords'][highway]:
            st.session_state['all_coords'][highway][city] = load_or_cache_coords(highway, city)

# =========================
# 도시 선택
# =========================
selected_city = st.selectbox(f"{highway_choice} 주요 도시 선택", cities_dict[highway_choice])

# =========================
# 세션 데이터 초기화
# =========================
key_combo = f"{highway_choice}_{selected_city}"
if highway_choice not in st.session_state['highway_data']:
    st.session_state['highway_data'][highway_choice] = {}
if key_combo not in st.session_state['highway_data'][highway_choice]:
    st.session_state['highway_data'][highway_choice][key_combo] = pd.DataFrame(
        columns=["lon","lat","road_tmpr","atmp_tmpr","rltv_hmdt","hour","time_slot","risk"]
    )

road_df = st.session_state['all_coords'][highway_choice][selected_city]
df_points = st.session_state['highway_data'][highway_choice][key_combo]

# =========================
# 페이지 레이아웃
# =========================
left_col, right_col = st.columns([1.5, 2])

# ---------- Heatmap ----------
with left_col.container():
    st.markdown(f'<div class="subheader-box">위험도 Heatmap - {selected_city}</div>', unsafe_allow_html=True)
    if st.button("새로고침", key=f"refresh_{key_combo}"):
        new_rows = []
        for _ in range(5):
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
        
        # ------------------------
        # 빈 DataFrame 체크 후 concat
        # ------------------------
        existing_df = st.session_state['highway_data'][highway_choice][key_combo]
        new_df = pd.DataFrame(new_rows)
        if not new_df.empty:
            st.session_state['highway_data'][highway_choice][key_combo] = pd.concat(
                [existing_df, new_df],
                ignore_index=True
            )
        df_points = st.session_state['highway_data'][highway_choice][key_combo]

    if df_points.empty:
        m = folium.Map(location=[37.5665, 126.9780], zoom_start=12)
    else:
        lat_mean = df_points['lat'].mean()
        lon_mean = df_points['lon'].mean()
        lat_min, lat_max = df_points['lat'].min(), df_points['lat'].max()
        lon_min, lon_max = df_points['lon'].min(), df_points['lon'].max()
        zoom_level = int(14 - max(lat_max-lat_min, lon_max-lon_min)*30)
        zoom_level = max(12, min(zoom_level, 18))
        m = folium.Map(location=[lat_mean, lon_mean], zoom_start=zoom_level)
        HeatMap(df_points[['lat','lon','risk']].values, radius=18, blur=10, min_opacity=0.5).add_to(m)

    st_folium(m, width=700, height=500)

# ---------- 수치표 ----------
with right_col.container():
    st.markdown(f'<div class="subheader-box">모델 예측 데이터 수치표 - {selected_city}</div>', unsafe_allow_html=True)

    def highlight_risk(row):
        return ['background-color: #FFCCCC' if row['risk'] >= 70 else '' for _ in row]

    if not df_points.empty:
        styled_df = (
            df_points[["lat","lon","road_tmpr","atmp_tmpr","rltv_hmdt","hour","time_slot","risk"]]
            .sort_values(by="risk", ascending=False)
            .reset_index(drop=True)
            .style.apply(highlight_risk, axis=1)
        )
        st.dataframe(styled_df, height=400)
    else:
        st.info("데이터를 추가해 주세요.")

# ---------- 파이차트 ----------
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
