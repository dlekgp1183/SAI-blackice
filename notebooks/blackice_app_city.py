# -*- coding: utf-8 -*-
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import os
import osmnx as ox
from shapely.geometry import LineString, MultiLineString
import altair as alt
from datetime import datetime
import time

# =========================
# 페이지 설정
# =========================
st.set_page_config(page_title="Black Ice Safety Dashboard", page_icon="❄️", layout="wide")

# =========================
# CSS 커스텀
# =========================
st.markdown("""
<style>
/* =========================
   🔹 폰트 정의
========================= */
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

/* =========================
   🔹 전체 폰트
========================= */
body, p, h1, h2, h3, h4, h5, h6,
.stMetric-value, .stMetric-delta, .stDataFrame, .stMarkdown p,
.stText, .stButton > button, [class*="st-emotion-"] {
    font-family: 'LeeSunSinDotum', sans-serif !important;
}

/* =========================
   🔹 제목 스타일
========================= */
h1.title-font { 
    font-family: 'Cafe24Surround', sans-serif !important; 
}

/* =========================
   🔹 서브헤더 박스
========================= */
.subheader-box {
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

/* =========================
   🔹 사이드바 메뉴 폰트 스타일
========================= */

/* "MENU" 제목 폰트 → Cafe24Surround */
div[data-testid="stSidebarNav"] h2, 
section[data-testid="stSidebar"] h2,
div[data-testid="stSidebar"] h1 {
    font-family: 'Cafe24Surround', sans-serif !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #004D40 !important;
}

/* 메뉴 항목 (고속도로 리스트) → LeeSunSinDotum */
ul[class*="css-"] li,
div[data-testid="stSidebar"] div[role="listbox"] span {
    font-family: 'LeeSunSinDotum', sans-serif !important;
    font-size: 17px !important;
    color: #00332E !important;
}

/* 선택된 항목 강조 */
ul[class*="css-"] li[data-selected="true"] {
    background-color: rgba(0, 77, 64, 0.1) !important;
    border-radius: 8px !important;
}

/* =========================
   🔹 사이드바 기본 열림 유지
========================= */
[data-testid="stSidebarNavCollapseButton"] {
    display: none !important; /* 접기 버튼 숨김 */
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title-font">❄️ 블랙아이스 위험도 모니터링</h1>', unsafe_allow_html=True)

# =========================
# 메트릭
# =========================
cols = st.columns(6, gap="small")
cols[0].metric("Max temperature", "35.0°C", delta="-0.6°C")
cols[1].metric("Min temperature", "-3.8°C", delta="2.2°C")
cols[2].metric("Max precipitation", "55.9mm", delta="9.2mm")
cols[3].metric("Min precipitation", "0.0mm",delta="0.0mm")
cols[4].metric("Max wind", "8.0 m/s", delta="-0.8 m/s")
cols[5].metric("Min wind", "0.5 m/s", delta="-0.1 m/s")

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
# 샘플 데이터 로드
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "highway_sample_data")
SAMPLE_FILENAME = f"{highway_choice}_{selected_city}_sample.csv"
SAMPLE_PATH = os.path.join(SAMPLE_DIR, SAMPLE_FILENAME)

try:
    df = pd.read_csv(SAMPLE_PATH)
except FileNotFoundError:
    st.error(f"❌ '{SAMPLE_PATH}' 파일을 찾을 수 없습니다.")
    st.stop()

# =========================
# 좌표 캐시
# =========================
@st.cache_data
def load_or_cache_coords(highway_name, city_name):
    os.makedirs("coords_cache", exist_ok=True)
    filename = f"{highway_name}_{city_name}.csv"
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
# 자동 데이터 추가 (샘플 기반)
# =========================
current_hour = datetime.now().strftime("%H")
status_placeholder = st.empty()

def add_new_data(df_points, road_df, n=1):
    """샘플 데이터에서 데이터를 가져와 자동으로 추가"""
    if len(df_points) >= 50:
        return df_points
    new_rows = []
    for _ in range(n):
        sample = df.sample(1).iloc[0]
        coord = road_df.sample(1).iloc[0]
        new_rows.append({
            "lon": coord["lon"], "lat": coord["lat"],
            "road_tmpr": sample.get("road_tmpr", np.nan),
            "atmp_tmpr": sample.get("atmp_tmpr", np.nan),
            "rltv_hmdt": sample.get("rltv_hmdt", np.nan),
            "hour": int(sample.get("hour", 0)),
            "time_slot": sample.get("time_slot", "morning"),
            "risk": sample.get("risk", 0)
        })
    return pd.concat([df_points, pd.DataFrame(new_rows)], ignore_index=True)

# 1개씩 추가 & 상태 출력
if len(df_points) < 50:
    status_placeholder.markdown(
        f"<p style='color:#0277BD; font-size:18px; font-weight:600; "
        f"font-family:LeeSunSinDotum;'>🕓 {current_hour}시 데이터를 받고 있습니다...</p>",
        unsafe_allow_html=True
    )
    st.session_state['highway_data'][highway_choice][key_combo] = add_new_data(df_points, road_df, n=1)
    time.sleep(1)  # ⬅️ 3초 딜레이
    df_points = st.session_state['highway_data'][highway_choice][key_combo]
else:
    status_placeholder.markdown(
        f"<p style='color:#00695C; font-size:18px; font-weight:600; "
        f"font-family:LeeSunSinDotum;'>🕓 {current_hour}시 데이터 로드가 완료되었습니다.</p>",
        unsafe_allow_html=True
    )

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
    st.markdown(f'<div class="subheader-box">샘플 데이터 수치표 - {selected_city}</div>', unsafe_allow_html=True)
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
