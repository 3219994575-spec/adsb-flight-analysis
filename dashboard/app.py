
import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import json
import math
from pathlib import Path


# =====================================================
# 1. 页面配置
# =====================================================

st.set_page_config(
    page_title="ADS-B航空航迹分析系统",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. 项目路径与数据读取
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

trajectory_path = DATA_DIR / "trajectory.csv"
features_path = DATA_DIR / "flight_features_clean.csv"
anomaly_path = DATA_DIR / "flight_anomaly_results_clean.csv"
cluster_path = DATA_DIR / "flight_clustering_results_clean.csv"


@st.cache_data
def load_data():

    trajectory = pd.read_csv(trajectory_path)
    features = pd.read_csv(features_path)
    anomaly = pd.read_csv(anomaly_path)
    cluster = pd.read_csv(cluster_path)

    return trajectory, features, anomaly, cluster


try:
    trajectory_df, features_df, anomaly_df, clustering_df = load_data()

except Exception as e:
    st.error(f"数据读取失败：{e}")
    st.stop()


# =====================================================
# 3. 高德地图密钥
# =====================================================

try:
    AMAP_KEY = st.secrets["AMAP_KEY"]
    AMAP_SECURITY_CODE = st.secrets["AMAP_SECURITY_CODE"]

except Exception:
    st.error(
        "没有读取到高德地图密钥，请检查 "
        ".streamlit/secrets.toml 文件。"
    )
    st.stop()


# =====================================================
# 4. 页面样式
# =====================================================

st.markdown("""
<style>

.stApp {
    background-color: #071426;
    color: #FFFFFF;
}

section[data-testid="stSidebar"] {
    background-color: #10243B;
}

h1, h2, h3, h4 {
    color: #65C7FF !important;
    font-weight: 700 !important;
}

p, span, label, li {
    color: #F1F6FC;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: #FFFFFF !important;
}

div[data-testid="stMetric"] {
    background-color: #142B45;
    border: 1px solid #36B9FF;
    padding: 15px;
    border-radius: 10px;
}

div[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-weight: 700;
}

div[data-testid="stMetricLabel"] {
    color: #B9DFFF !important;
}

.stTextInput input {
    color: #FFFFFF !important;
    background-color: #142B45 !important;
}

.stSelectbox div[data-baseweb="select"] {
    background-color: #142B45;
}

hr {
    border-color: #31516F;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 5. 通用辅助函数
# =====================================================

def find_time_column(df):

    candidates = [
        "snapshot_datetime",
        "track_time",
        "timestamp",
        "time"
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def format_number(value, digits=2):

    if pd.isna(value):
        return "暂无数据"

    return f"{value:,.{digits}f}"


def haversine(lat1, lon1, lat2, lon2):

    R = 6371.0

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * R * np.arcsin(np.sqrt(a))


# =====================================================
# 6. WGS84 转 GCJ02
# =====================================================

def transform_lat(x, y):

    ret = (
        -100 + 2 * x + 3 * y
        + 0.2 * y * y
        + 0.1 * x * y
        + 0.2 * math.sqrt(abs(x))
    )

    ret += (
        20 * math.sin(6 * x * math.pi)
        + 20 * math.sin(2 * x * math.pi)
    ) * 2 / 3

    ret += (
        20 * math.sin(y * math.pi)
        + 40 * math.sin(y / 3 * math.pi)
    ) * 2 / 3

    ret += (
        160 * math.sin(y / 12 * math.pi)
        + 320 * math.sin(y * math.pi / 30)
    ) * 2 / 3

    return ret


def transform_lon(x, y):

    ret = (
        300 + x + 2 * y
        + 0.1 * x * x
        + 0.1 * x * y
        + 0.1 * math.sqrt(abs(x))
    )

    ret += (
        20 * math.sin(6 * x * math.pi)
        + 20 * math.sin(2 * x * math.pi)
    ) * 2 / 3

    ret += (
        20 * math.sin(x * math.pi)
        + 40 * math.sin(x / 3 * math.pi)
    ) * 2 / 3

    ret += (
        150 * math.sin(x / 12 * math.pi)
        + 300 * math.sin(x / 30 * math.pi)
    ) * 2 / 3

    return ret


def wgs84_to_gcj02(lon, lat):

    if not (
        72.004 <= lon <= 137.8347
        and 0.8293 <= lat <= 55.8271
    ):
        return lon, lat

    a = 6378245.0
    ee = 0.00669342162296594323

    dlat = transform_lat(lon - 105, lat - 35)
    dlon = transform_lon(lon - 105, lat - 35)

    radlat = lat / 180 * math.pi

    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic

    sqrtmagic = math.sqrt(magic)

    dlat = (
        dlat * 180
        / (
            (a * (1 - ee))
            / (magic * sqrtmagic)
            * math.pi
        )
    )

    dlon = (
        dlon * 180
        / (
            a / sqrtmagic
            * math.cos(radlat)
            * math.pi
        )
    )

    return lon + dlon, lat + dlat


# =====================================================
# 7. 高德地图绘制函数
# =====================================================

def show_amap_map(data, map_type="global", height=650):

    data = data.dropna(
        subset=["latitude", "longitude"]
    ).copy()

    if data.empty:
        st.warning("没有可用于绘制地图的坐标数据。")
        return

    if map_type == "global":
        data = data.sample(
            min(len(data), 5000),
            random_state=42
        )

    points = []

    for _, row in data.iterrows():

        lon, lat = wgs84_to_gcj02(
            float(row["longitude"]),
            float(row["latitude"])
        )

        points.append([lon, lat])

    if map_type == "track":

        center = points[len(points) // 2]
        zoom = 7

    else:

        center = [105, 35]
        zoom = 2

    points_json = json.dumps(points)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">

        <style>
            html, body, #amap-container {{
                width: 100%;
                height: 100%;
                margin: 0;
                padding: 0;
            }}
        </style>

        <script>
            window._AMapSecurityConfig = {{
                securityJsCode: "{AMAP_SECURITY_CODE}"
            }};
        </script>

        <script src="https://webapi.amap.com/loader.js"></script>

    </head>

    <body>

    <div id="amap-container"></div>

    <script>

    AMapLoader.load({{
        key: "{AMAP_KEY}",
        version: "2.0",
        plugins: [
            "AMap.Scale",
            "AMap.ToolBar"
        ]
    }}).then((AMap) => {{

        const map = new AMap.Map(
            "amap-container",
            {{
                zoom: {zoom},
                center: {json.dumps(center)},
                viewMode: "2D",
                mapStyle: "amap://styles/normal"
            }}
        );

        map.addControl(new AMap.Scale());
        map.addControl(new AMap.ToolBar());

        const points = {points_json};

        if ("{map_type}" === "global") {{

            const massData = points.map(
                p => ({{lnglat: p}})
            );

            const mass = new AMap.MassMarks(
                massData,
                {{
                    opacity: 0.85,
                    zIndex: 5,
                    cursor: "pointer",
                    style: {{
                        url: "https://webapi.amap.com/theme/v1.3/markers/n/mark_b.png",
                        anchor: new AMap.Pixel(5, 5),
                        size: new AMap.Size(8, 8)
                    }}
                }}
            );

            mass.setMap(map);

        }}

        if ("{map_type}" === "track" && points.length > 0) {{

            const line = new AMap.Polyline({{
                path: points,
                strokeColor: "#00BFFF",
                strokeWeight: 5,
                strokeOpacity: 0.9,
                lineJoin: "round"
            }});

            map.add(line);

            new AMap.Marker({{
                position: points[0],
                title: "航迹起点"
            }}).setMap(map);

            new AMap.Marker({{
                position: points[points.length - 1],
                title: "航迹终点"
            }}).setMap(map);

            map.setFitView([line]);

        }}

        // 台湾省文字标注
        const taiwanLabel = new AMap.Text({{
            text: "台湾省",
            position: [121.0, 23.7],
            style: {{
                "color": "#D71920",
                "font-size": "16px",
                "font-weight": "bold",
                "background-color": "rgba(255,255,255,0.8)",
                "border": "1px solid #D71920",
                "padding": "3px 6px"
            }}
        }});

        taiwanLabel.setMap(map);

    }}).catch((error) => {{

        document.getElementById(
            "amap-container"
        ).innerHTML =
            "<div style='padding:20px;color:red'>高德地图加载失败，请检查Key、安全密钥和网络连接。</div>";

        console.error(error);

    }});

    </script>

    </body>
    </html>
    """

    components.html(
        html_code,
        height=height,
        scrolling=False
    )


# =====================================================
# 8. 数据整理
# =====================================================

trajectory_df["icao24"] = (
    trajectory_df["icao24"].astype(str).str.lower()
)

features_df["icao24"] = (
    features_df["icao24"].astype(str).str.lower()
)

anomaly_df["icao24"] = (
    anomaly_df["icao24"].astype(str).str.lower()
)

clustering_df["icao24"] = (
    clustering_df["icao24"].astype(str).str.lower()
)

# 合并分析结果
result_df = anomaly_df.merge(
    clustering_df[["icao24", "cluster"]],
    on="icao24",
    how="left"
)

# 如果特征表中有分析字段，则补充到结果表
feature_columns = [
    "icao24",
    "flight_duration_s",
    "distance_km",
    "avg_speed_kmh",
    "max_speed_kmh",
    "max_altitude_m",
    "min_altitude_m",
    "altitude_change_m",
    "max_vertical_rate",
    "total_heading_change",
    "max_heading_change"
]

available_columns = [
    col for col in feature_columns
    if col in features_df.columns
]

result_df = result_df.merge(
    features_df[available_columns].drop_duplicates(
        subset=["icao24"]
    ),
    on="icao24",
    how="left",
    suffixes=("", "_feature")
)

# 清理重复字段
if "cluster" not in result_df.columns:
    result_df["cluster"] = np.nan


# =====================================================
# 9. 侧边栏导航
# =====================================================

st.sidebar.title("✈️ ADS-B航空分析系统")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "功能导航",
    [
        "项目总览",
        "全球航迹分布",
        "速度分布分析",
        "异常检测分析",
        "DBSCAN聚类分析",
        "航空器数据查询"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "基于 ADS-B 数据的航空航迹分析与异常检测系统"
)


# =====================================================
# 10. 项目总览
# =====================================================

if page == "项目总览":

    st.title("✈️ ADS-B航空航迹分析系统")

    st.markdown(
        "### 基于 ADS-B 数据的航空航迹分析与异常检测"
    )

    total_aircraft = len(result_df)

    if "anomaly_label" in result_df.columns:

        normal_count = (
            result_df["anomaly_label"] == 1
        ).sum()

        anomaly_count = (
            result_df["anomaly_label"] == -1
        ).sum()

    else:

        normal_count = 0
        anomaly_count = 0

    anomaly_rate = (
        anomaly_count / total_aircraft * 100
        if total_aircraft > 0 else 0
    )

    cluster_count = (
        result_df["cluster"].dropna().nunique()
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "航空器总数",
        f"{total_aircraft:,}"
    )

    col2.metric(
        "正常航空器",
        f"{normal_count:,}"
    )

    col3.metric(
        "异常航空器",
        f"{anomaly_count:,}"
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "异常比例",
        f"{anomaly_rate:.2f}%"
    )

    col5.metric(
        "DBSCAN聚类类别数",
        cluster_count
    )

    col6.metric(
        "航迹记录数",
        f"{len(trajectory_df):,}"
    )

    st.markdown("---")

    st.subheader("项目主要功能")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        **🌍 全球航空器空间分布**

        - 展示全球航空器位置
        - 观察航空器空间分布特征
        - 高德地图底图展示
        """)

        st.markdown("""
        **📊 飞行速度分布分析**

        - 飞行速度统计
        - 速度分布直方图
        - 飞行速度特征分析
        """)

        st.markdown("""
        **🔍 异常检测分析**

        - Isolation Forest
        - 异常航空器识别
        - 异常得分分析
        """)

    with col2:

        st.markdown("""
        **🧩 DBSCAN聚类分析**

        - 航空器行为聚类
        - 不同类别特征比较
        - 噪声点识别
        """)

        st.markdown("""
        **✈️ 航空器数据查询**

        - ICAO24查询
        - 单架航空器航迹展示
        - 高度和速度变化分析
        """)

    st.markdown("---")

    st.subheader("数据集基本信息")

    st.write(
        f"航迹数据记录数：{len(trajectory_df):,}"
    )

    st.write(
        f"特征数据记录数：{len(features_df):,}"
    )

    st.write(
        f"异常检测结果记录数：{len(anomaly_df):,}"
    )

    st.write(
        f"聚类结果记录数：{len(clustering_df):,}"
    )


# =====================================================
# 11. 全球航迹分布
# =====================================================

elif page == "全球航迹分布":

    st.title("🌍 全球航空器空间分布")

    map_df = trajectory_df.dropna(
        subset=["latitude", "longitude"]
    ).copy()

    st.metric(
        "地图航空器位置记录数",
        f"{len(map_df):,}"
    )

    st.info(
        "地图展示 ADS-B 采集窗口内航空器的位置，"
        "最多抽取5000个点用于显示。"
    )

    show_amap_map(
        map_df,
        map_type="global",
        height=700
    )


# =====================================================
# 12. 速度分布分析
# =====================================================

elif page == "速度分布分析":

    st.title("📊 航空器飞行速度分布分析")

    speed_col = None

    if "avg_speed_kmh_clean" in features_df.columns:
        speed_col = "avg_speed_kmh_clean"

    elif "avg_speed_kmh" in features_df.columns:
        speed_col = "avg_speed_kmh"

    if speed_col is None:

        st.error("没有找到平均速度字段。")

    else:

        speed_data = features_df[
            speed_col
        ].dropna()

        speed_data = speed_data[
            (speed_data >= 0)
            & (speed_data <= 1500)
        ]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "平均速度",
            format_number(speed_data.mean()) + " km/h"
        )

        col2.metric(
            "最高速度",
            format_number(speed_data.max()) + " km/h"
        )

        col3.metric(
            "速度中位数",
            format_number(speed_data.median()) + " km/h"
        )

        st.markdown("---")

        st.subheader("航空器平均速度分布")

        fig = px.histogram(
            speed_data,
            nbins=50,
            title="航空器平均飞行速度分布",
            labels={
                "value": "平均速度（km/h）",
                "count": "航空器数量"
            }
        )

        fig.update_traces(
            marker_color="#36B9FF"
        )

        fig.update_layout(
            template="plotly_dark",
            height=550,
            font=dict(color="white")
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("速度统计信息")

        st.dataframe(
            speed_data.describe().to_frame(
                name="速度统计"
            ),
            use_container_width=True
        )


# =====================================================
# 13. 异常检测分析
# =====================================================

elif page == "异常检测分析":

    st.title("🔍 Isolation Forest异常检测分析")

    if "anomaly_label" not in anomaly_df.columns:

        st.error("异常检测结果中没有 anomaly_label 字段。")
        st.stop()

    normal_count = (
        anomaly_df["anomaly_label"] == 1
    ).sum()

    abnormal_count = (
        anomaly_df["anomaly_label"] == -1
    ).sum()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "正常航空器",
        f"{normal_count:,}"
    )

    col2.metric(
        "异常航空器",
        f"{abnormal_count:,}"
    )

    col3.metric(
        "异常比例",
        f"{abnormal_count / len(anomaly_df) * 100:.2f}%"
    )

    st.markdown("---")

    # 异常得分分布
    if "anomaly_score" in anomaly_df.columns:

        st.subheader("异常得分分布")

        fig1 = px.histogram(
            anomaly_df,
            x="anomaly_score",
            color="anomaly_label",
            nbins=50,
            barmode="overlay",
            opacity=0.7,
            color_discrete_map={
                1: "#36B9FF",
                -1: "#FF5252"
            },
            title="正常与异常航空器得分分布"
        )

        fig1.update_layout(
            template="plotly_dark",
            height=450,
            font=dict(color="white")
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    # 异常航空器排名
    st.subheader("异常航空器排名")

    abnormal_df = anomaly_df[
        anomaly_df["anomaly_label"] == -1
    ].copy()

    if "anomaly_score" in abnormal_df.columns:

        abnormal_df = abnormal_df.sort_values(
            "anomaly_score",
            ascending=True
        )

    st.dataframe(
        abnormal_df.head(20),
        use_container_width=True
    )

    st.download_button(
        "下载异常航空器数据",
        data=abnormal_df.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="异常航空器数据.csv",
        mime="text/csv"
    )

    # 特征比较
    st.subheader("正常与异常航空器特征比较")

    compare_features = [
        "distance_km",
        "avg_speed_kmh",
        "max_altitude_m",
        "altitude_change_m",
        "total_heading_change"
    ]

    available_features = [
        col for col in compare_features
        if col in result_df.columns
    ]

    if available_features:

        compare_df = result_df[
            ["anomaly_label"] + available_features
        ].copy()

        compare_df = compare_df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        compare_df = compare_df.groupby(
            "anomaly_label"
        )[available_features].median()

        compare_df = compare_df.T

        compare_df.columns = [
            "异常航空器" if x == -1
            else "正常航空器"
            for x in compare_df.columns
        ]

        fig2 = go.Figure()

        for col in compare_df.columns:

            fig2.add_trace(
                go.Bar(
                    x=compare_df.index,
                    y=compare_df[col],
                    name=col
                )
            )

        fig2.update_layout(
            barmode="group",
            template="plotly_dark",
            height=500,
            title="正常与异常航空器特征中位数对比",
            font=dict(color="white")
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )


# =====================================================
# 14. DBSCAN聚类分析
# =====================================================

elif page == "DBSCAN聚类分析":

    st.title("🧩 DBSCAN航空器聚类分析")

    if "cluster" not in clustering_df.columns:

        st.error("聚类结果中没有 cluster 字段。")
        st.stop()

    cluster_counts = (
        clustering_df["cluster"]
        .value_counts()
        .sort_index()
    )

    st.subheader("各聚类类别航空器数量")

    fig1 = px.bar(
        x=cluster_counts.index.astype(str),
        y=cluster_counts.values,
        text=cluster_counts.values,
        labels={
            "x": "聚类类别",
            "y": "航空器数量"
        },
        title="DBSCAN聚类数量分布"
    )

    fig1.update_traces(
        marker_color="#36B9FF",
        textposition="outside"
    )

    fig1.update_layout(
        template="plotly_dark",
        height=450,
        font=dict(color="white")
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # 合并聚类结果和特征
    cluster_data = clustering_df.merge(
        features_df.drop_duplicates(
            subset=["icao24"]
        ),
        on="icao24",
        how="left",
        suffixes=("", "_feature")
    )

    st.markdown("---")

    # 平均距离
    distance_col = None

    if "distance_km_clean" in cluster_data.columns:
        distance_col = "distance_km_clean"

    elif "distance_km" in cluster_data.columns:
        distance_col = "distance_km"

    # 平均速度
    speed_col = None

    if "avg_speed_kmh_clean" in cluster_data.columns:
        speed_col = "avg_speed_kmh_clean"

    elif "avg_speed_kmh" in cluster_data.columns:
        speed_col = "avg_speed_kmh"

    col1, col2 = st.columns(2)

    if distance_col:

        with col1:

            distance_group = (
                cluster_data.groupby("cluster")[
                    distance_col
                ].mean().dropna()
            )

            fig2 = px.bar(
                x=distance_group.index.astype(str),
                y=distance_group.values,
                text=[
                    f"{x:.1f}"
                    for x in distance_group.values
                ],
                labels={
                    "x": "聚类类别",
                    "y": "平均飞行距离（km）"
                },
                title="各聚类平均飞行距离"
            )

            fig2.update_traces(
                textposition="outside",
                marker_color="#36B9FF"
            )

            fig2.update_layout(
                template="plotly_dark",
                height=500,
                font=dict(color="white")
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

    if speed_col:

        with col2:

            speed_group = (
                cluster_data.groupby("cluster")[
                    speed_col
                ].mean().dropna()
            )

            fig3 = px.bar(
                x=speed_group.index.astype(str),
                y=speed_group.values,
                text=[
                    f"{x:.1f}"
                    for x in speed_group.values
                ],
                labels={
                    "x": "聚类类别",
                    "y": "平均飞行速度（km/h）"
                },
                title="各聚类平均飞行速度"
            )

            fig3.update_traces(
                textposition="outside",
                marker_color="#00C9A7"
            )

            fig3.update_layout(
                template="plotly_dark",
                height=500,
                font=dict(color="white")
            )

            st.plotly_chart(
                fig3,
                use_container_width=True
            )

    st.markdown("---")

    st.subheader("聚类数据明细")

    st.dataframe(
        cluster_data,
        use_container_width=True
    )

    st.download_button(
        "下载DBSCAN聚类结果",
        data=cluster_data.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="DBSCAN聚类结果.csv",
        mime="text/csv"
    )


# =====================================================
# 15. 航空器数据查询
# =====================================================

elif page == "航空器数据查询":

    st.title("✈️ 航空器数据查询")

    st.markdown(
        "输入 ICAO24 地址，查询对应航空器的航迹和飞行数据。"
    )

    aircraft_id = st.text_input(
        "请输入 ICAO24",
        placeholder="例如：30708"
    )

    if aircraft_id:

        aircraft_id = aircraft_id.strip().lower()

        aircraft_track = trajectory_df[
            trajectory_df["icao24"] == aircraft_id
        ].copy()

        if aircraft_track.empty:

            st.warning(
                "没有找到该 ICAO24 对应的航空器记录。"
            )

        else:

            # 时间排序
            time_col = find_time_column(
                aircraft_track
            )

            if time_col:

                aircraft_track[time_col] = pd.to_datetime(
                    aircraft_track[time_col],
                    errors="coerce"
                )

                aircraft_track = aircraft_track.sort_values(
                    time_col
                )

            # 累计距离
            aircraft_track["segment_distance_km"] = 0.0

            if len(aircraft_track) > 1:

                distances = []

                for i in range(1, len(aircraft_track)):

                    previous = aircraft_track.iloc[i - 1]
                    current = aircraft_track.iloc[i]

                    distance = haversine(
                        previous["latitude"],
                        previous["longitude"],
                        current["latitude"],
                        current["longitude"]
                    )

                    distances.append(distance)

                aircraft_track.loc[
                    aircraft_track.index[1:],
                    "segment_distance_km"
                ] = distances

            aircraft_track["cumulative_distance_km"] = (
                aircraft_track[
                    "segment_distance_km"
                ].cumsum()
            )

            st.markdown("---")

            st.subheader("航空器基本信息")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "ICAO24",
                aircraft_id
            )

            col2.metric(
                "航迹记录数",
                len(aircraft_track)
            )

            col3.metric(
                "累计航迹距离",
                format_number(
                    aircraft_track[
                        "cumulative_distance_km"
                    ].iloc[-1]
                ) + " km"
            )

            # 高德地图
            st.markdown("---")

            st.subheader("🗺️ 航空器飞行航迹")

            show_amap_map(
                aircraft_track,
                map_type="track",
                height=650
            )

            # 高度变化
            if "baro_altitude" in aircraft_track.columns:

                st.markdown("---")

                st.subheader("📈 飞行高度变化")

                altitude_df = aircraft_track.dropna(
                    subset=["baro_altitude"]
                )

                if time_col:

                    fig_alt = px.line(
                        altitude_df,
                        x=time_col,
                        y="baro_altitude",
                        markers=True,
                        title="航空器气压高度变化"
                    )

                else:

                    fig_alt = px.line(
                        altitude_df,
                        y="baro_altitude",
                        markers=True,
                        title="航空器气压高度变化"
                    )

                fig_alt.update_layout(
                    template="plotly_dark",
                    height=400,
                    yaxis_title="高度（m）",
                    font=dict(color="white")
                )

                st.plotly_chart(
                    fig_alt,
                    use_container_width=True
                )

            # 速度变化
            if "velocity" in aircraft_track.columns:

                st.markdown("---")

                st.subheader("📈 飞行速度变化")

                speed_df = aircraft_track.copy()

                speed_df["speed_kmh"] = (
                    speed_df["velocity"] * 3.6
                )

                speed_df = speed_df.dropna(
                    subset=["speed_kmh"]
                )

                if time_col:

                    fig_speed = px.line(
                        speed_df,
                        x=time_col,
                        y="speed_kmh",
                        markers=True,
                        title="航空器地速变化"
                    )

                else:

                    fig_speed = px.line(
                        speed_df,
                        y="speed_kmh",
                        markers=True,
                        title="航空器地速变化"
                    )

                fig_speed.update_layout(
                    template="plotly_dark",
                    height=400,
                    yaxis_title="速度（km/h）",
                    font=dict(color="white")
                )

                st.plotly_chart(
                    fig_speed,
                    use_container_width=True
                )

            # 航迹数据
            st.markdown("---")

            st.subheader("航迹数据明细")

            st.dataframe(
                aircraft_track,
                use_container_width=True
            )

            st.download_button(
                "下载该航空器航迹数据",
                data=aircraft_track.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name=f"{aircraft_id}_航迹数据.csv",
                mime="text/csv"
            )

