# ADS-B 航空航迹分析系统
# ADS-B Flight Analysis System

## 一、项目简介 | Project Overview

本项目基于 OpenSky Network 提供的 ADS-B 数据，使用 Python 对航空器飞行数据进行采集、清洗和分析，并通过机器学习方法进行异常检测和聚类分析。

项目使用 Streamlit 搭建交互式网页，展示航空器分布、速度特征、异常检测和聚类结果。

This project uses ADS-B data from OpenSky Network to collect, process, and analyze aircraft flight data. Isolation Forest and DBSCAN are used for anomaly detection and clustering.

A Streamlit dashboard is developed to display aircraft distribution, speed characteristics, and analysis results.

## 二、使用技术 | Technologies

- Python
- Pandas、NumPy
- Matplotlib、Plotly
- Scikit-learn
- Streamlit
- OpenSky Network API

## 三、主要功能 | Features

- ADS-B 数据获取与清洗
- 航迹距离、速度及高度特征计算
- Isolation Forest 异常检测
- DBSCAN 聚类分析
- 航空器分布及飞行特征可视化
- Streamlit 交互式数据查询与展示

**Main Features:**
- ADS-B data collection and preprocessing
- Flight distance, speed, and altitude analysis
- Anomaly detection using Isolation Forest
- Clustering using DBSCAN
- Aircraft data visualization
- Interactive Streamlit dashboard

## 四、项目结构 | Project Structure

```text
adsb-flight-analysis/
├── dashboard/      # Streamlit 网页系统
├── notebooks/      # 数据处理与分析
├── results/        # 可视化结果
├── data/           # 数据文件
├── .gitignore
└── README.md
```

## 五、运行方法 | How to Run

安装依赖：

```bash
pip install -r requirements.txt
```

启动项目：

```bash
python -m streamlit run dashboard/app.py
```

浏览器访问：

```text
http://localhost:8501
```

运行前需要准备处理后的数据文件，并配置高德地图 API。

## 六、数据来源 | Data Source

[OpenSky Network](https://opensky-network.org/)

## 七、作者 | Author

**Benson**

GitHub: [3219994575-spec](https://github.com/3219994575-spec)
