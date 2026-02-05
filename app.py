import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from datetime import datetime, timedelta, time

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Lil-Mamba Flood Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Thiết lập nền trắng
sns.set_theme(style="whitegrid")

# --- 2. LOAD DỮ LIỆU ---
@st.cache_data
def load_data():
    file_path = 'Mersey_Data_2025_Renamed.csv'
    
    if not os.path.exists(file_path):
        dates = pd.date_range(start='2025-01-01', periods=744, freq='H') 
        df = pd.DataFrame({
            'Time': dates,
            'Sea Surface Height': 3 * np.sin(np.linspace(0, 30*np.pi, 744)) + 30,
            'Significant Wave Height': np.random.gamma(2, 1, 744),
            '10m u-component of wind': np.random.normal(0, 5, 744),
            '10m v-component of wind': np.random.normal(0, 5, 744),
            'Mean Sea Level Pressure': np.random.normal(101325, 1000, 744),
            'Mean Wave Direction': np.random.uniform(0, 360, 744),
            'Potential Temperature': np.random.uniform(5, 15, 744),
            'Bottom Temperature': np.random.uniform(4, 14, 744)
        })
    else:
        df = pd.read_csv(file_path)
        df['Time'] = pd.to_datetime(df['Time'])

    # 1. Calc Wind Speed
    df['Wind Speed'] = np.sqrt(df['10m u-component of wind']**2 + df['10m v-component of wind']**2)
    
    # 2. Simulate Lil-Mamba Prediction (Shifted for visual effect later)
    np.random.seed(42)
    noise = np.random.normal(0, 0.0682, size=len(df))
    df['Lil-Mamba Prediction'] = df['Sea Surface Height'] + noise
    
    # 3. Categorize Wave Direction
    def categorize_direction(deg):
        if deg is np.nan: return 'Unknown'
        deg = deg % 360
        dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        return dirs[int((deg + 22.5) // 45) % 8]
    df['WaveDirCat'] = df['Mean Wave Direction'].apply(categorize_direction)
    
    # 4. Categorize Sea State
    bins = [-0.1, 0.1, 0.5, 1.25, 2.5, 4, 6, 9, 14, 20]
    labels = ['Calm', 'Smooth', 'Slight', 'Moderate', 'Rough', 'Very Rough', 'High', 'Very High', 'Phenomenal']
    df['SeaStateCat'] = pd.cut(df['Significant Wave Height'], bins=bins, labels=labels)
    
    # 5. Month Extraction
    df['Month'] = df['Time'].dt.month_name().str[:3]

    return df

df = load_data()

# --- 3. SIDEBAR CONTROLS (TÁCH BIỆT 2 PHẦN) ---
st.sidebar.title("🎛️ Control Panel")

# --- PHẦN 1: ĐIỀU KHIỂN RIÊNG CHO SEA LEVEL (FORECAST) ---
st.sidebar.markdown("---")
st.sidebar.header("1. Sea Level Forecast (24h Window)")
st.sidebar.caption("Chọn thời điểm bắt đầu quan trắc:")

# Mặc định lấy ngày đầu tiên trong dữ liệu
min_date = df['Time'].min().date()
max_date = df['Time'].max().date()

sea_start_date = st.sidebar.date_input("Start Date", min_date, key='sea_date')
sea_start_time = st.sidebar.slider("Start Time (Hour)", 0, 23, 0, key='sea_time')

# Tính toán mốc thời gian cho Sea Level
# Mốc 1: Bắt đầu (Ví dụ: 0h ngày 2/1)
dt_start_obs = datetime.combine(sea_start_date, time(sea_start_time, 0))
# Mốc 2: Kết thúc quan trắc = Bắt đầu + 24h (Ví dụ: 0h ngày 3/1)
dt_end_obs = dt_start_obs + timedelta(hours=24)
# Mốc 3: Kết thúc dự báo = Kết thúc quan trắc + 1h (Ví dụ: 1h ngày 3/1)
dt_end_pred = dt_end_obs + timedelta(hours=1)

# Lọc dữ liệu riêng cho Sea Level
# Data Quan trắc (24h đầu)
mask_obs = (df['Time'] >= dt_start_obs) & (df['Time'] <= dt_end_obs)
df_obs = df.loc[mask_obs]

# Data Dự báo (1h sau đó)
mask_pred = (df['Time'] > dt_end_obs) & (df['Time'] <= dt_end_pred)
df_pred = df.loc[mask_pred]

# --- PHẦN 2: ĐIỀU KHIỂN CHO CÁC BIỂU ĐỒ CÒN LẠI ---
st.sidebar.markdown("---")
st.sidebar.header("2. General Analysis (Other Charts)")
st.sidebar.caption("Chọn khoảng thời gian phân tích tổng quan:")

gen_start_date = st.sidebar.date_input("From Date", min_date, key='gen_start')
gen_end_date = st.sidebar.date_input("To Date", min_date + pd.Timedelta(days=7), key='gen_end')

if gen_start_date > gen_end_date:
    st.sidebar.error("Error: Start Date must be before End Date.")

mask_general = (df['Time'].dt.date >= gen_start_date) & (df['Time'].dt.date <= gen_end_date)
df_general = df.loc[mask_general]

# --- FLOOD WARNING CONFIGURATION ---
st.sidebar.markdown("---")
st.sidebar.header("⚠️ Flood Warning System")

default_threshold = 3.7
st.sidebar.info(f"**Custom Threshold:** Fixed at {default_threshold}m")

flood_threshold = st.sidebar.slider(
    "Set Flood Threshold (m):", 
    min_value=2.0, 
    max_value=5.0, 
    value=default_threshold, 
    step=0.1
)

# Logic tìm điểm ngập (Dựa trên dữ liệu dự báo hoặc tổng quan tùy bạn chọn, ở đây mình dùng data tổng quan để cảnh báo chung)
flood_events = pd.DataFrame()
if not df_general.empty:
    flood_events = df_general[df_general['Lil-Mamba Prediction'] > flood_threshold].copy()
    is_flooding = not flood_events.empty
else:
    is_flooding = False

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Model Info:**
    * **Name:** Lil-Mamba
    * **Task:** Flood Nowcasting
    * **RMSE:** 0.0682
    """
)

# --- 4. MAIN DASHBOARD ---
st.title("🌊 Mersey MetOcean Data Analysis 2025 (Lil-Mamba Model)")
st.markdown(f"**General View:** `{gen_start_date}` to `{gen_end_date}` | **Forecast Mode:** `{dt_end_obs}` (+1h)")

# --- ALERT BOX ---
if is_flooding:
    num_hours = len(flood_events)
    max_level = flood_events['Lil-Mamba Prediction'].max()
    
    st.error(
        f"🚨 **DANGER: FLOOD WARNING DETECTED (In General View)!**\n\n"
        f"Found **{num_hours} hours** where water level exceeds **{flood_threshold}m**.\n"
        f"🌊 **Highest Peak:** {max_level:.2f} m"
    )
    
    with st.expander("🔻 View Detailed Flood Times (Click to expand)", expanded=True):
        display_df = flood_events[['Time', 'Lil-Mamba Prediction']].copy()
        display_df.columns = ['Time of Occurrence', 'Predicted Level (m)']
        display_df['Predicted Level (m)'] = display_df['Predicted Level (m)'].map('{:.2f}'.format)
        
        st.dataframe(
            display_df, 
            use_container_width=True, 
            height=200, 
            hide_index=True 
        )
else:
    st.success(f"✅ **SAFE:** No flood risk detected. Water levels are below {flood_threshold} m.")

# --- KPI METRICS (Dùng Data General) ---
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Avg Sea Level", f"{df_general['Sea Surface Height'].mean():.2f} m")
kpi2.metric("Max Wave Height", f"{df_general['Significant Wave Height'].max():.2f} m")
kpi3.metric("Avg Wind Speed", f"{df_general['Wind Speed'].mean():.2f} m/s")
kpi4.metric("Avg Pressure", f"{df_general['Mean Sea Level Pressure'].mean():.0f} Pa")

st.markdown("---")

# ====================================================
# PHẦN 1: BIỂU ĐỒ QUAN TRỌNG NHẤT (SEA LEVEL FORECAST)
# ====================================================
# Layout [1, 6, 1] như cũ
c_pad1, c_hero, c_pad2 = st.columns([1, 6, 1]) 

with c_hero:
    st.subheader(f"Sea Level Forecast: {dt_start_obs.strftime('%d/%m %H:00')} - {dt_end_pred.strftime('%d/%m %H:00')}")

    fig_hero, ax_hero = plt.subplots(figsize=(7, 3.5), dpi=2500)

    # 1. Vẽ vùng Observed (24h) - Đường màu Tím
    p1 = ax_hero.plot(df_obs['Time'], df_obs['Sea Surface Height'], color='#9467bd', label='Observed (Past 24h)', linewidth=2.5, alpha=0.8)
    
    # 2. Vẽ vùng Forecast (1h sau đó) - Đường màu Đỏ/Cam (chỉ xuất hiện sau mốc NOW)
    # Lưu ý: Để đường liền mạch, ta cần lấy điểm cuối của obs nối với pred
    if not df_obs.empty and not df_pred.empty:
        # Tạo cầu nối để vẽ liền nét
        last_obs = df_obs.iloc[[-1]]
        df_pred_plot = pd.concat([last_obs, df_pred])
        p2 = ax_hero.plot(df_pred_plot['Time'], df_pred_plot['Lil-Mamba Prediction'], color='#d62728', label='Prediction (Next 1h)', linestyle='--', linewidth=2.5)
    else:
        # Fallback nếu thiếu data
        p2 = ax_hero.plot(df_pred['Time'], df_pred['Lil-Mamba Prediction'], color='#d62728', label='Prediction (Next 1h)', linestyle='--', linewidth=2.5)

    # 3. Vẽ vạch "NOW" ngăn cách
    ax_hero.axvline(x=dt_end_obs, color='black', linestyle=':', linewidth=1.5)
    ax_hero.text(dt_end_obs, ax_hero.get_ylim()[1], 'NOW', ha='right', va='top', fontsize=6, rotation=90, color='black')

    # 4. Ngưỡng cảnh báo (Toàn bộ trục)
    p3 = ax_hero.axhline(y=flood_threshold, color='#FF6600', linewidth=2.5, linestyle='-', label=f'Threshold ({flood_threshold}m)')
    
    # Draw Danger Zone
    ax_hero.axhspan(flood_threshold, 10, color='red', alpha=0.1, label='Flood Zone')

    # Fix Y-Axis Top
    ax_hero.set_ylim(top=4.21)

    # Font chữ
    ax_hero.set_ylabel('Sea Level (m)', fontsize=9)
    ax_hero.tick_params(axis='both', which='major', labelsize=8)

    # Xử lý Legend
    lines = p1 + p2 + [p3]
    labels_legend = [l.get_label() for l in lines]
    ax_hero.legend(
        lines, 
        labels_legend, 
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.3), 
        fancybox=True, 
        shadow=True, 
        ncol=3,
        fontsize=8
    )

    # Ngày tháng nghiêng
    plt.xticks(rotation=30, fontsize=8) 
    st.pyplot(fig_hero)

st.markdown("---")

# ====================================================
# PHẦN 2: CÁC BIỂU ĐỒ CÒN LẠI (DÙNG DATA GENERAL)
# ====================================================

# --- HÀNG 1 ---
c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Seawater Temperature Evolution")
    fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=300)
    ax1.plot(df_general['Time'], df_general['Potential Temperature'], label='Surface Temp', color='#ff7f0e', linewidth=2)
    ax1.plot(df_general['Time'], df_general['Bottom Temperature'], label='Bottom Temp', color='#1f77b4', linestyle='--', linewidth=2)
    ax1.set_ylabel('Temperature (°C)', fontsize=9)
    ax1.tick_params(labelsize=8)
    ax1.legend(fontsize=8)
    plt.xticks(rotation=30, fontsize=8)
    st.pyplot(fig1)

with c2:
    st.subheader("Wave Direction Frequency")
    dir_order = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    wave_counts = df_general['WaveDirCat'].value_counts().reindex(dir_order, fill_value=0)
    fig3, ax3 = plt.subplots(figsize=(6, 4), dpi=300)
    sns.barplot(x=wave_counts.index, y=wave_counts.values, ax=ax3, palette='viridis')
    ax3.set_ylabel('Count', fontsize=9)
    ax3.tick_params(labelsize=8)
    st.pyplot(fig3)

with c3:
    st.subheader("Wind Speed (m/s)")
    fig4, ax4 = plt.subplots(figsize=(6, 4), dpi=300)
    ax4.plot(df_general['Time'], df_general['Wind Speed'], color='#d62728', linewidth=1.5)
    ax4.fill_between(df_general['Time'], df_general['Wind Speed'], color='#d62728', alpha=0.1)
    ax4.set_ylabel('Speed (m/s)', fontsize=9)
    ax4.tick_params(labelsize=8)
    plt.xticks(rotation=30, fontsize=8)
    st.pyplot(fig4)

st.markdown("---")

# --- HÀNG 2 ---
c4, c5, c6 = st.columns(3)

with c4:
    st.subheader("Atmospheric Pressure (Pa)")
    fig5, ax5 = plt.subplots(figsize=(6, 4), dpi=300)
    ax5.plot(df_general['Time'], df_general['Mean Sea Level Pressure'], color='#8c564b', linewidth=2)
    ax5.set_ylabel('Pressure (Pa)', fontsize=9)
    ax5.tick_params(labelsize=8)
    plt.xticks(rotation=30, fontsize=8)
    st.pyplot(fig5)

with c5:
    st.subheader("Sea State Proportions")
    sea_counts = df_general['SeaStateCat'].value_counts().sort_index()
    sea_counts = sea_counts[sea_counts > 0]
    
    fig6, ax6 = plt.subplots(figsize=(6, 4), dpi=300)
    colors = sns.color_palette('Blues', len(sea_counts))
    wedges, texts, autotexts = ax6.pie(sea_counts, labels=None, autopct='%1.1f%%', startangle=140, colors=colors, pctdistance=0.85)
    ax6.legend(wedges, sea_counts.index, title="Sea States", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize='small')
    plt.setp(autotexts, size=8, weight="bold", color="white")
    st.pyplot(fig6)

with c6:
    st.subheader("Avg Surface Temp by Hour")
    df['Hour'] = df['Time'].dt.hour
    hourly_stats = df.groupby('Hour')[['Potential Temperature']].mean()
    
    fig7, ax7 = plt.subplots(figsize=(6, 4), dpi=300)
    ax7.plot(hourly_stats.index, hourly_stats['Potential Temperature'], marker='o', color='#ff7f0e')
    ax7.set_xlabel('Hour (0-23)', fontsize=9)
    ax7.set_ylabel('Temp (°C)', fontsize=9)
    ax7.tick_params(labelsize=8)
    st.pyplot(fig7)

st.markdown("---")

# --- HÀNG 3 ---
c7, c8 = st.columns(2)

with c7:
    st.subheader("Wave Direction vs Height")
    fig8, ax8 = plt.subplots(figsize=(6, 4), dpi=300)
    scatter = ax8.scatter(df['Mean Wave Direction'], df['Significant Wave Height'], alpha=0.5, s=15, c=df['Significant Wave Height'], cmap='viridis')
    ax8.set_xlabel('Direction (°)', fontsize=9)
    ax8.set_ylabel('Height (m)', fontsize=9)
    ax8.tick_params(labelsize=8)
    plt.colorbar(scatter, ax=ax8, label='Height (m)')
    st.pyplot(fig8)

with c8:
    st.subheader("Monthly Avg Temp")
    monthly_temp = df.groupby('Month', sort=False)['Potential Temperature'].mean()
    
    fig9, ax9 = plt.subplots(figsize=(6, 4), dpi=300)
    sns.barplot(x=monthly_temp.index, y=monthly_temp.values, ax=ax9, palette='magma')
    ax9.set_ylim(min(monthly_temp.values)*0.9, max(monthly_temp.values)*1.05)
    ax9.tick_params(labelsize=8)
    st.pyplot(fig9)
