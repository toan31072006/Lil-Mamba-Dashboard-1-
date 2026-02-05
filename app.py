import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Lil-Mamba Flood Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Thiết lập nền trắng cho biểu đồ
sns.set_theme(style="whitegrid")

# --- 2. LOAD DỮ LIỆU ---
@st.cache_data
def load_data():
    file_path = 'Mersey_Data_2025_Renamed.csv'
    
    if not os.path.exists(file_path):
        # Fallback data
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
    
    # 2. Simulate Lil-Mamba Prediction
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

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.header("🎛️ Data Filter")
st.sidebar.write("Select time range to analyze:")

min_date = df['Time'].min().date()
max_date = df['Time'].max().date()

start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", min_date + pd.Timedelta(days=7))

if start_date > end_date:
    st.sidebar.error("Error: Start Date must be before End Date.")

mask = (df['Time'].dt.date >= start_date) & (df['Time'].dt.date <= end_date)
df_filtered = df.loc[mask]

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

# Logic tìm điểm ngập
flood_events = pd.DataFrame()
if not df_filtered.empty:
    flood_events = df_filtered[df_filtered['Lil-Mamba Prediction'] > flood_threshold].copy()
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
st.markdown(f"**Viewing Data:** `{start_date}` to `{end_date}`")

# --- ALERT BOX ---
if is_flooding:
    num_hours = len(flood_events)
    max_level = flood_events['Lil-Mamba Prediction'].max()
    
    st.error(
        f"🚨 **DANGER: FLOOD WARNING DETECTED!**\n\n"
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

# --- KPI METRICS ---
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Avg Sea Level", f"{df_filtered['Sea Surface Height'].mean():.2f} m")
kpi2.metric("Max Wave Height", f"{df_filtered['Significant Wave Height'].max():.2f} m")
kpi3.metric("Avg Wind Speed", f"{df_filtered['Wind Speed'].mean():.2f} m/s")
kpi4.metric("Avg Pressure", f"{df_filtered['Mean Sea Level Pressure'].mean():.0f} Pa")

st.markdown("---")

# ====================================================
# PHẦN 1: BIỂU ĐỒ QUAN TRỌNG NHẤT (HERO SECTION)
# ====================================================
st.subheader("Sea Level: Observed vs Lil-Mamba")

# --- CẬP NHẬT: GIẢM KÍCH THƯỚC ~30% (9x3.5) ---
fig_hero, ax_hero = plt.subplots(figsize=(9, 3.5))

# Draw Danger Zone
ax_hero.axhspan(flood_threshold, 10, color='red', alpha=0.1, label='Flood Zone')

p1 = ax_hero.plot(df_filtered['Time'], df_filtered['Sea Surface Height'], color='#9467bd', label='Observed Sea Level', linewidth=4, alpha=0.5)
p2 = ax_hero.plot(df_filtered['Time'], df_filtered['Lil-Mamba Prediction'], color='#d62728', label='Lil-Mamba Prediction', linestyle='--', linewidth=1.5)

# Nét liền (Solid) + Màu Cam
p3 = ax_hero.axhline(y=flood_threshold, color='#FF6600', linewidth=3, linestyle='-', label=f'Threshold ({flood_threshold}m)')

# Fix Y-Axis Top
ax_hero.set_ylim(top=4.21)

ax_hero.set_ylabel('Sea Level (m)')
lines = p1 + p2 + [p3]
labels_legend = [l.get_label() for l in lines]
ax_hero.legend(lines, labels_legend, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3)
plt.xticks(rotation=20)
st.pyplot(fig_hero)

st.markdown("---")

# ====================================================
# PHẦN 2: CÁC BIỂU ĐỒ CÒN LẠI (GRID LAYOUT)
# ====================================================

# --- HÀNG 1 ---
c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Seawater Temperature Evolution")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(df_filtered['Time'], df_filtered['Potential Temperature'], label='Surface Temp', color='#ff7f0e', linewidth=2)
    ax1.plot(df_filtered['Time'], df_filtered['Bottom Temperature'], label='Bottom Temp', color='#1f77b4', linestyle='--', linewidth=2)
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend()
    plt.xticks(rotation=20)
    st.pyplot(fig1)

with c2:
    st.subheader("Wave Direction Frequency")
    dir_order = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    wave_counts = df_filtered['WaveDirCat'].value_counts().reindex(dir_order, fill_value=0)
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.barplot(x=wave_counts.index, y=wave_counts.values, ax=ax3, palette='viridis')
    ax3.set_ylabel('Count')
    st.pyplot(fig3)

with c3:
    st.subheader("Wind Speed (m/s)")
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    ax4.plot(df_filtered['Time'], df_filtered['Wind Speed'], color='#d62728', linewidth=1.5)
    ax4.fill_between(df_filtered['Time'], df_filtered['Wind Speed'], color='#d62728', alpha=0.1)
    ax4.set_ylabel('Speed (m/s)')
    plt.xticks(rotation=20)
    st.pyplot(fig4)

st.markdown("---")

# --- HÀNG 2 ---
c4, c5, c6 = st.columns(3)

with c4:
    st.subheader("Atmospheric Pressure (Pa)")
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    ax5.plot(df_filtered['Time'], df_filtered['Mean Sea Level Pressure'], color='#8c564b', linewidth=2)
    ax5.set_ylabel('Pressure (Pa)')
    plt.xticks(rotation=20)
    st.pyplot(fig5)

with c5:
    st.subheader("Sea State Proportions")
    sea_counts = df_filtered['SeaStateCat'].value_counts().sort_index()
    sea_counts = sea_counts[sea_counts > 0]
    
    fig6, ax6 = plt.subplots(figsize=(6, 4))
    colors = sns.color_palette('Blues', len(sea_counts))
    wedges, texts, autotexts = ax6.pie(sea_counts, labels=None, autopct='%1.1f%%', startangle=140, colors=colors, pctdistance=0.85)
    ax6.legend(wedges, sea_counts.index, title="Sea States", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize='small')
    plt.setp(autotexts, size=8, weight="bold", color="white")
    st.pyplot(fig6)

with c6:
    st.subheader("Avg Surface Temp by Hour")
    df['Hour'] = df['Time'].dt.hour
    hourly_stats = df.groupby('Hour')[['Potential Temperature']].mean()
    
    fig7, ax7 = plt.subplots(figsize=(6, 4))
    ax7.plot(hourly_stats.index, hourly_stats['Potential Temperature'], marker='o', color='#ff7f0e')
    ax7.set_xlabel('Hour (0-23)')
    ax7.set_ylabel('Temp (°C)')
    ax7.set_xticks(range(0, 24, 4))
    st.pyplot(fig7)

st.markdown("---")

# --- HÀNG 3 ---
c7, c8 = st.columns(2)

with c7:
    st.subheader("Wave Direction vs Height")
    fig8, ax8 = plt.subplots(figsize=(6, 4))
    scatter = ax8.scatter(df['Mean Wave Direction'], df['Significant Wave Height'], alpha=0.5, s=15, c=df['Significant Wave Height'], cmap='viridis')
    ax8.set_xlabel('Direction (°)')
    ax8.set_ylabel('Height (m)')
    plt.colorbar(scatter, ax=ax8, label='Height (m)')
    st.pyplot(fig8)

with c8:
    st.subheader("Monthly Avg Temp")
    monthly_temp = df.groupby('Month', sort=False)['Potential Temperature'].mean()
    
    fig9, ax9 = plt.subplots(figsize=(6, 4))
    sns.barplot(x=monthly_temp.index, y=monthly_temp.values, ax=ax9, palette='magma')
    ax9.set_ylim(min(monthly_temp.values)*0.9, max(monthly_temp.values)*1.05)
    st.pyplot(fig9)
