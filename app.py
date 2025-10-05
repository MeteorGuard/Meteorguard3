import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

try:
    import geocoder
    import numpy as np
except ImportError:
    st.warning("Gerekli kütüphaneler ('geocoder' ve 'numpy') yüklü değil. Konum/Mesafe gibi bazı özellikler devre dışı bırakılacaktır.")
    geocoder = None
    np = None


st.set_page_config(
    page_title="MeteorGuard | NASA Space Apps",
    page_icon="☄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("☄️ MeteorGuard: A Real-Time Meteor Impact Tracking System")
st.markdown("""
    Developed for the *NASA Space Apps Izmir Hackathon*, this application harnesses NASA's publicly available data to
    visualize meteor impacts in real-time and generate a dynamic risk assessment map.
""")

@st.cache_data(ttl=3600)
def get_nasa_fireball_data(days=30):
    """Retrieves fireball data from NASA's API."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        start_date_str = start_date.strftime("%Y-%m-%d")
        
        api_url = f"https://ssd-api.jpl.nasa.gov/fireball.api?date-min={start_date_str}&req-loc=true"
        
        response = requests.get(api_url)
        response.raise_for_status() 
        
        data = response.json()
        headers = data['fields']
        records = data['data']
        
        df = pd.DataFrame(records, columns=headers)
    
        for col in ['lat', 'lon', 'energy', 'impact_e']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df.rename(columns={
            'lat': 'latitude',
            'lon': 'longitude',
            'energy': 'energy_kton'
        }, inplace=True)

        return df
    
    except requests.exceptions.RequestException as e:
        st.error(f"API'den veri çekerken bir istek hatası oluştu: {e}")
        return pd.DataFrame() 
    except Exception as e:
        st.error(f"Veri işleme sırasında bir hata oluştu: {e}")
        return pd.DataFrame()

def calculate_risk_score(df):
    """Computes a simplified risk score based on the meteor's energy level."""
    if 'energy_kton' in df.columns and not df['energy_kton'].empty and df['energy_kton'].max() > 0:
        max_energy = df['energy_kton'].max()
        df['risk_score'] = (df['energy_kton'] / max_energy) * 100
        df['risk_score'] = df['risk_score'].round(2)
    else:
        df['risk_score'] = 0.0
    return df

def predict_future_impacts(df, days_ahead=7):
    """Simple linear prediction of future impacts."""
    if df.empty or 'date' not in df.columns:
        return 0
    df_clean = df.copy()
    df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce').dropna()
    if df_clean['date'].empty:
         return 0
         
    duration_days = (df_clean['date'].max() - df_clean['date'].min()).days
    if duration_days <= 0: duration_days = 1 
    
    avg_per_day = len(df_clean) / duration_days
    return int(avg_per_day * days_ahead)


st.sidebar.header("Application Controls")
st.sidebar.markdown("""
    Customize the dataset and visualization using the filters below.
""")

days_to_load = st.sidebar.slider(
    'Days of Data to Display',
    min_value=1, max_value=365, value=30, step=1
)

df = get_nasa_fireball_data(days=days_to_load)


if not df.empty:
    df.dropna(subset=['latitude', 'longitude', 'energy_kton'], inplace=True)
    df = calculate_risk_score(df)
    
   
    min_risk = st.sidebar.slider(
        'Minimum Risk Score',
        min_value=0.0, max_value=100.0,
        value=0.0, step=1.0
    )

    filtered_df = df[df['risk_score'] >= min_risk].copy()

   
    if geocoder and np:
        try:
            user_location = geocoder.ip('me')
            if user_location.lat and user_location.lng:
                st.sidebar.write(f"🌍 Approximate Location: **{user_location.city}, {user_location.country}**")
                
                
                df['distance'] = np.sqrt((df['latitude'] - user_location.lat)**2 + (df['longitude'] - user_location.lng)**2)
                nearest = df.sort_values('distance').head(1)
                
                if not nearest.empty:
                    st.sidebar.success(
                        f"☄️ **Nearest Meteor:** {nearest.iloc[0]['date']} \n\n"
                        f"📍 **Distance:** {nearest.iloc[0]['distance']:.2f}° | "
                        f"⚡ **Energy:** {nearest.iloc[0]['energy_kton']:.2f} kton"
                    )
        except Exception:
            st.sidebar.warning("Couldn't detect your location automatically.")
    
    
    max_risk = df['risk_score'].max() if not df['risk_score'].empty else 0
    if max_risk > 80:
        st.sidebar.error("🚨 **High-Risk Meteor Detected!**")
    elif max_risk > 50:
        st.sidebar.warning("🟠 **Medium Meteor Activity Detected.**")
    else:
        st.sidebar.success("🟢 **Low Meteor Activity** — No Major Threats.")


    st.markdown("---")
    st.subheader("Global Meteor Impact Map")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Meteors", len(df))
    col2.metric("Filtered Results", len(filtered_df))
    col3.metric("Maximum Energy", f"{df['energy_kton'].max():.2f} kton" if not df['energy_kton'].empty else "0.00 kton")

    if not filtered_df.empty:
        
        fig = px.scatter_geo(
            filtered_df,
            lat='latitude',
            lon='longitude',
            color='risk_score',
            hover_name='date',
            hover_data={
                'energy_kton': ':.2f',
                'risk_score': True,
                'latitude': ':.2f',
                'longitude': ':.2f'
            },
            color_continuous_scale=px.colors.sequential.Inferno,
            size='energy_kton',
            title='Meteor Impacts and Risk Assessment Map',
            projection="natural earth",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        
        st.subheader("Tabulated Meteor Data")
        st.dataframe(filtered_df[['date', 'latitude', 'longitude', 'energy_kton', 'risk_score']].sort_values('risk_score', ascending=False), use_container_width=True)
        
        st.download_button(
            "💾 Download Meteor Data (CSV)",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='meteor_data.csv',
            mime='text/csv'
        )
        
    else:
        st.warning("No meteor data matches the selected filters. Please adjust your criteria.")

else:
    st.error("Failed to retrieve data from the API. Please try again later.")

st.markdown("---")
st.header("🧠 Advanced Analytics and Tools")


if not df.empty and 'date' in df.columns:
    try:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        daily_counts = df.groupby(df['date'].dt.date).size().reset_index(name='count')
        fig2 = px.line(
            daily_counts,
            x='date',
            y='count',
            title='📊 Meteor Frequency Over Time',
            markers=True,
            template='plotly_dark'
        )
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.warning(f"Couldn't render daily frequency chart: {e}")

predicted_hits = predict_future_impacts(df)
st.info(f"🧮 Estimated meteor impacts in the next 7 days: **{predicted_hits} events** (approx.)")


st.markdown("---")
enable_3d = st.checkbox("🌌 Enable 3D Globe Visualization")

if enable_3d and not df.empty:
    fig3 = go.Figure(
        data=go.Scatter3d(
            x=df['longitude'],
            y=df['latitude'],
            z=df['energy_kton'],
            mode='markers',
            marker=dict(
                size=5,
                color=df['risk_score'],
                colorscale='Inferno',
                opacity=0.8,
                colorbar=dict(title='Risk Level')
            )
        )
    )
    fig3.update_layout(
        title="🌍 3D Meteor Impact Energy Visualization",
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title='Energy (kton)'
        ),
        template='plotly_dark',
        height=700
    )
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

st.header("🧮 Meteor Impact Simulator")

mass = st.number_input("Meteor Mass (tons)", min_value=0.1, value=10.0, step=0.1, key="sim_mass")
velocity = st.number_input("Velocity (km/s)", min_value=1.0, value=20.0, step=0.5, key="sim_velocity")
density = st.selectbox("Composition", ["Iron", "Stone", "Ice"], key="sim_density")
angle = st.slider("Impact Angle (degrees)", 10, 90, 45, key="sim_angle")
surface = st.selectbox("Impact Surface", ["Land", "Ocean", "Mountain", "Ice Field"], key="sim_surface")

if st.button("💥 Simulate Impact"):
   
    m_kg = mass * 1000  # ton -> kg
    v_ms = velocity * 1000  # km/s -> m/s
    
    E_joule = 0.5 * m_kg * (v_ms ** 2)
    
    E_kton = E_joule / 4.184e12  
    
    
    radius_km = (E_kton ** (1/3)) * 2
    
    st.success(f"💣 **Estimated Energy:** {E_kton:.2f} kilotons TNT")
    st.info(f"🌍 **Approximate Impact Radius:** {radius_km:.2f} km")
    
    
    if E_kton > 5000:
        st.error("🚨 **CATACLYSMIC EVENT:** Tsunami, global climate change, and mass extinction possible!")
    elif E_kton > 1000:
        st.error("🚨 **MASSIVE EVENT:** Global-level destruction possible.")
    elif E_kton > 100:
        st.warning("⚠️ **REGIONAL DAMAGE:** Significant local blast and thermal radiation expected.")
    else:
        st.success("🟢 **LOCAL IMPACT:** Limited damage or atmospheric explosion.")

    st.markdown(f"**Surface Type:** {surface} | **Composition:** {density} | **Angle:** {angle}°")
    st.markdown("---")


st.header("🌠 Meteor Info Corner")

st.markdown("""
Meteors, also known as meteoroids when in space, are fragments of rock or metal that enter Earth's atmosphere. 
Most are small and burn up upon entry, but larger meteors can cause significant damage.
""")

st.subheader("💡 Interesting Facts")
st.markdown("""
- The largest recorded meteor event was the **Tunguska event** (1908, Siberia), which flattened ~2,000 km² of forest. 
- Earth is hit by about **100 tons** of meteoritic dust daily. 
- Most meteors are composed of stone or iron. 
- Meteor velocities typically range between 11 km/s and 72 km/s. 
- Large meteor impacts have historically caused **mass extinctions**, like the one that wiped out the dinosaurs.
""")

st.subheader("❓ Test Your Knowledge")
q1 = st.radio(
    "Which of the following is NOT a typical meteor composition?",
    ("Iron", "Stone", "Ice", "Plastic")
)

if q1:
    if q1 == "Plastic":
        st.success("✅ Correct! Meteors are mostly made of stone, iron, or ice.")
    else:
        st.error("❌ Not quite. Remember, meteors are natural rocks or metals!")
