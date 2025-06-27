import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Flight Delay & Weather Dashboard",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Load Data ----------------
def load_data():
    try:
        df = pd.read_csv("merged.csv")
        if 'hour' in df.columns:
            df['hour_str'] = df['hour'].apply(lambda x: f"{int(x):02d}:00" if pd.notnull(x) else None)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

# ---------------- Title Section ----------------
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🛫 Flight Delay & Weather Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: grey;'>Understand delay patterns & weather impact on flights 🚀</h4>", unsafe_allow_html=True)
st.markdown("---")

if df.empty:
    st.warning("⚠️ No data available. Please check the CSV file path.")
    st.stop()

# ---------------- Sidebar Filters ----------------
st.sidebar.markdown("### ✈️ **Filter Flights**")
carriers = df['carrier'].dropna().unique()
selected_carrier = st.sidebar.selectbox("Select Airline", options=carriers)

origins = df['origin'].dropna().unique()
selected_origin = st.sidebar.selectbox("Select Origin Airport", options=origins)

filtered_df = df[(df['carrier'] == selected_carrier) & (df['origin'] == selected_origin)]

# ---------------- Key Metrics ----------------
st.markdown("### 📊 Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("🕒 Avg Departure Delay", f"{filtered_df['dep_delay'].mean():.2f} min")
col2.metric("🛬 Avg Arrival Delay", f"{filtered_df['arr_delay'].mean():.2f} min")
col3.metric("❌ Cancelled Flights", int(filtered_df['cancelled'].sum()) if 'cancelled' in filtered_df.columns else "N/A")

# ---------------- Tabs Section ----------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Delay Trends", "🌤️ Weather Impact", "📊 Extra Analysis", "🤖 ML Model"])

   # Tab 1: Delay Trends
# Tab 1: Delay Trends
with tab1:
    st.subheader("📍 Hourly Average Departure Delay")

    # Ensure hour_str exists
    if 'hour_str' not in filtered_df.columns and 'hour' in filtered_df.columns:
        filtered_df['hour_str'] = filtered_df['hour'].apply(lambda x: f"{int(x):02d}:00" if pd.notnull(x) else None)

    # Check for data
    if not filtered_df.empty and 'hour_str' in filtered_df.columns and 'dep_delay' in filtered_df.columns:
        hourly = filtered_df.groupby('hour_str')['dep_delay'].mean().reset_index()

        fig1 = px.line(
            hourly,
            x='hour_str',
            y='dep_delay',
            markers=True,
            title="📈 Average Departure Delay by Hour",
            labels={'hour_str': 'Hour of Day', 'dep_delay': 'Avg Delay (min)'}
        )
        fig1.update_traces(line_color='royalblue')
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.warning("⚠️ No data available to generate line chart.")

    # Interactive Heatmap
    st.subheader("🧊 Interactive Heatmap: Delay by Hour vs Day")
    if 'day' in filtered_df.columns and 'hour_str' in filtered_df.columns:
        heat_data = filtered_df.pivot_table(
            index='hour_str',
            columns='day',
            values='dep_delay',
            aggfunc='mean'
        ).fillna(0)

        import plotly.graph_objects as go
        fig2 = go.Figure(
            data=go.Heatmap(
                z=heat_data.values,
                x=heat_data.columns,   # days
                y=heat_data.index,     # hours
                colorscale='RdBu_r',
                hoverongaps=False,
                hovertemplate="📅 Day %{x}<br>⏰ Hour %{y}<br>🕒 Avg Delay %{z:.1f} min<extra></extra>"
            )
        )
        fig2.update_layout(
            title="Heatmap: Avg Departure Delay (Hour vs Day)",
            xaxis_title="Day of Month",
            yaxis_title="Hour of Day",
            height=500
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("⚠️ Not enough data to display heatmap.")





# Tab 2: Weather Impact
with tab2:
    st.subheader("🌡️ Temperature vs Delay")
    if 'temperature' in filtered_df.columns and 'dep_delay' in filtered_df.columns:
        fig3 = px.scatter(filtered_df, x='temperature', y='dep_delay', color='precip',
                          hover_data=['carrier', 'origin', 'dest'])
        st.plotly_chart(fig3)

    st.subheader("📘 Correlation Between Weather and Delay")
    weather_cols = ['temperature', 'wind_speed', 'precip', 'dep_delay', 'arr_delay']
    valid_cols = [col for col in weather_cols if col in filtered_df.columns]
    if len(valid_cols) > 1:
        corr = filtered_df[valid_cols].corr()
        fig_corr, ax_corr = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax_corr)
        st.pyplot(fig_corr)

# Tab 3: Extra Analysis
with tab3:
    st.subheader("📅 Flights per Month")
    if 'month' in filtered_df.columns:
        month_counts = filtered_df['month'].value_counts().sort_index().reset_index()
        month_counts.columns = ['Month', 'Flights']
        fig4 = px.bar(month_counts, x='Month', y='Flights', color='Flights')
        st.plotly_chart(fig4)

    st.subheader("📍 Average Arrival Delay by Destination")
    if 'dest' in filtered_df.columns and 'arr_delay' in filtered_df.columns:
        avg_delay = filtered_df.groupby('dest')['arr_delay'].mean().reset_index()
        fig5 = px.bar(avg_delay, x='dest', y='arr_delay')
        st.plotly_chart(fig5)

    # ✅ NEW GRAPH replacing boxplot
    st.subheader("✈️ Average Departure Delay by Airline")
    if 'carrier' in filtered_df.columns and 'dep_delay' in filtered_df.columns:
        delay_by_carrier = (
            filtered_df.groupby('carrier')['dep_delay']
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        fig6 = px.bar(
            delay_by_carrier,
            x='carrier',
            y='dep_delay',
            color='dep_delay',
            color_continuous_scale='Reds',
            labels={'carrier': 'Airline', 'dep_delay': 'Avg Departure Delay (min)'},
            title="Average Departure Delay by Airline"
        )
        fig6.update_layout(height=400)
        st.plotly_chart(fig6, use_container_width=True)
    else:
        st.warning("⚠️ 'carrier' or 'dep_delay' column not found.")

    
    st.subheader("⏰ Early vs Late Arrivals")
    if 'arr_delay' in filtered_df.columns:
        filtered_df['arrival_status'] = filtered_df['arr_delay'].apply(lambda x: 'Early' if x < 0 else 'Late')
        status_counts = filtered_df['arrival_status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        fig8 = px.pie(status_counts, names='Status', values='Count')
        st.plotly_chart(fig8)

    st.subheader("📈 Monthly Avg Arrival Delay")
    if 'month' in filtered_df.columns and 'arr_delay' in filtered_df.columns:
        monthly_avg = filtered_df.groupby('month')['arr_delay'].mean().reset_index()
        fig9 = px.line(monthly_avg, x='month', y='arr_delay', markers=True)
        st.plotly_chart(fig9)

    st.subheader("🧭 Route-wise Avg Arrival Delay")
    if 'origin' in filtered_df.columns and 'dest' in filtered_df.columns and 'arr_delay' in filtered_df.columns:
        filtered_df['route'] = filtered_df['origin'] + " → " + filtered_df['dest']
        route_delay = filtered_df.groupby('route')['arr_delay'].mean().reset_index()
        fig10 = px.bar(route_delay, x='route', y='arr_delay')
        st.plotly_chart(fig10)


# Tab 4: ML Model
with tab4:
    st.subheader("🤖 Regression Model: Predict Departure Delay")
    df_model_dep = df[['month', 'day', 'hour', 'distance', 'air_time', 'arr_delay', 'dep_delay']].dropna()
    X_dep = df_model_dep.drop('dep_delay', axis=1)
    y_dep = df_model_dep['dep_delay']
    X_train_dep, X_test_dep, y_train_dep, y_test_dep = train_test_split(X_dep, y_dep, test_size=0.2, random_state=42)
    model_dep = RandomForestRegressor(n_estimators=100, random_state=42)
    model_dep.fit(X_train_dep, y_train_dep)
    y_pred_dep = model_dep.predict(X_test_dep)

    st.markdown("##### 🎯 Departure Delay Model Evaluation")
    st.write(f"MAE: {mean_absolute_error(y_test_dep, y_pred_dep):.2f}")
    st.write(f"RMSE: {mean_squared_error(y_test_dep, y_pred_dep, squared=False):.2f}")
    st.write(f"R² Score: {r2_score(y_test_dep, y_pred_dep):.2f}")

    st.markdown("---")
    st.markdown("#### ✏️ Try Departure Delay Prediction")
    month_input = st.slider("Month", 1, 12, 1)
    day_input = st.slider("Day", 1, 31, 1)
    hour_input = st.slider("Hour", 0, 23, 12)
    distance_input = st.number_input("Distance", value=500)
    airtime_input = st.number_input("Air Time (mins)", value=100)
    arr_delay_input = st.number_input("Arrival Delay (mins)", value=0)

    user_input_dep = pd.DataFrame({
        'month': [month_input],
        'day': [day_input],
        'hour': [hour_input],
        'distance': [distance_input],
        'air_time': [airtime_input],
        'arr_delay': [arr_delay_input]
    })
    prediction_dep = model_dep.predict(user_input_dep)
    st.success(f"🎯 Predicted Departure Delay: {prediction_dep[0]:.2f} minutes")

    st.markdown("---")
    st.subheader("🛬 Predict Arrival Delay")
    df_model_arr = df[['month', 'day', 'hour', 'distance', 'air_time', 'dep_delay', 'arr_delay']].dropna()
    X_arr = df_model_arr.drop('arr_delay', axis=1)
    y_arr = df_model_arr['arr_delay']
    X_train_arr, X_test_arr, y_train_arr, y_test_arr = train_test_split(X_arr, y_arr, test_size=0.2, random_state=42)
    model_arr = RandomForestRegressor(n_estimators=100, random_state=42)
    model_arr.fit(X_train_arr, y_train_arr)
    y_pred_arr = model_arr.predict(X_test_arr)

    st.markdown("##### 📊 Arrival Delay Model Evaluation")
    st.write(f"MAE: {mean_absolute_error(y_test_arr, y_pred_arr):.2f}")
    st.write(f"RMSE: {mean_squared_error(y_test_arr, y_pred_arr, squared=False):.2f}")
    st.write(f"R² Score: {r2_score(y_test_arr, y_pred_arr):.2f}")

    dep_delay_input = st.number_input("Departure Delay (mins)", value=0)
    user_input_arr = pd.DataFrame({
        'month': [month_input],
        'day': [day_input],
        'hour': [hour_input],
        'distance': [distance_input],
        'air_time': [airtime_input],
        'dep_delay': [dep_delay_input]
    })
    prediction_arr = model_arr.predict(user_input_arr)
    st.success(f"🎯 Predicted Arrival Delay: {prediction_arr[0]:.2f} minutes")

# ---------------- Footer ----------------
st.markdown("""<hr style='margin-top:30px;margin-bottom:10px'>""", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; font-size:14px;'>Created by <a href='https://www.linkedin.com/in/abhishek-pandey-a4aa06220/' target='_blank'>Abhishek Pandey</a> — Data Science & AI Dashboard</p>",
    unsafe_allow_html=True
)  