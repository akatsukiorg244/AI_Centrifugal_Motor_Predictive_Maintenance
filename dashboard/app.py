import streamlit as st
import numpy as np
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Pump Digital Twin Dashboard",
    page_icon="⚙️",
    layout="wide"
)

# Custom CSS for UI styling & color-coded badges
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #007bff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .status-healthy { background-color: #28a745; color: white; padding: 4px 10px; border-radius: 5px; font-weight: bold; }
    .status-good { background-color: #20c997; color: white; padding: 4px 10px; border-radius: 5px; font-weight: bold; }
    .status-neutral { background-color: #ffc107; color: black; padding: 4px 10px; border-radius: 5px; font-weight: bold; }
    .status-needs-attention { background-color: #fd7e14; color: white; padding: 4px 10px; border-radius: 5px; font-weight: bold; }
    .status-failure { background-color: #dc3545; color: white; padding: 4px 10px; border-radius: 5px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("⚙️ Centrifugal Pump Digital Twin")
st.caption("Real-Time Telemetry Processing & Predictive Maintenance Diagnostic Engine")

# ==========================================
# 1. TOP SECTION: MOTOR CONTROLS & PRESETS
# ==========================================
st.subheader("1. Motor Power & Operational Set States")

col_motor1, col_motor2, col_motor3 = st.columns([1, 1.5, 2.5])

with col_motor1:
    motor_state = st.toggle("🔌 Motor Power State", value=True)
    motor_status_text = "RUNNING" if motor_state else "STOPPED"
    status_color = "green" if motor_state else "red"
    st.markdown(f"**Motor Status:** <span style='color:{status_color}; font-weight:bold;'>{motor_status_text}</span>", unsafe_allow_html=True)

with col_motor2:
    set_state = st.selectbox(
        "📋 Load Operational Preset State",
        ["Normal", "Lightly Used", "Under Stress", "Heavily Used", "Roughly Used"]
    )

# Preset Dictionary Mapping
preset_defaults = {
    "Normal": {"rpm": 1800, "flow": 80.0, "p_in": 1.8, "p_out": 4.5, "vib": 1.2, "temp": 45.0},
    "Lightly Used": {"rpm": 1200, "flow": 55.0, "p_in": 2.2, "p_out": 3.8, "vib": 0.8, "temp": 38.0},
    "Under Stress": {"rpm": 2400, "flow": 110.0, "p_in": 1.2, "p_out": 6.5, "vib": 4.8, "temp": 72.0},
    "Heavily Used": {"rpm": 2900, "flow": 135.0, "p_in": 0.9, "p_out": 8.0, "vib": 8.5, "temp": 88.0},
    "Roughly Used": {"rpm": 3400, "flow": 150.0, "p_in": 0.5, "p_out": 9.5, "vib": 13.8, "temp": 105.0}
}

active_preset = preset_defaults[set_state]

with col_motor3:
    st.info(f"**Current Preset:** `{set_state}` — Operating at **{active_preset['rpm']} RPM** target.")

st.divider()

# ==========================================
# 2. INPUT PARAMETERS (IN PARTICULAR RANGES)
# ==========================================
st.subheader("2. Physical Input Telemetry Parameters")

col_in1, col_in2, col_in3 = st.columns(3)

with col_in1:
    rpm = st.slider("Motor Speed (RPM)", min_value=0, max_value=3600, value=active_preset["rpm"], step=50)
    flow = st.slider("Flow Rate Q (L/min)", min_value=0.0, max_value=200.0, value=active_preset["flow"], step=1.0)

with col_in2:
    p_in = st.slider("Suction Pressure P_in (bar)", min_value=0.1, max_value=3.0, value=active_preset["p_in"], step=0.1)
    p_out = st.slider("Discharge Pressure P_out (bar)", min_value=0.5, max_value=12.0, value=active_preset["p_out"], step=0.1)

with col_in3:
    vib = st.slider("Vibration RMS (mm/s)", min_value=0.1, max_value=20.0, value=active_preset["vib"], step=0.1)
    temp = st.slider("Bearing & Winding Temp (°C)", min_value=20.0, max_value=120.0, value=active_preset["temp"], step=1.0)

st.divider()

# ==========================================
# 3. DIAGNOSTICS & PART STATE CLASSIFICATION
# ==========================================
st.subheader("3. Component Part States & Health Index")

# Helper function to categorize state into exact 5 requested buckets
def classify_health(score):
    if score >= 85:
        return "Healthy", "status-healthy"
    elif score >= 70:
        return "Good", "status-good"
    elif score >= 50:
        return "Neutral", "status-neutral"
    elif score >= 25:
        return "Needs Attention", "status-needs-attention"
    else:
        return "Failure", "status-failure"

if not motor_state:
    st.warning("⚠️ Motor is currently Powered OFF. Telemetry diagnostics suspended.")
else:
    # Component Health Calculations (0 to 100 Scale)
    impeller_score = 100 - max(0, (1.2 - p_in) * 35) - max(0, (rpm - 2800) * 0.03)
    bearing_score = 100 - (vib * 4.5) - max(0, (temp - 60) * 0.8)
    delta_p = max(0, p_out - p_in)
    seal_score = 100 - (delta_p * 5.0) - max(0, (temp - 70) * 0.9)
    winding_score = 100 - max(0, (temp - 50) * 1.3) - (rpm / 3600 * 15)
    casing_score = 100 - max(0, (p_out - 8.0) * 12) - (vib * 2.0)

    parts_data = [
        {"Part": "Impeller", "Score": max(0, min(100, impeller_score))},
        {"Part": "Rolling Bearings", "Score": max(0, min(100, bearing_score))},
        {"Part": "Mechanical Seal", "Score": max(0, min(100, seal_score))},
        {"Part": "Motor Winding", "Score": max(0, min(100, winding_score))},
        {"Part": "Volute Casing", "Score": max(0, min(100, casing_score))}
    ]

    col_parts, col_rul = st.columns([3, 2])

    with col_parts:
        part_rows = []
        for p in parts_data:
            state_label, css_class = classify_health(p["Score"])
            part_rows.append({
                "Component Part": p["Part"],
                "Health Score": f"{p['Score']:.1f} / 100",
                "Part State": f'<span class="{css_class}">{state_label}</span>'
            })
        
        df_parts = pd.DataFrame(part_rows)
        st.write(df_parts.to_html(escape=False, index=False), unsafe_allow_html=True)

    # ==========================================
    # 4. DYNAMIC RUL (REMAINING USEFUL LIFE)
    # ==========================================
    with col_rul:
        baseline_hours = 10000.0
        overall_health = np.mean([p["Score"] for p in parts_data])
        min_health = min([p["Score"] for p in parts_data])
        
        # Dynamic non-unitary RUL calculation
        stress_factor = (100 - min_health) / 100.0
        degradation_rate = 1.0 + (stress_factor ** 2.5) * 8.0
        
        calculated_rul_hours = int((overall_health / 100.0) * (baseline_hours / degradation_rate))
        calculated_rul_days = round(calculated_rul_hours / 24, 1)

        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric(
            label="⏱️ Dynamic Remaining Useful Life (RUL)",
            value=f"{calculated_rul_hours:,} Hours",
            delta=f"~{calculated_rul_days} Days remaining" if calculated_rul_hours > 0 else "CRITICAL RISK",
            delta_color="normal" if calculated_rul_hours > 1000 else "inverse"
        )
        
        # Life Percentage Bar
        rul_pct = min(100, int((calculated_rul_hours / baseline_hours) * 100))
        st.progress(rul_pct / 100, text=f"Pump Useful Life Remaining: {rul_pct}%")
        
        if min_health < 25:
            st.error("🚨 Immediate Maintenance Recommended! Component state in FAILURE zone.")
        elif min_health < 50:
            st.warning("⚠️ High Wear Detected. Schedule maintenance soon.")
        else:
            st.success("✅ Operational Parameters within safe working thresholds.")
        st.markdown("</div>", unsafe_allow_html=True)