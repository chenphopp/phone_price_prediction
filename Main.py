# main.py - หน้าหลัก
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Phone Number Price Prediction",
    page_icon="📱",
    layout="wide"
)

# Main page content
st.title("📱 Phone Number Price Prediction")
st.write("Price prediction using AI and Machine Learning")

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("🤖 AI-Powered Prediction (Gemini)")
    st.write("วิเคราะห์และประเมินราคาเบอร์โทรศัพท์ด้วย Google Gemini AI")
    st.write("• วิเคราะห์รูปแบบตัวเลข")
    st.write("• พิจารณาความเชื่อทางวัฒนธรรมไทย")
    st.write("• ให้เหตุผลการประเมินราคาอย่างละเอียด")
    if st.button("🚀 เข้าสู่หน้า AI Prediction", key="ai_predict"):
        st.switch_page("pages/AI_Prediction.py")

with col2:
    st.subheader("🧠 Machine Learning Model")
    st.write("ประเมินราคาด้วย Machine Learning Model")
    st.write("• ใช้ข้อมูลราคาจริงจากตลาด")
    st.write("🚧 **Under developing**")
    if st.button("⏳ ML Model (Coming Soon)", key="ml_predict", disabled=True):
        st.switch_page("pages/ML_Prediction.py")


st.markdown("---")
st.caption("Nida")