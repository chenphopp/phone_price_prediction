# pages/1_📱_Phone_Prediction.py
import streamlit as st
import google.generativeai as genai
import re

# Page configuration
st.set_page_config(
    page_title="Phone Number Price Prediction with AI",
    page_icon="📱",
    layout="centered"
)

# Page title
st.title("📱 Phone Number Price Prediction with AI")

st.write("Enter a phone number and get AI-powered analysis and price prediction in Thai Baht!")

# Sidebar for API key
# st.sidebar.header("Configuration")
# api_key = st.sidebar.text_input("Enter your Gemini API Key:", type="password")
api_key = "AIzaSyD98blARCQvuKC2nV-wmzRq2LGfIf95W70"

if api_key:
    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')

# Get user input
phone_number = st.text_input("What's your phone number?", placeholder="e.g., 081-234-5678")

# Function to analyze Thai phone number and return response in Thai
def analyze_phone_number(number):
    # Clean the phone number
    cleaned = re.sub(r'[^\d]', '', number)
    
    prompt = f"""
    วิเคราะห์เบอร์โทรศัพท์ไทย: {number}
    
    กรุณาตอบเป็นภาษาไทย โดยแสดงผลในรูปแบบดังนี้:
    
    1. วิเคราะห์รูปแบบของเบอร์ (เลขซ้ำ, เลขเรียง, เลขมงคล)
    2. ความหมายทางวัฒนธรรมไทย (เลขมงคล/เลขอัปมงคล)
    3. คะแนนความจำง่าย (1-10)
    4. ประเมินช่วงราคาที่เหมาะสมในตลาดไทย (บาท) โดยพิจารณา:
       - รูปแบบของตัวเลข
       - เลขมงคล (เช่น 8, 9)
       - ความง่ายในการจดจำ
       - ลักษณะของเลขเรียง
    
    กรุณาวิเคราะห์อย่างละเอียด พร้อมให้เหตุผลประกอบและช่วงราคาที่เหมาะสมตามจริงในตลาดไทย
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการวิเคราะห์เบอร์โทรศัพท์: {str(e)}"

# Analyze button
if phone_number and api_key:
    if st.button("🔍 Analyze & Predict Price"):
        if len(re.sub(r'[^\d]', '', phone_number)) >= 9:  # Basic validation
            with st.spinner("Analyzing phone number..."):
                analysis = analyze_phone_number(phone_number)
                
                st.subheader("📊 Analysis Results")
                st.write(analysis)
                
                # Add some visual elements
                st.success("Analysis completed!")
                
        else:
            st.error("Please enter a valid phone number with at least 9 digits.")

elif phone_number and not api_key:
    st.warning("Please enter your Gemini API key in the sidebar to get analysis.")

# Information section
st.markdown("---")
st.subheader("ℹ️ How it works")
st.write("""
- Enter a Thai phone number
- AI analyzes digit patterns, lucky numbers, and memorability
- Get price prediction based on Thai market preferences
- Considers cultural factors like lucky number 8 and 9
""")

# st.subheader("🔑 API Key Setup")
# st.write("""
# 1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
# 2. Create a new API key
# 3. Enter it in the sidebar
# """)

# Footer
st.markdown("---")
st.caption("Nida")
