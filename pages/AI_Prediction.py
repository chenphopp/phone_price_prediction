# pages/1_📱_Phone_Prediction.py
import streamlit as st
import google.generativeai as genai
import re
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json

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

# Add example buttons
st.write("**ลองกับเบอร์ตัวอย่าง:**")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("081-888-8888"):
        phone_number = "081-888-8888"
        st.rerun()
with col2:
    if st.button("089-999-9999"):
        phone_number = "089-999-9999"
        st.rerun()
with col3:
    if st.button("066-123-4567"):
        phone_number = "066-123-4567"
        st.rerun()

# Enhanced function to analyze Thai phone number and return structured response
def analyze_phone_number(number):
    # Clean the phone number
    cleaned = re.sub(r'[^\d]', '', number)
    
    prompt = f"""
    วิเคราะห์เบอร์โทรศัพท์ไทย: {number}
    
    กรุณาวิเคราะห์และประเมินราคาจริงตามตลาดไทย แล้วให้ข้อมูลในรูปแบบ JSON ดังนี้:
    {{
        "pattern_analysis": {{
            "repeated_digits": "อธิบายเลขซ้ำที่พบ",
            "sequential": "อธิบายเลขเรียงที่พบ", 
            "lucky_numbers": "เลขมงคลที่พบในเบอร์นี้"
        }},
        "cultural_meaning": {{
            "luck_score": [คะแนนมงคล 0-100],
            "meaning": "ความหมายทางวัฒนธรรมไทยของเบอร์นี้"
        }},
        "memorability_score": [คะแนนความจำง่าย 1-10],
        "price_analysis": {{
            "min_price": [ราคาต่ำสุดที่คาดการณ์],
            "max_price": [ราคาสูงสุดที่คาดการณ์],
            "average_price": [ราคาเฉลี่ยที่คาดการณ์],
            "reasoning": "เหตุผลการตั้งราคาอิงจากรูปแบบเลข ความมงคล และตลาดจริง"
        }},
        "overall_analysis": "การวิเคราะห์โดยรวมและคำแนะนำสำหรับเจ้าของเบอร์"
    }}
    
    โปรดประเมินราคาให้สมจริงตามตลาดเบอร์โทรศัพท์ไทย พิจารณาจาก:
    - รูปแบบของตัวเลข (เลขซ้ำ เลขเรียง เลขง่าย)
    - เลขมงคลตามความเชื่อไทย (8=มั่งคั่ง, 9=ยิ่งใหญ่, 6=เรียบร้อย)
    - ความง่ายในการจดจำ
    - ความหายากของรูปแบบ
    
    หลังจากนั้นให้วิเคราะห์เพิ่มเติมเป็นภาษาไทยอย่างละเอียด
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการวิเคราะห์เบอร์โทรศัพท์: {str(e)}"

# Function to extract data for visualization
def extract_price_data(analysis_text):
    try:
        # Try to find JSON in the response
        start = analysis_text.find('{')
        end = analysis_text.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = analysis_text[start:end]
            data = json.loads(json_str)
            return data
    except:
        pass
    
    # Fallback data if JSON parsing fails
    return {
        "cultural_meaning": {"luck_score": 50},
        "memorability_score": 5,
        "price_analysis": {
            "min_price": 5000,
            "max_price": 15000,
            "average_price": 10000
        }
    }

# Analyze button
if phone_number and api_key:
    if st.button("🔍 Analyze & Predict Price"):
        if len(re.sub(r'[^\d]', '', phone_number)) >= 9:  # Basic validation
            with st.spinner("Analyzing phone number..."):
                analysis = analyze_phone_number(phone_number)
                
                st.subheader("📊 Analysis Results")
                
                # Extract structured data for visualization
                data = extract_price_data(analysis)
                
                # Display key metrics
                col1, col2, col3 = st.columns(3)
                
                luck_score = data.get('cultural_meaning', {}).get('luck_score', 50)
                memory_score = data.get('memorability_score', 5)
                avg_price = data.get('price_analysis', {}).get('average_price', 10000)
                
                with col1:
                    st.metric("🍀 คะแนนมงคล", f"{luck_score}/100")
                with col2:
                    st.metric("🧠 ความจำง่าย", f"{memory_score}/10")
                with col3:
                    st.metric("💰 ราคาเฉลี่ย", f"฿{avg_price:,}")
                
                # Price range visualization
                st.subheader("💎 ช่วงราคาที่แนะนำ")
                
                price_data = data.get('price_analysis', {})
                min_price = price_data.get('min_price', 5000)
                max_price = price_data.get('max_price', 15000)
                
                # Create price range chart
                fig = go.Figure()
                
                categories = ['ราคาต่ำสุด', 'ราคาเฉลี่ย', 'ราคาสูงสุด']
                prices = [min_price, avg_price, max_price]
                colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
                
                fig.add_trace(go.Bar(
                    x=categories,
                    y=prices,
                    marker_color=colors,
                    text=[f'฿{price:,}' for price in prices],
                    textposition='auto',
                    textfont=dict(size=14, color='white')
                ))
                
                fig.update_layout(
                    title="ช่วงราคาที่คาดการณ์",
                    xaxis_title="ประเภทราคา",
                    yaxis_title="ราคา (บาท)",
                    showlegend=False,
                    height=400,
                    title_font_size=16
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Score visualization
                st.subheader("📈 คะแนนการประเมิน")
                
                # Create radar chart for scores
                categories_radar = ['คะแนนมงคล', 'ความจำง่าย', 'ความนิยม', 'ความหายาก']
                scores = [
                    luck_score/10,  # Scale to 0-10
                    memory_score,
                    min(luck_score/10 + 2, 10),  # Popularity score
                    max(10 - memory_score, 1)     # Rarity score
                ]
                
                fig_radar = go.Figure()
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=scores,
                    theta=categories_radar,
                    fill='toself',
                    name='คะแนนประเมิน',
                    marker_color='rgba(79, 172, 254, 0.6)'
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 10]
                        )),
                    showlegend=False,
                    title="แผนภูมิคะแนนการประเมิน",
                    height=400
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
                
                # Market comparison
                st.subheader("📊 เปรียบเทียบกับตลาด")
                
                # Sample market data for comparison
                market_data = pd.DataFrame({
                    'ประเภทเบอร์': ['เบอร์ธรรมดา', 'เบอร์ปานกลาง', 'เบอร์ของคุณ', 'เบอร์พรีเมี่ยม'],
                    'ราคาเฉลี่ย': [2000, 15000, avg_price, 80000],
                    'สี': ['#95a5a6', '#f39c12', '#e74c3c', '#9b59b6']
                })
                
                fig_comparison = px.bar(
                    market_data, 
                    x='ประเภทเบอร์', 
                    y='ราคาเฉลี่ย',
                    color='สี',
                    color_discrete_map={color: color for color in market_data['สี']},
                    title="เปรียบเทียบราคากับตลาด"
                )
                
                fig_comparison.update_layout(
                    showlegend=False,
                    height=400,
                    yaxis_title="ราคา (บาท)",
                    xaxis_title="ประเภทเบอร์โทรศัพท์"
                )
                
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Full analysis text
                st.subheader("📝 การวิเคราะห์ละเอียด")
                st.write(analysis)
                
                # Add some visual elements
                st.success("✅ Analysis completed!")
                
        else:
            st.error("Please enter a valid phone number with at least 9 digits.")

elif phone_number and not api_key:
    st.warning("Please enter your Gemini API key in the sidebar to get analysis.")

# Popular patterns section
st.markdown("---")
st.subheader("🔥 เบอร์ยอดนิยมและราคา")

# Create sample data for popular numbers
popular_data = pd.DataFrame({
    'เบอร์โทรศัพท์': ['081-888-8888', '089-999-9999', '081-111-1111', '066-168-168', '089-123-456'],
    'ราคาประมาณ': [150000, 120000, 50000, 35000, 15000],
    'ประเภท': ['เลขมงคล', 'เลขมงคล', 'เลขซ้ำ', 'เลขโชคลาภ', 'เลขเรียง']
})

fig_popular = px.bar(
    popular_data, 
    x='เบอร์โทรศัพท์', 
    y='ราคาประมาณ',
    color='ประเภท',
    title="ตัวอย่างเบอร์ยอดนิยมและราคา"
)

fig_popular.update_layout(
    height=400,
    xaxis_title="เบอร์โทรศัพท์",
    yaxis_title="ราคา (บาท)"
)

st.plotly_chart(fig_popular, use_container_width=True)

# Information section
st.markdown("---")
st.subheader("ℹ️ How it works")
st.write("""
- Enter a Thai phone number
- AI analyzes digit patterns, lucky numbers, and memorability
- Get price prediction based on Thai market preferences
- Considers cultural factors like lucky number 8 and 9
- View interactive charts and comparisons
""")

# Market insights
st.subheader("💡 ข้อมูลตลาดเบอร์โทรศัพท์")

insights_col1, insights_col2 = st.columns(2)

with insights_col1:
    st.info("""
    **เลขมงคลที่นิยม:**
    - เลข 8: ความมั่งคั่ง
    - เลข 9: ความยิ่งใหญ่  
    - เลข 6: ความเรียบร้อย
    - เลข 168: โชคลาภ
    """)

with insights_col2:
    st.info("""
    **ปัจจัยที่ส่งผลต่อราคา:**
    - รูปแบบเลขซ้ำ/เรียง
    - ความง่ายในการจดจำ
    - ความหมายทางวัฒนธรรม
    - ความหายากของรูปแบบ
    """)

# Footer
st.markdown("---")
st.caption("Nida")
