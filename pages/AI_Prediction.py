# pages/1_📱_Phone_Prediction.py
import streamlit as st
import google.generativeai as genai
import re
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
from datetime import datetime
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Phone Number Price Prediction with AI",
    page_icon="📱",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .price-highlight {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-size: 1.2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .analysis-box {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
        line-height: 1.6;
    }
    .summary-text {
        background: #f8f9fa;
        border: 1px solid #007bff;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-size: 1.1rem;
        line-height: 1.8;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# Page title with enhanced styling
st.markdown("# 📱 Phone Number Price Prediction with AI")

# Sidebar for configuration
with st.sidebar:
    # st.header("⚙️ การตั้งค่า")
    # api_key = st.text_input("Enter your Gemini API Key:", type="password")
    api_key = "AIzaSyD98blARCQvuKC2nV-wmzRq2LGfIf95W70"

    st.markdown("### 🔥 เทรนด์เบอร์ขายดี")
    st.markdown("- เลข 8 ติดกัน 3 ตัว")
    st.markdown("- เลข 168 (โชคลาภ)")
    st.markdown("- เลข 888 (มั่งคั่ง)")
    st.markdown("- เลข 999 (ยิ่งใหญ่)")

if api_key:
    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')

phone_number = st.text_input(
    "🔢 Fill phone number for analysis", 
    placeholder="081-234-5678",
    help="Supports all networks"
)

# Example numbers with enhanced display
st.markdown("### 💎 Popular Number")
col1, col2, col3, col4 = st.columns(4)

example_numbers = [
    ("081-888-8888", "เบอร์ซุปเปอร์มงคล", "🔥"),
    ("089-999-9999", "เบอร์ยิ่งใหญ่", "⭐"),
    ("066-168-168", "เบอร์โชคลาภ", "💰"),
    ("081-234-5678", "เบอร์เรียงลำดับ", "📈")
]

for i, (num, desc, icon) in enumerate(example_numbers):
    with [col1, col2, col3, col4][i]:
        if st.button(f"{icon} {num}", key=f"example_{i}"):
            phone_number = num
            st.rerun()
        st.caption(desc)

# Enhanced analysis function with better prompt for Thai summary
def analyze_phone_number_advanced(number):
    # Clean the phone number
    cleaned = re.sub(r'[^\d]', '', number)
    
    # Get current market trends (simulated)
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    prompt = f"""
    คุณเป็นผู้เชี่ยวชาญด้านการประเมินราคาเบอร์โทรศัพท์ในประเทศไทยที่มีประสบการณ์ 25 ปี 
    
    เบอร์ที่ต้องวิเคราะห์: {number} (ตัวเลข: {cleaned})
    วันที่วิเคราะห์: {current_date}
    
    กรุณาวิเคราะห์อย่างละเอียดและให้ข้อมูลในรูปแบบ JSON ก่อน แล้วตามด้วยสรุปภาษาไทยที่อ่านง่าย:
    
    {{
        "phone_info": {{
            "network": "เครือข่ายที่เป็นไปได้ (AIS/DTAC/TRUE/PENGUIN SIM/TOT/i-Mobile)",
            "region": "ภูมิภาคที่น่าจะเป็น",
            "format_type": "รูปแบบเบอร์ (มือถือ/บ้าน/สำนักงาน)"
        }},
        "pattern_analysis": {{
            "repeated_digits": "วิเคราะห์เลขซ้ำ เลขคู่ เลขเดี่ยว และอื่นๆ",
            "sequential_patterns": "วิเคราะห์เลขเรียง ขึ้น-ลง และ อื่นๆ",
            "special_combinations": "รูปแบบพิเศษที่พบ (เช่น ABAB, ABBA) และอื่นๆ",
            "digit_frequency": "ความถี่ของแต่ละเลข 0-9"
        }},
        "cultural_significance": {{
            "lucky_numbers": "เลขมงคลที่พบและความหมาย",
            "unlucky_numbers": "เลขที่ไม่มงคล (ถ้ามี)",
            "chinese_numerology": "ความหมายตามหลักโหราศาสตร์จีน",
            "thai_beliefs": "ความเชื่อไทยที่เกี่ยวข้อง",
            "feng_shui_score": [คะแนนฮวงซุ้ย 0-100]
        }},
        "memorability_analysis": {{
            "ease_of_memory": [คะแนนความจำง่าย 1-10],
            "pronunciation_ease": [คะแนนออกเสียงง่าย 1-10],
            "visual_pattern": "รูปแบบที่เห็นได้ชัด",
            "rhythm_flow": "จังหวะการออกเสียง"
        }},
        "market_analysis": {{
            "rarity_score": [คะแนนความหายาก 1-10],
            "demand_level": "ระดับความต้องการ (สูง/กลาง/ต่ำ)",
            "target_buyers": "กลุ่มเป้าหมายผู้ซื้อ",
            "seasonal_factors": "ปัจจัยตามฤดูกาล/เทศกาล"
        }},
        "price_prediction": {{
            "base_price": [ราคาพื้นฐาน],
            "min_price": [ราคาต่ำสุด],
            "max_price": [ราคาสูงสุด],
            "most_likely_price": [ราคาที่น่าจะได้จริง],
            "premium_factors": ["ปัจจัยที่เพิ่มค่า"],
            "discount_factors": ["ปัจจัยที่ลดค่า"],
            "market_comparison": "เปรียบเทียบกับเบอร์ใกล้เคียง",
            "price_trend": "แนวโน้มราคา 6 เดือนข้างหน้า"
        }},
        "investment_analysis": {{
            "investment_potential": [คะแนนการลงทุน 1-10],
            "liquidity": "ความสามารถในการขาย",
            "appreciation_forecast": "การเพิ่มค่าในอนาคต",
            "risk_factors": ["ปัจจัยเสี่ยง"]
        }},
        "detailed_breakdown": "การวิเคราะห์เชิงลึกทุกด้าน รวมถึงเหตุผลการตั้งราคา",
        "recommendations": {{
            "for_seller": "คำแนะนำสำหรับผู้ขาย",
            "for_buyer": "คำแนะนำสำหรับผู้ซื้อ",
            "best_selling_time": "ช่วงเวลาที่เหมาะกับการขาย",
            "marketing_strategy": "กลยุทธ์การตลาด"
        }},
        "confidence_level": [ระดับความมั่นใจในการทำนาย 0-100]
    }}
    
    === สรุปการวิเคราะห์ ===
    
    หลังจาก JSON ให้เขียนสรุปภาษาไทยที่อ่านง่าย โดยแบ่งเป็นหัวข้อต่างๆ ดังนี้:
    
    🔍 **การวิเคราะห์รูปแบบตัวเลข**
    [อธิบายรูปแบบของเบอร์นี้ เช่น มีเลขซ้ำ เลขเรียง หรือรูปแบบพิเศษอะไรบ้าง]
    
    🎋 **ความหมายทางวัฒนธรรมและฮวงซุ้ย**
    [อธิบายความหมายของตัวเลขตามความเชื่อไทยและจีน รวมถึงคะแนนฮวงซุ้ย]
    
    💰 **การประเมินราคา**
    [บอกราคาที่คาดการณ์พร้อมเหตุผล และปัจจัยที่ทำให้ราคาสูงหรือต่ำ]
    
    📈 **ศักยภาพการลงทุน**
    [วิเคราะห์ว่าเบอร์นี้เหมาะกับการลงทุนหรือไม่ และแนวโน้มในอนาคต]
    
    💡 **คำแนะนำ**
    [ให้คำแนะนำสำหรับผู้ที่สนใจซื้อหรือขาย]
    
    ข้อมูลอ้างอิง:
    - ราคาเบอร์ธรรมดา: 500-5,000 บาท
    - ราคาเบอร์ดี: 5,000-50,000 บาท  
    - ราคาเบอร์พรีเมี่ยม: 50,000-500,000 บาท
    - ราคาเบอร์ซุปเปอร์พรีเมี่ยม: 500,000+ บาท
    
    โปรดให้การประเมินที่สมจริงและสามารถนำไปใช้ได้จริงในตลาด
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการวิเคราะห์: {str(e)}"

# Enhanced data extraction function
def extract_comprehensive_data(analysis_text):
    try:
        # Try to find JSON in the response
        start = analysis_text.find('{')
        end = analysis_text.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = analysis_text[start:end]
            data = json.loads(json_str)
            return data
    except Exception as e:
        print(f"JSON parsing error: {e}")
    
    # Enhanced fallback data
    return {
        "phone_info": {"network": "AIS", "region": "กรุงเทพฯ"},
        "cultural_significance": {"feng_shui_score": 65},
        "memorability_analysis": {"ease_of_memory": 6},
        "market_analysis": {"rarity_score": 5, "demand_level": "กลาง"},
        "price_prediction": {
            "min_price": 8000,
            "max_price": 25000,
            "most_likely_price": 15000,
            "base_price": 12000
        },
        "investment_analysis": {"investment_potential": 6},
        "confidence_level": 75
    }

# Function to extract Thai summary from analysis
def extract_thai_summary(analysis_text):
    # Look for the summary section after "=== สรุปการวิเคราะห์ ==="
    summary_marker = "=== สรุปการวิเคราะห์ ==="
    if summary_marker in analysis_text:
        summary_part = analysis_text.split(summary_marker)[1].strip()
        return summary_part
    
    # If no marker found, try to extract text after JSON
    try:
        end_json = analysis_text.rfind('}')
        if end_json != -1:
            after_json = analysis_text[end_json + 1:].strip()
            if len(after_json) > 100:  # Only if there's substantial text
                return after_json
    except:
        pass
    
    # Fallback: return a portion of the text that looks like Thai analysis
    lines = analysis_text.split('\n')
    thai_lines = [line for line in lines if any(char in line for char in 'กขคงจฉชซญดตถทธนบปผฝพฟภมยรลวศษสหอฮ')]
    if len(thai_lines) > 5:
        return '\n'.join(thai_lines[-20:])  # Last 20 Thai lines
    
    return "ไม่สามารถแสดงสรุปได้ กรุณาลองใหม่อีกครั้ง"

# Main analysis section
if phone_number and api_key:
    if st.button("🚀 Analyze and Predict the Price", type="primary", use_container_width=True):
        cleaned_number = re.sub(r'[^\d]', '', phone_number)
        if len(cleaned_number) >= 9:
            
            # Progress bar for better UX
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("🔍 Analyzing..."):
                status_text.text("🔄 Number pattern processing...")
                progress_bar.progress(20)
                
                status_text.text("🎯 Culture analyzing...")
                progress_bar.progress(40)
                
                status_text.text("💰 Market price processing...")
                progress_bar.progress(60)
                
                analysis = analyze_phone_number_advanced(phone_number)
                progress_bar.progress(80)
                
                status_text.text("📊 Creating report...")
                progress_bar.progress(100)
                
                status_text.empty()
                progress_bar.empty()
                
                # Extract comprehensive data
                data = extract_comprehensive_data(analysis)
                
                # Extract Thai summary
                thai_summary = extract_thai_summary(analysis)
                
                # Display results with enhanced layout
                st.markdown("## 📊 Results")
                
                # Key metrics in enhanced format
                col1, col2, col3, col4 = st.columns(4)
                
                feng_shui = data.get('cultural_significance', {}).get('feng_shui_score', 65)
                memory_score = data.get('memorability_analysis', {}).get('ease_of_memory', 6)
                rarity = data.get('market_analysis', {}).get('rarity_score', 5)
                investment = data.get('investment_analysis', {}).get('investment_potential', 6)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3>🔮 {feng_shui}/100</h3>
                        <p>คะแนนฮวงซุ้ย</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3>🧠 {memory_score}/10</h3>
                        <p>ความจำง่าย</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col3:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3>💎 {rarity}/10</h3>
                        <p>ความหายาก</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col4:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3>📈 {investment}/10</h3>
                        <p>ศักยภาพการลงทุน</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Price prediction section
                st.markdown("## 💰 Price Prediction")
                
                price_data = data.get('price_prediction', {})
                min_price = price_data.get('min_price', 8000)
                max_price = price_data.get('max_price', 25000)
                likely_price = price_data.get('most_likely_price', 15000)
                base_price = price_data.get('base_price', 12000)
                
                # Highlight the most likely price
                st.markdown(f"""
                <div class="price-highlight">
                    <h2>🎯 ฿{likely_price:,}</h2>
                    <p>ช่วงราคา: ฿{min_price:,} - ฿{max_price:,}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Enhanced price visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # Price range chart
                    fig_price = go.Figure()
                    
                    categories = ['ราคาต่ำสุด', 'ราคาพื้นฐาน', 'ราคาที่คาดการณ์', 'ราคาสูงสุด']
                    prices = [min_price, base_price, likely_price, max_price]
                    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
                    
                    fig_price.add_trace(go.Bar(
                        x=categories,
                        y=prices,
                        marker_color=colors,
                        text=[f'฿{price:,}' for price in prices],
                        textposition='auto',
                        textfont=dict(size=12, color='white')
                    ))
                    
                    fig_price.update_layout(
                        title="ช่วงราคาที่ทำนาย",
                        xaxis_title="ประเภทราคา",
                        yaxis_title="ราคา (บาท)",
                        showlegend=False,
                        height=400
                    )
                    
                    st.plotly_chart(fig_price, use_container_width=True)
                
                with col2:
                    # Score comparison radar chart
                    categories_radar = ['ฮวงซุ้ย', 'ความจำง่าย', 'ความหายาก', 'การลงทุน', 'ความนิยม']
                    scores = [
                        feng_shui/10,
                        memory_score,
                        rarity,
                        investment,
                        min((feng_shui + memory_score*10)/20, 10)
                    ]
                    
                    fig_radar = go.Figure()
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=scores,
                        theta=categories_radar,
                        fill='toself',
                        name='คะแนนประเมิน',
                        marker_color='rgba(79, 172, 254, 0.6)',
                        line=dict(color='rgba(79, 172, 254, 0.8)', width=2)
                    ))
                    
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 10],
                                tickmode='linear',
                                tick0=0,
                                dtick=2
                            )),
                        showlegend=False,
                        title="แผนภูมิคะแนนรวม",
                        height=400
                    )
                    
                    st.plotly_chart(fig_radar, use_container_width=True)
                
                # Market position analysis
                st.markdown("## 📈 Market Price Analysis")
                
                # Enhanced market comparison
                market_segments = pd.DataFrame({
                    'ประเภทเบอร์': ['เบอร์ธรรมดา', 'เบอร์ดี', 'เบอร์ของคุณ', 'เบอร์พรีเมี่ยม', 'เบอร์ซุปเปอร์'],
                    'ราคาเฉลี่ย': [2500, 12000, likely_price, 75000, 300000],
                    'สี': ['#95a5a6', '#f39c12', '#e74c3c', '#9b59b6', '#2c3e50']
                })
                
                fig_market = px.bar(
                    market_segments,
                    x='ประเภทเบอร์',
                    y='ราคาเฉลี่ย',
                    color='ประเภทเบอร์',
                    title="ตำแหน่งของเบอร์คุณในตลาด"
                )
                
                fig_market.update_layout(
                    showlegend=False,
                    height=400,
                    yaxis_title="ราคา (บาท)",
                    xaxis_title="ประเภทเบอร์โทรศัพท์"
                )
                
                st.plotly_chart(fig_market, use_container_width=True)
                
                # Investment timeline
                st.markdown("## 📊 แนวโน้มการลงทุน")
                
                # Create investment projection
                months = ['ปัจจุบัน', '3 เดือน', '6 เดือน', '1 ปี', '2 ปี']
                projections = [
                    likely_price,
                    likely_price * 1.05,
                    likely_price * 1.12,
                    likely_price * 1.25,
                    likely_price * 1.45
                ]
                
                fig_timeline = go.Figure()
                fig_timeline.add_trace(go.Scatter(
                    x=months,
                    y=projections,
                    mode='lines+markers',
                    name='การเติบโตราคา',
                    line=dict(color='#17a2b8', width=3),
                    marker=dict(size=8, color='#17a2b8')
                ))
                
                fig_timeline.update_layout(
                    title="แนวโน้มราคาในอนาคต",
                    xaxis_title="ช่วงเวลา",
                    yaxis_title="ราคาคาดการณ์ (บาท)",
                    height=400
                )
                
                st.plotly_chart(fig_timeline, use_container_width=True)
                
                # Thai Summary Section (แทนที่ JSON)
                st.markdown("## 📝 สรุปการวิเคราะห์")
                
                st.markdown(f"""
                <div class="summary-text">
                {thai_summary}
                </div>
                """, unsafe_allow_html=True)
                
        else:
            st.error("⚠️ กรุณากรอกเบอร์โทรศัพท์ที่ถูกต้อง (อย่างน้อย 9 หลัก)")

elif phone_number and not api_key:
    st.warning("⚠️ กรุณากรอก API key ในแถบด้านข้าง")

# Footer with version info
st.markdown("---")
st.markdown("Nida")
