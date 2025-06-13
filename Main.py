# main.py - หน้าหลัก (Optimized Version - Fixed Serialization)
import streamlit as st
import duckdb as db
import pandas as pd
from pymongo import MongoClient
import time

# Page configuration
st.set_page_config(
    page_title="Phone Number Price Prediction",
    page_icon="📱",
    layout="wide"
)

# === PERFORMANCE OPTIMIZATIONS ===

# 1. Cache เฉพาะข้อมูล DataFrame (ไม่ cache connection)
@st.cache_data(ttl=3600, show_spinner="กำลังโหลดข้อมูลจาก MongoDB และ CSV...")
def load_phone_data():
    """โหลดและรวมข้อมูลจาก MongoDB และ CSV - Return เฉพาะ DataFrame"""
    try:
        # เชื่อมต่อ MongoDB
        client = MongoClient(
            "mongodb+srv://TharathipK:TharathipK@tharathipk.xk7qsqc.mongodb.net/",
            serverSelectionTimeoutMS=5000
        )
        duckdb1 = client["phone_db"]
        collection = duckdb1["phone_numbers"]

        # Query ข้อมูลจาก MongoDB
        cursor = collection.find({}, {
            "_id": 1, 
            "description": 1, 
            "provider": 1, 
            "seller_id": 1, 
            "seller_name": 1
        })
        df_mongo = pd.DataFrame(list(cursor))
        
        if df_mongo.empty:
            return None, "ไม่พบข้อมูลใน MongoDB"

        df_mongo = df_mongo.rename(columns={"_id": "phone_number"})

        # อ่าน CSV
        csv_df = pd.read_csv("109k_phone_number_price.csv")
        
        if csv_df.empty:
            return None, "ไม่พบข้อมูลใน CSV"

        # Join ข้อมูล
        merged_df = pd.merge(csv_df, df_mongo, on='phone_number', how='inner')
        
        # เพิ่มคอลัมน์ผลรวมตัวเลข
        def calculate_digit_sum(phone_number):
            """คำนวณผลรวมของตัวเลขในเบอร์โทร"""
            return sum(int(digit) for digit in str(phone_number) if digit.isdigit())
        
        merged_df['digit_sum'] = merged_df['phone_number'].apply(calculate_digit_sum)
        
        # ปิด MongoDB connection
        client.close()
        
        return merged_df, None
        
    except Exception as e:
        return None, f"Error loading data: {str(e)}"

# 2. Cache connection แยกต่างหาก
@st.cache_resource(ttl=3600)
def get_duckdb_connection():
    """สร้าง DuckDB connection แยกต่างหาก"""
    return db.connect()

# 3. ฟังก์ชันค้นหาที่ไม่ cache connection
def search_phone_numbers(df, search_pattern, price_range, provider, digit_pattern):
    """ฟังก์ชันค้นหาโดยใช้ DataFrame ที่โหลดแล้ว"""
    try:
        if df is None or df.empty:
            return None, "ไม่มีข้อมูลให้ค้นหา"
            
        # สร้าง connection ใหม่สำหรับการค้นหา (ไม่ cache)
        con = get_duckdb_connection()
        con.register('phone_data', df)
        
        conditions = []
        
        # Filter ตามผลรวมตัวเลข
        if search_pattern != 'All':
            conditions.append(f"digit_sum = {search_pattern}")
        
        # Filter ตามราคา
        if price_range != 'All':
            if price_range == 'ไม่เกิน 1,000':
                conditions.append("price <= 1000")
            elif price_range == '1,001 - 3,000':
                conditions.append("price BETWEEN 1001 AND 3000")
            elif price_range == '3,001 - 5,000':
                conditions.append("price BETWEEN 3001 AND 5000")
            elif price_range == '5,001 - 10,000':
                conditions.append("price BETWEEN 5001 AND 10000")
            elif price_range == '10,001 - 20,000':
                conditions.append("price BETWEEN 10001 AND 20000")
            elif price_range == '20,001 - 40,000':
                conditions.append("price BETWEEN 20001 AND 40000")
            elif price_range == '40,001 - 100,000':
                conditions.append("price BETWEEN 40001 AND 100000")
            elif price_range == 'มากกว่า 100,000':
                conditions.append("price > 100000")
        
        # Filter ตามผู้ให้บริการ
        if provider != 'All':
            conditions.append(f"provider = '{provider}'")
        
        # Filter ตามรูปแบบเบอร์
        if digit_pattern and digit_pattern.strip():
            # แปลง pattern เป็น SQL LIKE - แปลง _ เป็น % สำหรับ wildcard หลายตัว
            phone_like = digit_pattern.strip()
            # แทนที่ _ ด้วย % สำหรับ SQL LIKE wildcard
            phone_like = phone_like.replace('_', '%')
            # ถ้าลงท้ายด้วย % แล้ว ไม่ต้องเพิ่ม
            if not phone_like.endswith('%'):
                phone_like += '%'
            conditions.append(f"phone_number LIKE '{phone_like}'")
        
        # สร้าง SQL query
        sql_query = "SELECT * FROM phone_data"
        if conditions:
            sql_query += " WHERE " + " AND ".join(conditions)
        # ql_query += " ORDER BY price DESC LIMIT 500"
        sql_query += " ORDER BY price DESC"
        
        results = con.execute(sql_query).df()
        return results, None
        
    except Exception as e:
        return None, f"Search error: {str(e)}"

# === MAIN APP ===

st.title("📱 Phone Number Price Prediction")
st.write("ค้นหาและประเมินราคาเบอร์โทรศัพท์ด้วย AI และ Machine Learning")

# สร้าง tabs
tab1, tab2, tab3 = st.tabs(["🔍 ค้นหาเบอร์", "🤖 AI Prediction", "🧠 ML Model"])

with tab1:
    st.write("# เบอร์ดีเบอร์ดัง มาแว้ววว! 👋")

    # โหลดข้อมูลครั้งเดียว
    df, error_msg = load_phone_data()
    
    if error_msg:
        st.error(f"ไม่สามารถโหลดข้อมูลได้: {error_msg}")
        st.stop()
    
    if df is not None and not df.empty:
        # แสดงสถิติข้อมูลแบบง่าย
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("จำนวนเบอร์ทั้งหมด", f"{len(df):,}")
        with col_stat2:
            st.metric("ผู้ให้บริการ", f"{df['provider'].nunique()} เครือข่าย")

        st.success(f"✅ โหลดข้อมูลสำเร็จ - พร้อมค้นหา!")

    # === SEARCH INTERFACE ===
    st.subheader("🔍 ค้นหาเบอร์")
    
    # Phone number input
    st.write("**กรอกเบอร์ที่ต้องการค้นหา:**")
    phone_input = st.text_input(
        "เบอร์โทรศัพท์ (ใช้ _ แทนตัวเลขที่ไม่ระบุ)", 
        max_chars=10,
        placeholder="เช่น 091_______ หรือ 081234567_",
        help="กรอก 10 หลัก เช่น 0812345678 หรือ 091_______ สำหรับเบอร์ AIS ที่ขึ้นต้น 091"
    )

    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        option1 = st.selectbox('ผลรวมตัวเลข', ['All'] + list(range(9, 82)))
    
    with col2:
        option2 = st.selectbox('ช่วงราคา', [
            'All', 'ไม่เกิน 1,000', '1,001 - 3,000', '3,001 - 5,000',
            '5,001 - 10,000', '10,001 - 20,000', '20,001 - 40,000', 
            '40,001 - 100,000', 'มากกว่า 100,000'
        ])
    
    with col3:
        option3 = st.selectbox('เครือข่าย', [
            'All', 'AIS', 'DTAC', 'TRUE MOVE', 'i-Mobile', 
            'My by CAT', 'TOT 3G', 'PENGUIN', 'Myworld', 'อื่นๆ'
        ])

    # Search controls
    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        search_clicked = st.button("🔍 ค้นหา", type="primary")
    with col_btn2:
        auto_search = st.checkbox("ค้นหาอัตโนมัติเมื่อเปลี่ยนตัวกรอง", value=False)
    
    # ทำการค้นหา
    should_search = search_clicked or (auto_search and (phone_input or option2 != 'All' or option3 != 'All'))
    
    if should_search and df is not None:
        with st.spinner("กำลังค้นหา..."):
            start_time = time.time()
            
            # แปลง phone_input เป็น pattern
            digit_pattern = phone_input.strip() if phone_input else ""
            
            results, search_error = search_phone_numbers(
                df, option1, option2, option3, digit_pattern
            )
            
            search_time = time.time() - start_time
        
        if search_error:
            st.error(f"เกิดข้อผิดพลาด: {search_error}")
        elif results is not None and len(results) > 0:
            # แสดงผลลัพธ์
            st.success(f"🎯 พบ {len(results):,} เบอร์ (ใช้เวลา {search_time:.2f} วินาที)")
            
            # สถิติผลลัพธ์แบบง่าย
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.metric("ราคาสูงสุด", f"{results['price'].max():,.0f} ฿") 
            with col_r2:
                st.metric("ราคาต่ำสุด", f"{results['price'].min():,.0f} ฿")
            
            # แสดงตาราง
            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "phone_number": "เบอร์โทร",
                    "price": st.column_config.NumberColumn("ราคา (฿)", format="%d"),
                    "provider": "เครือข่าย", 
                    "description": "รายละเอียด",
                    "seller_name": "ผู้ขาย"
                }
            )
            
            # Export button
            if len(results) > 0:
                csv = results.to_csv(index=False)
                st.download_button(
                    label="📥 ดาวน์โหลดผลลัพธ์ (CSV)",
                    data=csv,
                    file_name=f"phone_search_results_{int(time.time())}.csv",
                    mime="text/csv"
                )
            
        else:
            st.warning("ไม่พบเบอร์ที่ตรงกับเงื่อนไข ลองเปลี่ยนเงื่อนไขการค้นหา")

    # แสดงตัวอย่างการใช้งาน
    with st.expander("💡 วิธีการใช้งาน"):
        st.write("""
        **การกรอกเบอร์:**
        - `091_______` = เบอร์ AIS ที่ขึ้นต้นด้วย 091
        - `081234567_` = เบอร์ที่ขึ้นต้นด้วย 081234567 และลงท้ายด้วยอะไรก็ได้
        - `0812345678` = เบอร์ที่ระบุแน่นอน
        - `08________` = เบอร์ AIS ทุกเบอร์
        
        **ผลรวมตัวเลข:**
        - ผลรวมของตัวเลขทั้งหมดในเบอร์โทร
        - เช่น 0812345678 = 0+8+1+2+3+4+5+6+7+8 = 44
        
        **การกรองข้อมูล:**
        - เลือกช่วงราคาที่สนใจ
        - เลือกเครือข่ายที่ต้องการ
        - เปิด "ค้นหาอัตโนมัติ" เพื่อความสะดวก
        """)
        
        # แสดงตัวอย่างผลรวมตัวเลข
        if df is not None and not df.empty:
            st.write("**ตัวอย่างผลรวมตัวเลขที่มีในข้อมูล:**")
            sample_sums = sorted(df['digit_sum'].unique())[:10]
            st.write(f"ผลรวมที่พบ: {', '.join(map(str, sample_sums))}...")

with tab2:
    st.subheader("🤖 AI-Powered Prediction (Gemini)")
    st.write("วิเคราะห์และประเมินราคาเบอร์โทรศัพท์ด้วย Google Gemini AI")
    st.write("• วิเคราะห์รูปแบบตัวเลข")
    st.write("• พิจารณาความเชื่อทางวัฒนธรรมไทย")
    st.write("• ให้เหตุผลการประเมินราคาอย่างละเอียด")
    if st.button("🚀 เข้าสู่หน้า AI Prediction", key="ai_predict"):
        st.switch_page("pages/AI_Prediction.py")

with tab3:
    st.subheader("🧠 Machine Learning Model")
    st.write("ประเมินราคาด้วย Machine Learning Model")
    st.write("• ใช้ข้อมูลราคาจริงจากตลาด")
    st.write("🚧 **Under developing**")
    if st.button("⏳ ML Model (Coming Soon)", key="ml_predict", disabled=True):
        st.switch_page("pages/ML_Prediction.py")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ การจัดการข้อมูล")
    if st.button("🔄 รีเฟรชข้อมูล"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    
    st.caption("💡 กดรีเฟรชเมื่อข้อมูลไม่อัปเดต")

st.markdown("---")
st.caption("Nida | ⚡ Optimized for Speed | 🔄 Auto-cache system")
