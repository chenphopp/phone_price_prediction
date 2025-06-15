# main.py - หน้าหลัก (Updated Search Tab)
import streamlit as st
import duckdb as db
import pandas as pd
from pymongo import MongoClient
import time
import re
import plotly.express as px
import math

# Page configuration
st.set_page_config(
    page_title="Phone Number Price Prediction",
    page_icon="📱",
    layout="wide"
)

# === PERFORMANCE OPTIMIZATIONS ===

# 1. Cache เฉพาะข้อมูล DataFrame (ไม่ cache connection)
# @st.cache_data(ttl=3600, show_spinner="กำลังโหลดข้อมูลจาก MongoDB และ CSV...")
@st.cache_data(ttl=3600, show_spinner="กำลังโหลดข้อมูลจาก MongoDB...")
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
            "price": 1,
            "description": 1, 
            "provider": 1, 
            "seller_id": 1, 
            "seller_name": 1
        })
        df_mongo = pd.DataFrame(list(cursor))
        
        if df_mongo.empty:
            return None, "ไม่พบข้อมูลใน MongoDB"

        df_mongo = df_mongo.rename(columns={"_id": "phone_number"})

        # # อ่าน CSV
        # csv_df = pd.read_csv("109k_phone_number_price.csv")
        
        # if csv_df.empty:
        #     return None, "ไม่พบข้อมูลใน CSV"

        # Join ข้อมูล
        # merged_df = pd.merge(csv_df, df_mongo, on='phone_number', how='inner')
        merged_df = df_mongo
        
        # เพิ่มคอลัมน์ผลรวมตัวเลข
        def calculate_digit_sum(phone_number):
            """คำนวณผลรวมของตัวเลขในเบอร์โทร"""
            return sum(int(digit) for digit in str(phone_number) if digit.isdigit())
        
        merged_df['digit_sum'] = merged_df['phone_number'].apply(calculate_digit_sum)
        
        # เพิ่มคอลัมน์ช่วงราคา
        def categorize_price(price):
            """ฟังก์ชันจัดกลุ่มราคา"""
            if price <= 1000:
                return 'ไม่เกิน 1,000'
            elif price <= 3000:
                return '1,001 - 3,000'
            elif price <= 5000:
                return '3,001 - 5,000'
            elif price <= 10000:
                return '5,001 - 10,000'
            elif price <= 20000:
                return '10,001 - 20,000'
            elif price <= 40000:
                return '20,001 - 40,000'
            elif price <= 100000:
                return '40,001 - 100,000'
            else:
                return 'มากกว่า 100,000'
        
        merged_df['price_range'] = merged_df['price'].apply(categorize_price)
        
        # เพิ่มคอลัมน์ sum_numbers จาก description
        def extract_numbers_after_sum(text):
            """ฟังก์ชันดึงเลขหลังคำว่า ผลรวม"""
            if pd.isna(text) or text is None:
                return None
            
            text = str(text)
            pattern = r'ผลรวม\s*(\d{1,2})'
            match = re.search(pattern, text)
            
            if match:
                return int(match.group(1))
            else:
                return None
        
        merged_df['sum_numbers'] = merged_df['description'].apply(extract_numbers_after_sum)
        
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

# 3. ฟังก์ชันค้นหาแบบใหม่ (จาก 7_Number.py)
# แก้ไขฟังก์ชัน search_phone_advanced ให้รองรับ format 083-383-2360

def search_phone_advanced(df, input_digits, sum_filter, price_range_filter, provider_filter, sort_by="price", sort_order="DESC", limit=math.inf):
    """ฟังก์ชันค้นหาขั้นสูง - ค้นหาตำแหน่งที่แน่นอนของตัวเลข (รองรับ format xxx-xxx-xxxx)"""
    try:
        if df is None or df.empty:
            return None, "ไม่มีข้อมูลให้ค้นหา"
            
        # สร้าง connection ใหม่สำหรับการค้นหา
        con = get_duckdb_connection()
        con.register('phone_data', df)
        
        conditions = []
        
        # 1. Phone number pattern - ค้นหาตำแหน่งที่แน่นอน
        # ลบเครื่องหมาย - ออกก่อนเช็คตำแหน่ง
        phone_conditions = []
        for i, digit in enumerate(input_digits):
            if digit and digit.strip():  # ถ้ามีการกรอกตัวเลข
                # ใช้ REPLACE เพื่อลบเครื่องหมาย - ก่อน แล้วใช้ SUBSTRING เช็คตำแหน่ง
                phone_conditions.append(f"SUBSTRING(REPLACE(CAST(phone_number AS VARCHAR), '-', ''), {i+1}, 1) = '{digit.strip()}'")
        
        # รวมเงื่อนไขตำแหน่งทั้งหมด
        if phone_conditions:
            combined_phone_condition = " AND ".join(phone_conditions)
            conditions.append(f"({combined_phone_condition})")
        
        # 2. Sum numbers filter
        if sum_filter != 'All':
            conditions.append(f"sum_numbers = {sum_filter}")
        
        # 3. Price range filter
        if price_range_filter != 'All':
            conditions.append(f"price_range = '{price_range_filter}'")
        
        # 4. Provider filter
        if provider_filter != 'All':
            conditions.append(f"provider = '{provider_filter}'")
        
        # สร้าง SQL Query
        base_query = "SELECT * FROM phone_data"
        
        if conditions:
            where_clause = " AND ".join(conditions)
            sql_query = f"{base_query} WHERE {where_clause}"
        else:
            sql_query = base_query
        
        # เพิ่ม ORDER BY และ LIMIT (ถ้าไม่ใช่ infinity)
        sql_query += f" ORDER BY {sort_by} {sort_order}"
        
        # เพิ่ม LIMIT เฉพาะเมื่อไม่ใช่ infinity
        if not math.isinf(limit):
            sql_query += f" LIMIT {int(limit)}"
        
        # Debug: แสดง SQL query (สามารถเอาออกได้ในโปรดักชัน)
        # print(f"SQL Query: {sql_query}")
        
        results = con.execute(sql_query).df()
        return results, None
        
    except Exception as e:
        return None, f"Search error: {str(e)}"


# ทางเลือกที่ 2: ใช้ Regular Expression (รองรับ format xxx-xxx-xxxx)
def search_phone_advanced(df, input_digits, sum_filter, price_range_filter, provider_filter, sort_by="price", sort_order="DESC", limit=math.inf):
    """ฟังก์ชันค้นหาขั้นสูงด้วย Regular Expression (รองรับ format xxx-xxx-xxxx)"""
    try:
        if df is None or df.empty:
            return None, "ไม่มีข้อมูลให้ค้นหา"
        
        # สร้าง connection ใหม่สำหรับการค้นหา
        con = get_duckdb_connection()
        con.register('phone_data', df)
        
        conditions = []
        
        # 1. Phone number pattern ด้วย Regular Expression
        # สร้าง regex pattern สำหรับ format xxx-xxx-xxxx
        if any(d and d.strip() for d in input_digits):
            # สร้าง pattern โดยคำนึงถึงตำแหน่งของ - (หลังตำแหน่งที่ 3 และ 6)
            regex_pattern = ""
            for i, digit in enumerate(input_digits):
                # เพิ่ม - ในตำแหน่งที่เหมาะสม
                if i == 3:
                    regex_pattern += "-"
                elif i == 6:
                    regex_pattern += "-"
                
                # เพิ่มตัวเลขหรือ wildcard
                if digit and digit.strip():
                    regex_pattern += digit.strip()
                else:
                    regex_pattern += "\\d"  # จับคู่กับตัวเลขใดก็ได้
            
            # ใช้ REGEXP_MATCHES สำหรับ DuckDB
            conditions.append(f"REGEXP_MATCHES(CAST(phone_number AS VARCHAR), '^{regex_pattern}$')")
        
        # 2. Sum numbers filter
        if sum_filter != 'All':
            conditions.append(f"sum_numbers = {sum_filter}")
        
        # 3. Price range filter
        if price_range_filter != 'All':
            conditions.append(f"price_range = '{price_range_filter}'")
        
        # 4. Provider filter
        if provider_filter != 'All':
            conditions.append(f"provider = '{provider_filter}'")
        
        # สร้าง SQL Query
        base_query = "SELECT * FROM phone_data"
        
        if conditions:
            where_clause = " AND ".join(conditions)
            sql_query = f"{base_query} WHERE {where_clause}"
        else:
            sql_query = base_query
        
        # เพิ่ม ORDER BY และ LIMIT
        sql_query += f" ORDER BY {sort_by} {sort_order}"
        
        if not math.isinf(limit):
            sql_query += f" LIMIT {int(limit)}"
        
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
    # st.write("# เบอร์ดีเบอร์ดัง มาแว้ววว! 👋")

    # โหลดข้อมูลครั้งเดียว
    df, error_msg = load_phone_data()
    
    if error_msg:
        st.error(f"ไม่สามารถโหลดข้อมูลได้: {error_msg}")
        st.stop()
    
    if df is not None and not df.empty:
        # แสดงสถิติข้อมูลแบบละเอียด
        # col_stat1, col_stat2, col_stat3 = st.columns(3)
        # with col_stat1:
        #     st.metric("จำนวนเบอร์ทั้งหมด", f"{len(df):,}")
        # with col_stat2:
        #     st.metric("เครือข่าย", f"{df['provider'].nunique()} เครือข่าย")
        # with col_stat3:
        #     max_sum = df['sum_numbers'].max() if not df['sum_numbers'].isna().all() else 0
        #     st.metric("ผลรวมสูงสุด", f"{max_sum}")

        st.success("✅ Data loaded - Ready to search!")
        
        # === แสดงกราฟจำนวนเบอร์ตามเครือข่าย ===
        # ใช้ session state เพื่อเก็บข้อมูลกราฟ
        if 'chart_data' not in st.session_state:
            st.session_state.chart_data = df
            st.session_state.chart_title = "จำนวนเบอร์ในแต่ละเครือข่าย (ทั้งหมด)"
        
        st.subheader("📊 จำนวนเบอร์ตามเครือข่าย")
        
        provider_stats = st.session_state.chart_data.groupby('provider').agg({
            'phone_number': 'count'
        }).reset_index()
        
        provider_stats.columns = ['provider', 'จำนวน']
        provider_stats = provider_stats.sort_values('จำนวน', ascending=False)
        
        # สร้าง Bar Chart ด้วย Plotly
        fig = px.bar(
            provider_stats, 
            x='provider', 
            y='จำนวน',
            title=st.session_state.chart_title,
            color='จำนวน',
            color_continuous_scale='viridis',
            text='จำนวน'
        )
        
        # ปรับแต่งกราฟ
        fig.update_layout(
            height=400,
            showlegend=False,
            title_font_size=16,
            title_x=0.5,  # จัดกลาง
            xaxis_title="เครือข่าย",
            yaxis_title="จำนวนเบอร์",
            plot_bgcolor='rgba(0,0,0,0)',  # พื้นหลังใส
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12)
        )
        
        # ปรับแต่ง bar
        fig.update_traces(
            texttemplate='%{text:,.0f}',
            textposition='outside',
            textfont_size=12
        )
        
        # แสดงกราฟ
        chart_container = st.empty()
        chart_container.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("💰 ราคาเฉลี่ยของเบอร์โทรศัพท์ตามผลรวม")

    top_n = st.slider("เลือกจำนวน Top ราคาเฉลี่ยของเบอร์โทรศัพท์ตามผลรวม", min_value=5, max_value=40, step=5, value=20)

    if 'sum_numbers' in df.columns and 'price' in df.columns:
        chart_df = df[['sum_numbers', 'price']].copy()

        # แปลง sum_numbers เป็น int ก่อน แล้วค่อยแปลงเป็น str
        chart_df['sum_numbers'] = pd.to_numeric(chart_df['sum_numbers'], errors='coerce').dropna().astype(int).astype(str)
        chart_df['price'] = pd.to_numeric(chart_df['price'], errors='coerce')
        chart_df.dropna(subset=['sum_numbers', 'price'], inplace=True)

        if not chart_df.empty:
            avg_df = chart_df.groupby('sum_numbers')['price'].mean().reset_index()
            avg_df['price'] = avg_df['price'].round(2)

            # ใช้ top_n ที่เลือกจาก slider
            avg_df = avg_df.sort_values(by='price', ascending=False).head(top_n)

            ordered_cats = avg_df['sum_numbers'].tolist()
            avg_df['sum_numbers'] = pd.Categorical(avg_df['sum_numbers'], categories=ordered_cats, ordered=True)

            fig = px.bar(
                avg_df,
                x='sum_numbers',
                y='price',
                title=f"Top {top_n} ราคาเฉลี่ยของเบอร์โทรศัพท์ตามผลรวม",
                color='price',
                color_continuous_scale='plasma',
                text='price'
            )

            fig.update_layout(
                height=500,
                title_x=0.5,
                xaxis_title="ผลรวมของเบอร์",
                yaxis_title="ราคาเฉลี่ย (บาท)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                xaxis=dict(
                    type='category',
                    categoryorder='array',
                    categoryarray=ordered_cats
                )
            )

            fig.update_traces(
                texttemplate='%{text:,.2f}',
                textposition='outside'
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("ไม่มีข้อมูลที่ใช้ได้หลังล้างค่า NaN")
    else:
        st.warning("ไม่พบคอลัมน์ 'sum_numbers' หรือ 'price'")

    # === SEARCH INTERFACE ===
    st.subheader("🔍 ค้นหาเบอร์")
    
    # Phone number input แบบ 10 กล่อง (ปรับปรุงตาม 7_Number.py)
    st.write("**กรอกหมายเลขโทรศัพท์:**")
    input_digits = []
    cols = st.columns(10)
    for i in range(10):
        with cols[i]:
            digit = st.text_input(
                f"ตำแหน่งที่ {i+1}", 
                max_chars=1, 
                key=f"digit_{i}",
                label_visibility="collapsed",
                placeholder=f"{i+1}"
            )
            input_digits.append(digit.strip() if digit else "")

    st.divider()

    # Filter options (ปรับปรุงตาม 7_Number.py)
    st.write("**เลือกเงื่อนไขการค้นหา:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # สร้าง unique values สำหรับ sum_numbers
        if df is not None:
            unique_sums = sorted([x for x in df['sum_numbers'].dropna().unique() if x is not None])
            option1 = st.selectbox(
                '📊 ผลรวม', 
                ['All'] + unique_sums,
                help="เลือกผลรวมของเบอร์ที่ต้องการ"
            )
        else:
            option1 = st.selectbox('📊 ผลรวม', ['All'])
    
    with col2:
        option2 = st.selectbox(
            '💰 ช่วงราคา', 
            ['All', 'ไม่เกิน 1,000', '1,001 - 3,000', '3,001 - 5,000',
             '5,001 - 10,000', '10,001 - 20,000', '20,001 - 40,000', 
             '40,001 - 100,000', 'มากกว่า 100,000'],
            help="เลือกช่วงราคาที่ต้องการ"
        )
    
    with col3:
        # สร้าง unique values สำหรับ provider
        if df is not None:
            unique_providers = ['All'] + sorted(df['provider'].dropna().unique().tolist())
            option3 = st.selectbox(
                '📡 เครือข่าย', 
                unique_providers,
                help="เลือกเครือข่ายที่ต้องการ"
            )
        else:
            option3 = st.selectbox('📡 เครือข่าย', ['All'])

    # Advanced search options
    with st.expander("🔧 ตัวเลือกขั้นสูง"):
        col_adv1, col_adv2 = st.columns(2)
        
        with col_adv1:
            sort_by = st.selectbox(
                "เรียงลำดับตาม",
                ["price", "sum_numbers", "phone_number"],
                format_func=lambda x: {"price": "ราคา", "sum_numbers": "ผลรวม", "phone_number": "หมายเลข"}[x]
            )
        
        with col_adv2:
            sort_order = st.selectbox("ลำดับ", ["DESC", "ASC"], format_func=lambda x: {"ASC": "น้อยไปมาก", "DESC": "มากไปน้อย"}[x])
        
        col_adv3, col_adv4 = st.columns(2)
        
        with col_adv3:
            limit_option = st.selectbox(
                "จำกัดผลลัพธ์",
                ["No Limit", "Custom"],
                help="เลือกการจำกัดผลลัพธ์"
            )
        
        with col_adv4:
            if limit_option == "Custom":
                limit_results = st.number_input("จำนวน", min_value=1, max_value=10000, value=1000, step=10)
            else:
                limit_results = float('inf')
                st.markdown("**ไม่จำกัด**")

    st.divider()

    # Search button
    search_col1, search_col2, search_col3 = st.columns([1, 2, 1])
    with search_col2:
        search_button = st.button("🔍 ค้นหา", use_container_width=True, type="primary")
    
    # ทำการค้นหา
    if search_button and df is not None:
        with st.spinner('กำลังค้นหา...'):
            start_time = time.time()
            
            results, search_error = search_phone_advanced(
                df, input_digits, option1, option2, option3, 
                sort_by, sort_order, limit_results
            )
            
            search_time = time.time() - start_time
        
        if search_error:
            st.error(f"❌ เกิดข้อผิดพลาดในการค้นหา: {search_error}")
        elif results is not None and len(results) > 0:
            countno = len(results)
            # แสดงผลลัพธ์
            st.success(f"✅ พบทั้งหมด: {countno:,} เบอร์ (ใช้เวลา {search_time:.2f} วินาที)")
            
            # อัปเดตกราฟด้วยผลลัพธ์ใหม่
            st.session_state.chart_data = results
            st.session_state.chart_title = f"จำนวนเบอร์ในแต่ละเครือข่าย (ผลลัพธ์การค้นหา: {countno:,} เบอร์)"
            
            # สร้างกราฟใหม่
            provider_stats_search = results.groupby('provider').agg({
                'phone_number': 'count'
            }).reset_index()
            
            provider_stats_search.columns = ['provider', 'จำนวน']
            provider_stats_search = provider_stats_search.sort_values('จำนวน', ascending=False)
            
            # สร้าง Bar Chart ใหม่
            fig_search = px.bar(
                provider_stats_search, 
                x='provider', 
                y='จำนวน',
                title=st.session_state.chart_title,
                color='จำนวน',
                color_continuous_scale='plasma',  # เปลี่ยนสีเพื่อแยกความแตกต่าง
                text='จำนวน'
            )
            
            # ปรับแต่งกราฟ
            fig_search.update_layout(
                height=400,
                showlegend=False,
                title_font_size=16,
                title_x=0.5,
                xaxis_title="เครือข่าย",
                yaxis_title="จำนวนเบอร์",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12)
            )
            
            fig_search.update_traces(
                texttemplate='%{text:,.0f}',
                textposition='outside',
                textfont_size=12
            )
            
            # อัปเดตกราฟ
            chart_container.plotly_chart(fig_search, use_container_width=True)
            
            # สถิติของผลลัพธ์
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            # with col_stat1:
            #     st.metric("ราคาเฉลี่ย", f"{results['price'].mean():,.0f} บาท")
            
            with col_stat1:
                st.metric("ราคาสูงสุด", f"{results['price'].max():,.0f} บาท")
            
            with col_stat2:
                st.metric("ราคาต่ำสุด", f"{results['price'].min():,.0f} บาท")
            
            with col_stat3:
                avg_sum = results['sum_numbers'].dropna().mean()
                # st.metric("ผลรวมเฉลี่ย", f"{avg_sum:.1f}" if not pd.isna(avg_sum) else "N/A")
            
            st.divider()
            
            # แสดงตาราง
            st.subheader("📋 ผลลัพธ์การค้นหา")
            
            # จัดรูปแบบการแสดงผล
            display_df = results.copy()
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "phone_number": st.column_config.TextColumn("หมายเลขโทรศัพท์", width="medium"),
                    "price": st.column_config.NumberColumn("ราคา (฿)", format="%d"),
                    "description": st.column_config.TextColumn("รายละเอียด", width="large"),
                    "provider": st.column_config.TextColumn("เครือข่าย", width="small"),
                    "price_range": st.column_config.TextColumn("ช่วงราคา", width="medium"),
                    "sum_numbers": st.column_config.NumberColumn("ผลรวม", width="small"),
                    "seller_name": st.column_config.TextColumn("ผู้ขาย", width="medium")
                }
            )
            
            # Export button
            csv = results.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 ดาวน์โหลดผลลัพธ์ (CSV)",
                data=csv,
                file_name=f'phone_search_results_{countno}_records.csv',
                mime='text/csv'
            )
            
        else:
            st.warning("⚠️ ไม่พบข้อมูลที่ตรงกับเงื่อนไขการค้นหา")
            st.info("💡 ลองปรับเงื่อนไขการค้นหาใหม่")
            
            # รีเซ็ตกราฟกลับเป็นข้อมูลทั้งหมด
            st.session_state.chart_data = df
            st.session_state.chart_title = "จำนวนเบอร์ในแต่ละเครือข่าย (ทั้งหมด)"
            st.rerun()

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
    st.write("• ใช้ข้อมูลราคาจริงจากตลาดในการ train model")
    st.write("• Predict ราคาด้วย Regression และ Classification Model")
    if st.button("⏳ เข้าสู่หน้า ML Prediction", key="ml_predict", disabled=False):
        st.switch_page("pages/ML_Prediction.py")

# === SIDEBAR - สวยงาม ===
with st.sidebar:
    st.markdown("## 📊 สถิติทั่วไป")
    
    if df is not None and not df.empty:
        # === ข้อมูลภาพรวม ===
        with st.container():
            total_phones = len(df)
            total_providers = df['provider'].nunique()
            avg_price = df['price'].mean()
            
            # แสดง metrics หลัก
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("📱 เบอร์ทั้งหมด", f"{total_phones:,}")
            with col_m2:
                st.metric("📡 เครือข่าย", f"{total_providers}")
            
            # st.metric("💰 ราคาเฉลี่ย", f"{avg_price:,.0f} ฿")
        
        st.markdown("---")
        
        # === สถิติเครือข่าย ===
        st.markdown("### 📡 เครือข่าย")
        
        provider_stats = df.groupby('provider').agg({
            'phone_number': 'count'
        }).reset_index()
        
        provider_stats.columns = ['provider', 'จำนวน']
        provider_stats = provider_stats.sort_values('จำนวน', ascending=False)
        
        # แสดงข้อมูลแบบ text
        for _, row in provider_stats.iterrows():
            provider = row['provider']
            count = int(row['จำนวน'])
            
            # แสดงข้อมูลเครือข่าย ในบรรทัดเดียว
            st.markdown(f"**{provider}** `{count:,}` เบอร์")
        
        st.markdown("---")
        
        # === ช่วงราคา ===
        st.markdown("### 💎 ช่วงราคา")
        
        price_ranges = df['price_range'].value_counts().sort_index()
        
        # แสดงเป็น bar chart แบบง่าย
        for price_range, count in price_ranges.items():
            percentage = count / len(df) * 100
            st.markdown(f"**{price_range}**")
            st.progress(percentage/100, text=f"{count:,} เบอร์ ({percentage:.1f}%)")
        
        st.markdown("---")
        
        # === ผลรวมตัวเลข ===
        if not df['sum_numbers'].isna().all():
            st.markdown("### 🔢 ผลรวมตัวเลข")
            
            sum_min = int(df['sum_numbers'].min())
            sum_max = int(df['sum_numbers'].max())
            sum_avg = df['sum_numbers'].mean()
            sum_mode = df['sum_numbers'].mode().iloc[0] if not df['sum_numbers'].mode().empty else sum_avg
            
            # แสดงเป็น metrics
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.metric("ต่ำสุด", sum_min)
                st.metric("เฉลี่ย", f"{sum_avg:.1f}")
            with col_s2:
                st.metric("สูงสุด", sum_max)
                st.metric("พบบ่อย", int(sum_mode))
            
            # แสดงช่วงที่นิยม
            popular_sums = df['sum_numbers'].value_counts().head(5)
            st.markdown("**🔥 ผลรวมยอดนิยม:**")
            for sum_val, count in popular_sums.items():
                st.write(f"• ผลรวม {int(sum_val)}: {count:,} เบอร์")
        
        st.markdown("---")
    
    # === การจัดการข้อมูล ===
    st.markdown("### ⚙️ Management")
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

st.markdown("---")
st.caption("Nida")
