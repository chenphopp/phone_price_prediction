# main.py - หน้าหลัก
import streamlit as st
import duckdb as db
import pandas as pd
from pymongo import MongoClient

# Page configuration
st.set_page_config(
    page_title="Phone Number Price Prediction",
    page_icon="📱",
    layout="wide"
)

# Main page content
st.title("📱 Phone Number Price Prediction & Search")
st.write("ค้นหาและประเมินราคาเบอร์โทรศัพท์ด้วย AI และ Machine Learning")

# สร้าง tabs
tab1, tab2, tab3 = st.tabs(["🔍 ค้นหาเบอร์", "🤖 AI Prediction", "🧠 ML Model"])

with tab1:
    st.write("# เบอร์ดีเบอร์ดัง มาแว้ววว! 👋")

    try:
        # เชื่อมต่อ MongoDB (localhost สำหรับ MongoDB Compass)
        client = MongoClient("mongodb+srv://TharathipK:TharathipK@tharathipk.xk7qsqc.mongodb.net/")
        duckdb1 = client["phone_db"]
        collection = duckdb1["phone_numbers"]

        # Query ข้อมูลจาก MongoDB กลับมาเป็น DataFrame
        cursor = collection.find()  # ดึงทั้งหมด
        df_mongo = pd.DataFrame(list(cursor))

        if df_mongo.empty:
            st.error("ไม่พบข้อมูลใน MongoDB")
            st.stop()

        # เปลี่ยนชื่อคอลัมน์ "_id" → "phone_number"
        df_mongo = df_mongo.rename(columns={"_id": "phone_number"}) 

        # อ่านไฟล์ CSV ด้วย pandas
        csv_df = pd.read_csv("109k_phone_number_price.csv")

        if csv_df.empty:
            st.error("ไม่พบข้อมูลใน CSV file")
            st.stop()

        # แสดงชื่อคอลัมน์เพื่อ debug
        # st.write("**CSV Columns:**", list(csv_df.columns))
        # st.write("**MongoDB Columns:**", list(df_mongo.columns))

        #  key ที่ใช้ join คือ 'phone_number'
        merged_df = pd.merge(csv_df, df_mongo, on='phone_number', how='inner')

        if merged_df.empty:
            st.warning("ไม่มีข้อมูลที่ตรงกันระหว่าง CSV และ MongoDB")
            st.stop()

        # ใช้ .sql แทน .query
        df = db.sql("""
            SELECT a.phone_number, a.price, b.description, b.provider, b.seller_id, b.seller_name
            FROM csv_df a
            JOIN df_mongo b
            ON a.phone_number = b.phone_number
        """).to_df()

        # สร้าง connection และ register table อย่างถูกต้อง
        con = db.connect()
        con.register('df_table', df)  # ใช้ชื่อ table ที่ชัดเจน

        st.success(f"โหลดข้อมูลสำเร็จ: {len(df):,} รายการ")

        # phone number input
        st.subheader("ค้นหาเบอร์")
        input_digits = []
        cols = st.columns(10)
        for i in range(10):
            digit = cols[i].text_input(f"{i+1}", max_chars=1, key=f"digit_{i}")
            input_digits.append(digit.strip() if digit else ".")

        # สร้าง regex pattern จาก input
        regex_pattern = "".join(input_digits)
        
        # Selectbox 
        col1, col2, col3 = st.columns(3)

        with col1:
            option1 = st.selectbox('ผลรวม', ['All'] + list(range(9, 82)))

        with col2:
            option2 = st.selectbox('ช่วงราคา', ['All', 'ไม่เกิน 1,000', 
                                                '1,001 - 3,000', 
                                                '3,001 - 5,000',
                                                '5,001 - 10,000', 
                                                '10,001 - 20,000',
                                                '20,001 - 40,000', 
                                                '40,001 - 100,000',
                                                'มากกว่า 100,000'
                                                ])

        with col3:
            option3 = st.selectbox('เครือข่าย', ['All', 
                                                'AIS',
                                                'DTAC',
                                                'TRUE MOVE',
                                                'i-Mobile',
                                                'My by CAT',
                                                'TOT 3G',
                                                'PENGUIN',
                                                'Myworld',
                                                'อื่นๆ'                                        
                                                ])

        # Search button
        if st.button("🔍 Search"):
            try:
                # dynamic SQL query
                conditions = []
                
                # ตรวจสอบชื่อคอลัมน์ที่แท้จริงในข้อมูล
                # available_columns = df.columns.tolist()
                # st.write("**Available columns:**", available_columns)
                
                # ปรับเงื่อนไขตามชื่อคอลัมน์ที่มีจริง
                if option1 != 'All':
                    # ใช้ชื่อคอลัมน์ที่มีจริงในข้อมูล
                    # เช่น 'sum_digits' หรือ 'phone_sum' แทน 'Phone Number No Dash'
                    conditions.append(f"price > 0")  # placeholder condition
                    
                if option2 != 'All':
                    if option2 == 'ไม่เกิน 1,000':
                        conditions.append("price <= 1000")
                    elif option2 == '1,001 - 3,000':
                        conditions.append("price BETWEEN 1001 AND 3000")
                    elif option2 == '3,001 - 5,000':
                        conditions.append("price BETWEEN 3001 AND 5000")
                    elif option2 == '5,001 - 10,000':
                        conditions.append("price BETWEEN 5001 AND 10000")
                    elif option2 == '10,001 - 20,000':
                        conditions.append("price BETWEEN 10001 AND 20000")
                    elif option2 == '20,001 - 40,000':
                        conditions.append("price BETWEEN 20001 AND 40000")
                    elif option2 == '40,001 - 100,000':
                        conditions.append("price BETWEEN 40001 AND 100000")
                    elif option2 == 'มากกว่า 100,000':
                        conditions.append("price > 100000")
                
                if option3 != 'All':
                    conditions.append(f"provider = '{option3}'")

                # เพิ่มเงื่อนไขการค้นหาเบอร์
                if any(d != "." for d in input_digits):
                    phone_like_pattern = "".join([d if d != "." else "_" for d in input_digits])
                    conditions.append(f"phone_number LIKE '{phone_like_pattern}'")

                # สร้าง SQL query
                sql_query = "SELECT * FROM df_table"
                if conditions:
                    where_clause = " AND ".join(conditions)
                    sql_query += f" WHERE {where_clause}"
                
                sql_query += " LIMIT 1000"  # จำกัดผลลัพธ์

                # st.write("**SQL Query:**", sql_query)  # debug
                
                results = con.execute(sql_query).df()
                countno = results.shape[0]
                
                st.subheader(f"พบทั้งหมด: {countno:,} เบอร์")
                
                if countno > 0:
                    st.dataframe(results, use_container_width=True)
                else:
                    st.warning("ไม่พบเบอร์ที่ตรงกับเงื่อนไข")
                    
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการค้นหา: {str(e)}")
                st.write("**Error details:**", str(e))

    except Exception as e:
        st.error(f"ไม่สามารถโหลดข้อมูลได้: {str(e)}")
        st.write("**Error details:**", str(e))

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

st.markdown("---")
st.caption("Nida")
