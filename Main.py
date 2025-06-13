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

    # เชื่อมต่อ MongoDB (localhost สำหรับ MongoDB Compass)
    client = MongoClient("mongodb+srv://TharathipK:TharathipK@tharathipk.xk7qsqc.mongodb.net/")
    duckdb1 = client["phone_db"]
    collection = duckdb1["phone_numbers"]

    # Query ข้อมูลจาก MongoDB กลับมาเป็น DataFrame
    cursor = collection.find()  # ดึงทั้งหมด
    df_mongo = pd.DataFrame(list(cursor))

    # ลบคอลัมน์ _id หากคุณไม่ต้องการให้แสดง
    # df_mongo.drop(columns=["_id"], inplace=True)

    # เปลี่ยนชื่อคอลัมน์ "_id" → "phone_number"
    df_mongo = df_mongo.rename(columns={"_id": "phone_number"}) 

    # 1. อ่านไฟล์ CSV ด้วย pandas
    # csv_df = pd.read_csv("C:/Users/USER/OneDrive/Desktop/NIDA/DADS5001 - Data analytics and Data Science tools programming/Week12_streamlit2/109k_phone_number_price.csv")
    csv_df = pd.read_csv("109k_phone_number_price.csv")

    #  key ที่ใช้ join คือ 'phone_number'
    merged_df = pd.merge(csv_df, df_mongo, on='phone_number', how='inner')


    # สมมุติว่าคุณมี DataFrame ชื่อ csv_df และ mongo_df

    # ใช้ .sql แทน .query
    df = db.sql("""
        SELECT a.phone_number, a.price , b.description, b.provider, b.seller_id, b.seller_name
        FROM csv_df a
        JOIN df_mongo b
        ON a.phone_number = b.phone_number

    """).to_df()

    #df = pd.read_csv("n_109k_phone_numbers_xls.csv")
    con = db.connect()
    con.register('SELECT * FROM df', df)

    # phone number input
    st.subheader("ค้นหาเบอร์")
    input_digits = []
    cols = st.columns(10)
    for i in range(10):
        digit = cols[i].text_input(f"{i+1}", max_chars=1)
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
        # dynamic SQL query
        conditions = []
        if option1 != 'All':
            conditions.append(f"Phone Number No Dash = '{option1}'")
        if option2 != 'All':
            conditions.append(f"Class Range Price = '{option2}'")
        if option3 != 'All':
            conditions.append(f"provider = '{option3}'")

        where_clause = " AND ".join(conditions)
        sql_query = f"SELECT * FROM df"
        if where_clause:
            sql_query += f" WHERE {where_clause}"

        results = con.execute(sql_query).df()
        countno = results.shape[0]
        
        st.subheader(f"พบทั้งหมด: {countno} เบอร์")
        st.write(results)

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
