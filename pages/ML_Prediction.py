# pages/2_🧠_ML_Prediction.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
np.random.seed(4)
import pandas as pd
import itertools as it
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering, OPTICS
from sklearn.mixture import GaussianMixture
# from matplotlib.colors import ListedColormap
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
# import seaborn as sns
from scipy import stats
import re
import warnings
from scipy.stats import t
import copy
from math import e
import math
import plotly.graph_objects as go

import pickle
warnings.filterwarnings("ignore")
# import seaborn as sns
# Load the saved model

loaded_model = pickle.load(open('regression_models.pkl', 'rb'))
loaded_class_model = pickle.load(open('classification_models.pkl', 'rb'))

class Feature:
  def __init__(self, name):
    self.name = name
    self.train = True
    self.select_column_name = 'Phone Number No Dash'
    self.verbose = False
  def execute_df_features(self, df):
    self.df = df
    print(f"execute_df new features {self.name}, Train : {self.train}")
  def getter(self):
    pass

class DigitSplitter(Feature):
  def __init__(self, name):
    super().__init__(name)
  def execute_df_features(self, df):
    _phone_number_len = len(df[self.select_column_name].iloc[0])
    for number in range(1, _phone_number_len):
      df[f'Digit_{number+1}'] = df[self.select_column_name].str[number]
    return df

class Binary(Feature):
  def __init__(self, name):
    super().__init__(name)

class OneHot(Feature):
  def __init__(self, name):
    super().__init__(name)
  def execute_df_features(self, df):

    for col in df.columns:
      if col.startswith('Digit_'):
        # Perform one-hot encoding for the current column
        one_hot = pd.get_dummies(df[col], prefix=col, dtype='int')
        # Concatenate the one-hot encoded columns to the DataFrame
        df = pd.concat([df, one_hot], axis=1)
    return df

class NumberDigitCounter(Feature):
  def __init__(self, name):
    super().__init__(name)
  def execute_df_features(self, df):
    self.df = df
    digit_counts_df = self.df[self.select_column_name].apply(self.count_digits).apply(pd.Series)
        # Rename the columns to indicate digit counts
    digit_counts_df = digit_counts_df.rename(columns={digit: f'Digit_{digit}_Count' for digit in digit_counts_df.columns})

    # Concatenate the new DataFrame with your original DataFrame
    self.df = pd.concat([self.df, digit_counts_df], axis=1)
    return self.df
  def count_digits(self, phone_number):
    digit_counts = {}
    for n in range(10):
      digit_counts[str(n)] = 0
    for digit in str(phone_number):
      if digit.isdigit():
        if digit not in digit_counts:
          digit_counts[digit] = 0
        digit_counts[digit] += 1
    return digit_counts

class FixedDigit(Feature):
  def __init__(self, name):
    pass
  def execute_df_features(self, df):
    self.df = df

    # return super().execute_df_features(df)
    for i in range(100):
      self.df[f'Feature_{i:02}'] = 0  # Initialize feature columns with 0

    for index, row in self.df.iterrows():
      phone_number = str(row['Phone Number No Dash'])
      for digit in range(100):
        if str(f'{digit:02}') in phone_number:
          self.df.at[index, f'Feature_{digit:02}'] = 1
    return self.df


class FixedSpecialDigit(Feature):
  def __init__(self, name):
    super().__init__(name)
    self.special_number_list = ['000', '0000',1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
                                16861 ,168, 861,
                                239, 932,
                                545, 456, 654, 45654, 65456, 6565,
                                289, 982,
                                9789, 789, 987, 78987, 98789, 7777, 8888, 9999,
                                5555
                                ]
  def execute_df_features(self, df):
    for special_number in self.special_number_list:
      df[f'Feature_Contains_{special_number}'] = 0

    for index, row in df.iterrows():
      phone_number = str(row['Phone Number No Dash'])
      for special_number in self.special_number_list:
        if str(special_number) in phone_number:
          df.at[index, f'Feature_Contains_{special_number}'] = 1
    return df

class ConsecutiveDigitScore(Feature):
  def __init__(self, name, var):
      super().__init__(name)
      self.var = var
      self.var_name_score = 'score_' + str(self.name)
  def execute_df_features(self, df):
      df[self.var_name_score] = df['Phone Number No Dash'].apply(self.group_consecutive_digits)
      df[self.var_name_score] = df[self.var_name_score]
      return df
  def group_consecutive_digits(self, phone_number):
      try:
        # Use a regular expression to find consecutive repeating digits
        phone_number = str(phone_number[1:len(phone_number)])
      except:
        print(phone_number)
        return 0
      pattern = r"(\d)\1*"
      groups = re.findall(pattern, phone_number)

      # List to store tuples of (digit, count)
      results = {}
      # for digit in range(0, 10):
      #     results[str(digit)] = 0
      results[self.var_name_score] = 0
      # Iterate through matches and count consecutive repetitions
      for match in re.finditer(r"(\d)\1*", phone_number):
          digit = match.group(0)[0]  # The repeating digit
          formula = self.var ** (len(match.group(0))-1)
          results[self.var_name_score] += formula
          count = len(match.group(0))  # Count how many times it repeats
          # results[digit]+=formula

      return results[self.var_name_score]
  def group_consecutive_digits_patterns(self, phone_number, repeating_patterns=['789','987','78','87','97','79','98','89']):
      # Strip the first digit (ignoring country code or leading digit)

      phone_number = str(phone_number[1:len(phone_number)])

      # Initialize results dictionary to store scores
      results = {}
      results[self.var_name_score] = 0

      # If no repeating patterns provided, default to single digits only
      if repeating_patterns is None:
          repeating_patterns = []

      # Step 1: Handle custom repeating patterns (like '78', '87', '89')
      for pattern in repeating_patterns:
          # Create a regex to find repeating patterns
          regex_pattern = f"({pattern})\\1*"
          matches = re.finditer(regex_pattern, phone_number)

          # Calculate score for each match
          for match in matches:
              repeating_group = match.group(0)  # Matched group (e.g., '787878')
              count = len(repeating_group) // len(pattern)  # Count how many times the pattern repeats
              formula = self.var ** (count - 1)
              results[self.var_name_score] += formula

      # Step 2: Handle single digit repetitions (default behavior)
      for match in re.finditer(r"(\d)\1*", phone_number):
          digit_group = match.group(0)  # The repeating digit group (e.g., '777')
          count = len(digit_group)  # Count how many times the single digit repeats
          formula = self.var  ** (count - 1)
          results[self.var_name_score] += formula

      return results[self.var_name_score]

class KMeanClusteringFeature(Feature):
  def __init__(self, name, n_clusters = 3):
    super().__init__(name)
    self.n_clusters = n_clusters
  def execute_df_features(self, df):
    global X_REMOVED_COLUMNS, Y_REMOVED_COLUMNS
    data = df.copy().drop(columns = X_REMOVED_COLUMNS + Y_REMOVED_COLUMNS)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Step 2: Reduce dimensions with PCA for visualization (2D)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_scaled)

    clusters = KMeans(n_clusters=self.n_clusters, random_state=42).fit(data_scaled).fit_predict(data_scaled)

    pca_df = pd.DataFrame(data = [])
    pca_df['Cluster'] = clusters
    # Perform one-hot encoding on the 'Cluster' column.
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_clusters = encoder.fit_transform(pca_df[['Cluster']])

    # Create a DataFrame with the one-hot encoded cluster features.
    one_hot_df = pd.DataFrame(one_hot_clusters, columns=[f'Cluster_{i}' for i in range(one_hot_clusters.shape[1])])

    # Concatenate the one-hot encoded features with the original DataFrame.
    pca_df_with_one_hot = pd.concat([pca_df, one_hot_df], axis=1).drop(columns = ['Cluster'])

    df = pd.concat([df, pca_df_with_one_hot], axis=1)
    return df

class RatioFeature(Feature):
  def __init__(self, name):
    super().__init__(name)
  def execute_df_features(self, df):
    # Assuming you want to permute the columns of your DataFrame
    column_names = list([str(n) for n in range(0, 10)])

    # Generate all possible permutations of the column names
    permutations = list(it.permutations(column_names, 2))

    for p in permutations:
      v, k = p
      df[f'Digit_{v}_{k}_ratio'] = df[f'Digit_{v}_Count'] / df[f'Digit_{k}_Count'].apply(lambda x: max(x, 1))
      df[f'Digit_{v}_{k}_average'] = (df[f'Digit_{v}_Count'] + df[f'Digit_{k}_Count'].apply(lambda x: max(x, 1)))/2
      df[f'Digit_{v}_{k}_average_ratio'] = df[f'Digit_{v}_{k}_average'] / df[f'Digit_{v}_Count'].apply(lambda x: max(x, 1))
      df[f'Digit_{v}_{k}_average_ratio_multiple'] = df[f'Digit_{v}_{k}_average'] * df[f'Digit_{v}_Count'].apply(lambda x: max(x, 1))

      df[f'Digit_{v}_{k}_diff'] = df[f'Digit_{v}_Count'] - df[f'Digit_{k}_Count'].apply(lambda x: max(x, 1))
      df[f'Digit_{v}_{k}_diff_ratio'] = df[f'Digit_{v}_{k}_diff'] / df[f'Digit_{v}_{k}_average']
      df[f'Digit_{v}_{k}_diff_ratio_multiple'] = df[f'Digit_{v}_{k}_diff_ratio'] * df[f'Digit_{v}_{k}_average_ratio']

      df[f'Digit_{v}_{k}_sum_agg'] = df[f'Digit_{v}_{k}_ratio']+df[f'Digit_{v}_{k}_average']+df[f'Digit_{v}_{k}_diff']+df[f'Digit_{v}_{k}_diff_ratio']
      df[f'Digit_{v}_{k}_multiple_agg'] = df[f'Digit_{v}_{k}_ratio']*df[f'Digit_{v}_{k}_average']*df[f'Digit_{v}_{k}_diff']*df[f'Digit_{v}_{k}_diff_ratio']
    return df

class NormalizeTargetFeature(Feature):
  def __init__(self, name):
    super().__init__(name)
  def execute_df_features(self, df):
    df['price_normalize'] = np.emath.logn(e, df['price'])
    return df

# ตัวแยก digit 02-777-9999 => [2, 7, 7, 7, 7, 9, 9, 9, 9]
digit_splitter_feature = DigitSplitter('DigitSplitter')
digit_splitter_feature.train = False

# ตัวยก digit 3 => [0, 0, 0, 1, 0, 0, 0, 0, 0]
one_hot_feature = OneHot('OneHot')
# binary_feature = Binary('Binary')

# นับตัวเลขใน Phone numbers such as 02-777-9999=> [0, 0, 1, 0, 0, 0, 3, 0, 4]
number_digit_counter_feature = NumberDigitCounter('NumberDigitCounter')

# นับว่าตัวเลขมีคู่เลขนั้นไหม เช่น 02, 77, 99, 79 => True ถ้าไม่มี False
fixed_digit_feature = FixedDigit('FixedDigit')

# นับว่าตัวเลขมีคู่เลขพิเศษ เช่น 123, 789, 999 => True ถ้าไม่มี False
fixed_special_digit_feature = FixedSpecialDigit('FixedSpecialDigit')

# สูตรคำนวนพิเศษ  sum of (e ^ (x - 1)) นับตัวเลขต่อเนื่องกัน เช่น 0[2] - [777] - [9999] => e^(1 - 1) + e ^ (3 - 1) + e ^ (4 - 1) = score
consecutive_digit_score_E = ConsecutiveDigitScore('ConsecutiveDigitScore', e)

kmean_clustering_feature = KMeanClusteringFeature('KMeanClusteringFeature', n_clusters = 5)

# avg ratio mean ...
ratio_feature = RatioFeature('RatioFeature')

# price normalize price
normalize_target_feature = NormalizeTargetFeature('NormalizeTargetFeature')

features_list = [
    digit_splitter_feature,
                 one_hot_feature,
                 number_digit_counter_feature,
                 fixed_digit_feature,
                 fixed_special_digit_feature,
                 consecutive_digit_score_E,
                 ratio_feature,
                #  kmean_clustering_feature,
                 normalize_target_feature
                 ]
# features_list = [digit_splitter_feature, kmean_clustering_feature]
one_hot_feature.getter()
X_REMOVED_COLUMNS = ['Phone Number No Dash',
                     'Class Range Price',
                     'price',
                     'phone_number',
                     'description',
                     'provider',
                     'seller_id',
                     'seller_name'
                     ] + [f'Digit_{number}' for number in range(2, 11)]

class Sample:
  OVER = 0
  UNDER = 1

class PhoneNumbers:
  def __init__(self, df, column_name : str, bins : list, features : list[Feature]):
    self.df = df
    self.filtered_df = self.df.copy()
    self.column = column_name
    self.bins = bins
    self.features = features
    self.sample_type = Sample.UNDER
    self.initializer()
    self.filter_data()
    self.feature_extractor()

  def feature_extractor(self):
    for feature in self.features:
      self.filtered_df = feature.execute_df_features(self.filtered_df)

  def oversample_data(self, df, class_column):
      # Step 1: Get the size of the majority class
      class_counts = df[class_column].value_counts()
      majority_class_size = class_counts.max()

      # Step 2: Create an empty list to store oversampled data
      oversampled_data = []

      # Step 3: Oversample each class to match the size of the majority class
      for class_label, count in class_counts.items():
          class_data = df[df[class_column] == class_label]
          if count < majority_class_size:
              # If the class size is less than the majority class size, oversample it
              oversampled_class_data = class_data.sample(n=majority_class_size, replace=True, random_state=42)
          else:
              # If the class size is already the majority size, take the entire class
              oversampled_class_data = class_data
          oversampled_data.append(oversampled_class_data)

      # Step 4: Combine all oversampled class data
      balanced_df = pd.concat(oversampled_data)

      # Step 5: Shuffle the data (optional)
      balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

      return balanced_df

  def undersample_data(self, df, class_column):
      # Step 1: Get the size of the minority class
      class_counts = df[class_column].value_counts()
      minority_class_size = class_counts.min()

      # Step 2: Create an empty list to store undersampled data
      undersampled_data = []

      # Step 3: Undersample each class to the size of the minority class
      for class_label, count in class_counts.items():
          class_data = df[df[class_column] == class_label]
          # If the class size is greater than the minority class size, sample it
          if count > minority_class_size:
              undersampled_class_data = class_data.sample(n=minority_class_size, random_state=42)
          else:
              # If the class size is already less or equal to the minority size, take all
              undersampled_class_data = class_data
          undersampled_data.append(undersampled_class_data)

      # Step 4: Combine all undersampled class data
      balanced_df = pd.concat(undersampled_data)

      # Step 5: Shuffle the data (optional)
      balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

      return balanced_df

  def filter_data(self):
    self.filtered_df['Phone Number No Dash'] = self.filtered_df['phone_number'].apply(self.remove_dash)
    self.filtered_df['Class Range Price'] = self.filtered_df['price'].apply(self.classify_price)
    #remove other class
    self.filtered_df = self.filtered_df[self.filtered_df['Class Range Price'] != 'Other']
    self.total_count = len(self.filtered_df)
    # if self.sample_type == False:
    if self.sample_type == Sample.OVER:
      self.filtered_df = self.oversample_data(self.filtered_df, 'Class Range Price')
    elif self.sample_type == Sample.UNDER:
      self.filtered_df = self.undersample_data(self.filtered_df, 'Class Range Price')
    # Create a OneHotEncoder object
    enc = OneHotEncoder(handle_unknown='ignore')

    # Fit and transform the 'Class Range Price' column
    encoded_labels = enc.fit_transform(self.filtered_df[['Class Range Price']]).toarray()

    # Create a new DataFrame with the one-hot encoded columns
    encoded_df = pd.DataFrame(encoded_labels, columns=enc.get_feature_names_out(['Class Range Price']))

    # Concatenate the encoded DataFrame with the original DataFrame
    self.filtered_df = pd.concat([self.filtered_df, encoded_df], axis=1)
    #class counts
    self.class_counts = self.filtered_df['Class Range Price'].value_counts()
  def classify_price(self, price):
    for ranger in range(len(self.bins)-1):

      if self.bins[ranger] + 1 <= price <= self.bins[ranger+1]:
        # print(price)
        # print(self.label_bins_generater[ranger])
        return self.label_bins_generater[ranger]
    return 'Other'  # or any other default value you prefer

  def initializer(self):
    global X_REMOVED_COLUMNS, Y_REMOVED_COLUMNS
    self.X_REMOVED_COLUMNS = X_REMOVED_COLUMNS.copy()
    self.label_bins_generater = []
    for ranger in range(len(self.bins)-1):
      self.label_bins_generater.append(f'{self.bins[ranger] + 1} - {self.bins[ranger+1]}')
    print(self.label_bins_generater)
    self.Y_REMOVED_COLUMNS = [f'Class Range Price_{l}' for l in self.label_bins_generater]
    self.X_REMOVED_COLUMNS += self.Y_REMOVED_COLUMNS
    Y_REMOVED_COLUMNS = self.Y_REMOVED_COLUMNS

  def remove_dash(self, phone_number):
    if isinstance(phone_number, str):
      return phone_number.replace('-', '')
    return phone_number
  def train(self):
    pass
  def score(self):
    pass
  def save_csv_file(self):
    _phones.filtered_df.to_csv(FILE_SAMPLE_OUTPUT_NAME, encoding='utf-8-sig', index=False)

# Page configuration
st.set_page_config(
    page_title="Phone Number Price Prediction with ML",
    page_icon="🧠",
    layout="wide"  # เปลี่ยนเป็น wide layout
)

# Custom CSS สำหรับปรับแต่ง UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 20px 20px;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        border: 2px solid #667eea;
        margin: 1rem 0;
        color: #2c3e50;
    }
    
    .model-section {
        background: linear-gradient(135deg, #ffffff 0%, #f1f3f4 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border: 2px solid #28a745;
        box-shadow: 0 2px 8px rgba(40, 167, 69, 0.2);
        transition: transform 0.3s ease;
    }
    
    .model-section:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .input-section {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border: 1px solid #e9ecef;
        margin: 2rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Header section
st.markdown("""
<div class="main-header">
    <h1>🧠 Phone Number Price Prediction</h1>
    <p>ระบบทำนายราคาเบอร์โทรศัพท์ด้วย Machine Learning</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### 📱 กรอกเบอร์โทรศัพท์ของคุณ")

col_input, col_button = st.columns([3, 1])

with col_input:
    phone_number = st.text_input(
        "", 
        placeholder="เช่น 063-345-6789",
        help="กรอกเบอร์โทรศัพท์ในรูปแบบ 0XX-XXX-XXXX"
    )

with col_button:
    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
    analyze_button = st.button("🔍 วิเคราะห์ราคา", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

def get_transform_number(number, feature_selection_columns):
    df = pd.read_csv('n_109k_phone_numbers_xls.csv')
    df_feature_contains = pd.DataFrame(columns=df.columns)
    column_name = 'phone_number'
    df = pd.DataFrame([{'phone_number':number,'price':99000,'description':'', 'provider':'', 'seller_id':'', 'seller_name':'', 'Class Range Price_99901 - 100000000':0}])
    bins = [50, 99900,100000000]
    p_phones = PhoneNumbers(df = df, column_name = column_name, bins = bins, features = features_list)
    new_df = pd.concat([df_feature_contains, p_phones.filtered_df.iloc[:, p_phones.filtered_df.columns.get_loc(f'Digit_2_{number[1]}'):].drop(columns=['price_normalize'])])
    new_df.fillna(0, inplace=True)
    return new_df.iloc[:, new_df.columns.get_loc('Digit_2_6'):].drop(columns=['price_normalize'])[feature_selection_columns]

feature_selection_columns = ['Digit_2_6', 'Digit_2_8', 'Digit_2_9', 'Digit_3_2', 'Digit_3_6',
       'Digit_3_8', 'Digit_4_2', 'Digit_4_4', 'Digit_4_8', 'Digit_5_8',
       'Digit_6_9', 'Digit_9_5', 'Digit_9_6', 'Digit_10_5', 'Digit_10_9',
       'Feature_24', 'Feature_25', 'Feature_29', 'Feature_36', 'Feature_42',
       'Feature_47', 'Feature_49', 'Feature_52', 'Feature_56', 'Feature_57',
       'Feature_61', 'Feature_63', 'Feature_68', 'Feature_69', 'Feature_74',
       'Feature_76', 'Feature_78', 'Feature_84', 'Feature_85', 'Feature_86',
       'Feature_87', 'Feature_94', 'Feature_95', 'Feature_Contains_168',
       'Feature_Contains_456', 'Feature_Contains_289',
       'score_ConsecutiveDigitScore', 'Digit_1_8_average',
       'Digit_2_5_average_ratio', 'Digit_3_0_average_ratio',
       'Digit_4_2_average', 'Digit_4_3_average_ratio', 'Digit_4_5_average',
       'Digit_4_9_average', 'Digit_5_0_average_ratio', 'Digit_5_4_average',
       'Digit_8_9_average', 'Digit_9_5_average_ratio', 'Digit_9_6_average']

feature_class_selection_columns = ['Digit_2_6', 'Digit_2_9', 'Digit_3_8', 'Digit_5_8', 'Digit_5_9',
       'Digit_6_9', 'Digit_7_9', 'Digit_8_6', 'Digit_9_5', 'Digit_9_6',
       'Digit_10_5', 'Digit_10_6', 'Digit_10_9', 'Feature_24', 'Feature_25',
       'Feature_29', 'Feature_42', 'Feature_49', 'Feature_52', 'Feature_53',
       'Feature_56', 'Feature_61', 'Feature_63', 'Feature_65', 'Feature_68',
       'Feature_69', 'Feature_74', 'Feature_78', 'Feature_85', 'Feature_86',
       'Feature_87', 'Feature_94', 'Feature_Contains_456',
       'Feature_Contains_289', 'Feature_Contains_789',
       'score_ConsecutiveDigitScore', 'Digit_3_0_average_ratio',
       'Digit_4_0_average_ratio', 'Digit_4_5_average_ratio',
       'Digit_4_5_multiple_agg', 'Digit_4_6_average', 'Digit_4_9_average',
       'Digit_5_4_average', 'Digit_5_6_average', 'Digit_5_6_average_ratio',
       'Digit_6_8_average_ratio', 'Digit_7_9_diff_ratio_multiple',
       'Digit_8_9_average', 'Digit_8_9_average_ratio', 'Digit_9_5_average',
       'Digit_9_5_average_ratio', 'Digit_9_6_average',
       'Digit_9_6_average_ratio', 'Digit_9_8_average_ratio_multiple']

def denormalize_log(y_log):
    return np.exp(y_log)

def predict_with_interval_log(model, input_features, X_train, y_train, alpha=0.05):
    if isinstance(input_features, pd.Series):
        input_features = input_features.to_frame().T

    input_features = input_features.reindex(columns=X_train.columns, fill_value=0)

    y_pred_log = model.predict(input_features)[0]

    # residual จาก log-normalized y
    y_train_pred_log = model.predict(X_train)
    residuals = y_train - y_train_pred_log
    s_yx = np.sqrt(np.sum(residuals**2) / (len(X_train) - 1))

    t_val = t.ppf(1 - alpha / 2, df=len(X_train) - 1)
    interval_log = t_val * s_yx

    lower_log = y_pred_log - interval_log
    upper_log = y_pred_log + interval_log

    # แปลงกลับเป็นราคาจริง
    pred_price = denormalize_log(y_pred_log)
    lower_price = denormalize_log(lower_log)
    upper_price = denormalize_log(upper_log)

    return pred_price, lower_price, upper_price

# Analysis section with enhanced styling
if analyze_button and phone_number:
    # Input validation
    if not phone_number.replace('-', '').replace(' ', '').isdigit():
        st.error("❌ กรุณากรอกเบอร์โทรศัพท์ที่ถูกต้อง")
        st.stop()
    
    if len(phone_number.replace('-', '').replace(' ', '')) != 10:
        st.error("❌ เบอร์โทรศัพท์ต้องมี 10 หลัก")
        st.stop()
    
    # Show loading
    with st.spinner('🔄 กำลังวิเคราะห์ข้อมูล...'):
        # โค้ดการประมวลผลเดิม
        new_data_number = get_transform_number(phone_number.replace('-',''), feature_selection_columns)
        df = pd.read_csv('n_109k_phone_numbers_xls.csv')
        X = df.iloc[:, 20:].drop(columns=['price_normalize'])[feature_selection_columns]
        y = df.iloc[:, 20:]['price_normalize']
        
        line_datas = []
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Results section
    st.markdown("## 📊 Results")
    
    # Regression Models
    st.markdown("""
    <div class="prediction-card">
        <h3>🎯 Regression Models</h3>
    </div>
    """, unsafe_allow_html=True)
    
    reg_col1, reg_col2, reg_col3 = st.columns(3)
    
    with reg_col1:
        # st.markdown('<div class="model-section">', unsafe_allow_html=True)
        st.markdown("#### 📈 Linear Model")
        pred, low, high = predict_with_interval_log(loaded_model['linear'], new_data_number, X_train, y_train)
        st.metric("ราคาทำนาย", f"{pred:,.0f} บาท")
        st.caption(f"ช่วงราคา: {low:,.0f} - {high:,.0f} บาท")
        line_datas.append([low, high, pred])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with reg_col2:
        # st.markdown('<div class="model-section">', unsafe_allow_html=True)
        st.markdown("#### 🌲 Elastic Model")
        pred, low, high = predict_with_interval_log(loaded_model['elas'], new_data_number, X_train, y_train)
        st.metric("ราคาทำนาย", f"{pred:,.0f} บาท")
        st.caption(f"ช่วงราคา: {low:,.0f} - {high:,.0f} บาท")
        line_datas.append([low, high, pred])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with reg_col3:
        # st.markdown('<div class="model-section">', unsafe_allow_html=True)
        st.markdown("#### 🎯 K-Nearest Neighbor")
        pred, low, high = predict_with_interval_log(loaded_model['knn'], new_data_number, X_train, y_train)
        st.metric("ราคาทำนาย", f"{pred:,.0f} บาท")
        st.caption(f"ช่วงราคา: {low:,.0f} - {high:,.0f} บาท")
        line_datas.append([low, high, pred])
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualization
    st.markdown("### 📊 การเปรียบเทียบผลลัพธ์")
    
    data = line_datas
    fig = go.Figure()
    model_names = ['Linear', 'Elastic', 'KNN']
    colors = ['#667eea', '#f093fb', '#4facfe']
    
    for i, row in enumerate(data):
        min_val, max_val, value = row
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[i+1, i+1],
            line=dict(color=colors[i], width=6),
            name=f'{model_names[i]} Range',
            legendgroup=f'group{i}',
        ))
        fig.add_trace(go.Scatter(
            x=[value],
            y=[i+1],
            mode='markers',
            marker=dict(size=12, color=colors[i]),
            name=f'{model_names[i]} Prediction',
            legendgroup=f'group{i}',
            showlegend=False
        ))
    
    fig.update_layout(
        title='ช่วงราคาและราคาทำนายจากแต่ละโมเดล',
        xaxis_title='ราคา (บาท)',
        yaxis_title='โมเดล',
        yaxis=dict(
            tickmode='array',
            tickvals=[1, 2, 3],
            ticktext=model_names
        ),
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Classification Models
    st.markdown("""
    <div class="prediction-card">
        <h3>🏷️ Classification Models</h3>
    </div>
    """, unsafe_allow_html=True)
    
    range_bins = ['699 ~ 3,990', '3,995 ~ 9,999', '10,000 ~ 35,000', '35,000 ~ 195,000','มากกว่า 195,000']
    X = df.iloc[:, 20:].drop(columns=['price_normalize'])[feature_class_selection_columns]
    y = df.iloc[:, 20:]['price_normalize']
    new_data_number = get_transform_number(phone_number.replace('-',''), feature_class_selection_columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    class_col1, class_col2, class_col3 = st.columns(3)
    
    with class_col1:
        # st.markdown('<div class="model-section">', unsafe_allow_html=True)
        st.markdown("#### 🧠 Naive Bayes")
        prediction = range_bins[loaded_class_model['ny'].predict(new_data_number)[0]]
        st.success(f"ระดับราคา: {prediction} บาท")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with class_col2:
        # st.markdown('<div class="model-section">', unsafe_allow_html=True)
        st.markdown("#### 📊 Logistic Regression")
        prediction = range_bins[loaded_class_model['lr'].predict(new_data_number)[0]]
        st.success(f"ระดับราคา: {prediction} บาท")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with class_col3:
        # st.markdown('<div class="model-section">', unsafe_allow_html=True)
        st.markdown("#### 🎯 K-Nearest Neighbors")
        prediction = range_bins[loaded_class_model['knn'].predict(new_data_number)[0]]
        st.success(f"ระดับราคา: {prediction} บาท")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Success message
    st.balloons()
    st.success("✅ การวิเคราะห์เสร็จสมบูรณ์!")

elif analyze_button and not phone_number:
    st.warning("⚠️ กรุณากรอกเบอร์โทรศัพท์ก่อนทำการวิเคราะห์")

# Footer
st.markdown("---")
st.markdown("Nida")
