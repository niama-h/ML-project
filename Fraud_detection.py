import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from transformers import pipeline
from transformers import BertTokenizer
import re
###########################################
file_path = r'Base.csv'
df = pd.read_csv(file_path)

###########################################
df.size
###########################################
df.shape
###########################################
df.describe()
###########################################
df[df['fraud_bool']==1]
###########################################

df = df[~((df['bank_months_count'].isna()) & (df['fraud_bool'] == 0))]

exclude_column = ['credit_risk_score','device_os','source','housing_status','employment_status','payment_type']
for col in df.columns:
    if col not in exclude_column:
        df[col] = df[col].apply(lambda x: x if x >= 0 else np.nan)

df.isnull().sum()       
df.drop(columns=['prev_address_months_count','intended_balcon_amount'], inplace=True)

numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()
plt.figure(figsize=(16, 13))  # Adjust the size of the plot
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.show()

df.drop(columns=['prev_address_months_count','intended_balcon_amount'], inplace=True)

