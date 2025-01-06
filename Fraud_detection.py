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
exclude_column = ['credit_risk_score','device_os','source','housing_status','employment_status','payment_type']
for col in df.columns:
    if col not in exclude_column:
        df[col] = df[col].apply(lambda x: x if x >= 0 else np.nan)

df.isnull().sum()       
df = df[~((df['bank_months_count'].isna()) & (df['fraud_bool'] == 0))]

numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()
plt.figure(figsize=(16, 13))  # Adjust the size of the plot
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.show()

df.drop(columns=['prev_address_months_count','intended_balcon_amount'], inplace=True)

df = df[~((df['session_length_in_minutes'].isna()) & (df['fraud_bool'] == 0))]
df = df[~((df['bank_months_count'].isna()) & (df['fraud_bool'] == 0))]
df = df[~((df['velocity_6h'].isna()) & (df['fraud_bool'] == 0))]
df= df[~((df['device_distinct_emails_8w'].isna()) & (df['fraud_bool'] == 0))]
count=((df['bank_months_count'].isna()) & (df['fraud_bool'] == 1)).sum()
print(count) ##toutes les valeurs nulles correspond a une fraude !!!!!!



### NAIVE BAYES ######################
dfb = pd.read_csv('/content/BBASE.csv')
dfb['email_is_free'] = dfb['email_is_free'].astype('category')
dfb['phone_home_valid'] = dfb['phone_home_valid'].astype('category')
dfb['phone_mobile_valid'] = dfb['phone_mobile_valid'].astype('category')
dfb['has_other_cards'] = dfb['has_other_cards'].astype('category')
dfb['foreign_request'] = dfb['foreign_request'].astype('category')
dfb['keep_alive_session'] = dfb['keep_alive_session'].astype('category')
dfb.drop(columns=['device_fraud_count'], inplace=True)
y = dfb['fraud_bool']
ds=dfb
ds.drop(columns=['fraud_bool'],inplace=True)
categorical_columns = dfb.select_dtypes(include=['object', 'category']).columns
numerical_columns = ds.select_dtypes(include=['number']).columns

# Separate the data
X_categorical = dfb[categorical_columns]
X_numerical = dfb[numerical_columns]

# Encode categorical data (One-hot encoding)
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
X_categorical_encoded = encoder.fit_transform(X_categorical)

# Combine encoded categorical and numerical data
X_processed = np.hstack((X_categorical_encoded, X_numerical))

X_processed = np.nan_to_num(X_processed, nan=-1)

from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Convert to numpy arrays (if not already)
X = np.array(X_processed)  # Full feature set
y = np.array(y)


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X = np.array(X_resampled)  # Full feature set
y = np.array(y_resampled)

# Stratified initial train-test split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_index, test_index = next(sss.split(X, y))

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

# Split categorical and numerical features
X_categorical_train, X_categorical_test = X_train[:, :X_categorical_encoded.shape[1]], X_test[:, :X_categorical_encoded.shape[1]]
X_numerical_train, X_numerical_test = X_train[:, X_categorical_encoded.shape[1]:], X_test[:, X_categorical_encoded.shape[1]:]

# Scale numerical features
scaler = MinMaxScaler()
X_numerical_train_scaled = scaler.fit_transform(X_numerical_train)
X_numerical_test_scaled = scaler.transform(X_numerical_test)

# Define hyperparameter grids
param_grid_gnb = {'var_smoothing': [1e-9, 1e-7, 1e-5, 1e-3]}
param_grid_mnb = {'alpha': [0.1, 0.5, 1.0, 2.0], 'fit_prior': [True, False]}

# Grid search for GaussianNB
gnb = GaussianNB()
grid_search_gnb = GridSearchCV(gnb, param_grid_gnb, cv=5, scoring='recall_macro')
grid_search_gnb.fit(X_numerical_train_scaled, y_train)
grid_result=grid_search_gnb.fit(X_categorical_train, y_train)
print(f"Best Parameters: {grid_result.best_params_}")
print(f"Best Score: {grid_result.best_score_:.2f}")
# Grid search for MultinomialNB
mnb = MultinomialNB()
grid_search_mnb = GridSearchCV(mnb, param_grid_mnb, cv=5, scoring='recall_macro')
grid_search_mnb.fit(X_categorical_train, y_train)
grid_result=grid_search_mnb.fit(X_categorical_train, y_train)
print(f"Best Parameters: {grid_result.best_params_}")
print(f"Best Score: {grid_result.best_score_:.2f}")
# Best estimators
best_gnb = grid_search_gnb.best_estimator_
best_mnb = grid_search_mnb.best_estimator_

# Predictions on test set
gnb_log_probs = best_gnb.predict_log_proba(X_numerical_test_scaled)
mnb_log_probs = best_mnb.predict_log_proba(X_categorical_test)

# Combine log probabilities
final_log_probs = gnb_log_probs + mnb_log_probs
y_pred = np.argmax(final_log_probs, axis=1)

# Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

### NAIVE BAYES ######################
### MLP ###############################
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from imblearn.over_sampling import SMOTE

# Load and preprocess the dataset
dfm = df
# Convert specific columns to categorical
dfm['email_is_free'] = dfm['email_is_free'].astype('category')
dfm['phone_home_valid'] = dfm['phone_home_valid'].astype('category')
dfm['phone_mobile_valid'] = dfm['phone_mobile_valid'].astype('category')
dfm['has_other_cards'] = dfm['has_other_cards'].astype('category')
dfm['foreign_request'] = dfm['foreign_request'].astype('category')
dfm['keep_alive_session'] = dfm['keep_alive_session'].astype('category')

# Drop unnecessary column
dfm.drop(columns=['device_fraud_count'], inplace=True)

# Separate target variable and features
y = dfm['fraud_bool']
ds = dfm.drop(columns=['fraud_bool'])

# Separate categorical and numerical columns
categorical_columns = ds.select_dtypes(include=['object', 'category']).columns
numerical_columns = ds.select_dtypes(include=['number']).columns

# Encode categorical data (One-hot encoding)
encoder = OneHotEncoder(sparse_output=False)
X_categorical_encoded = encoder.fit_transform(dfm[categorical_columns])

# Combine encoded categorical and numerical data
X_processed = np.hstack((X_categorical_encoded, ds[numerical_columns]))
X_processed = np.nan_to_num(X_processed, nan=-1)

# Convert to numpy arrays
X = np.array(X_processed)  # Full feature set
y = np.array(y)  # Labels

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X = np.array(X_resampled)
y = np.array(y_resampled)

# Stratified train/test split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_index, test_index = next(sss.split(X, y))

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

# Normalize numerical features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a function to create the Keras model
def create_model(activation='tanh', optimizer='adam', learning_rate=0.01):
    model = Sequential()
    model.add(Dense(10, activation=activation, input_dim=X_train_scaled.shape[1]))  # First hidden layer
    model.add(Dense(20, activation=activation))  # Second hidden layer
    model.add(Dense(1, activation='sigmoid'))  # Output layer (binary classification)
    
    # Instantiate the optimizer
    optimizers = {'adam': Adam}
    if optimizer not in optimizers:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    opt = optimizers[optimizer](learning_rate=learning_rate)
    
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Wrap the Keras model for GridSearchCV
model = KerasClassifier(build_fn=create_model, verbose=0)

# Define the hyperparameter grid
param_grid = {
    'activation': ['relu', 'tanh'],               # Activation functions
    'optimizer': ['adam'],                        # Optimizer
    'learning_rate': [0.001, 0.01],          # Learning rates
    'batch_size': [32],                       # Batch sizes
    'epochs': [10, 20]                            # Epochs
}

# Instantiate the GridSearchCV object
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)

# Perform grid search
grid_result = grid.fit(X_train_scaled, y_train)

# Print best parameters and score
print(f"Best Parameters: {grid_result.best_params_}")
print(f"Best Score: {grid_result.best_score_:.2f}")

# Evaluate on the test set with the best model
best_model = grid_result.best_estimator_
y_pred = (best_model.predict(X_test_scaled) > 0.5).astype(int)

print("\nFinal Test Set Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
###MLP#################################################



