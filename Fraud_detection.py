import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
# ML Libraries
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from transformers import pipeline
from transformers import BertTokenizer
import re
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

from scipy.spatial import distance
from itertools import combinations
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
        df[col] = df[col].apply(lambda x: x if x >= 0 else np.nan)## on transforme les valeurs negatives (car non significatives) en Nan

df.isnull().sum()       
df = df[~((df['bank_months_count'].isna()) & (df['fraud_bool'] == 0))]

numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()
plt.figure(figsize=(16, 13))  # Adjust the size of the plot
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.show()

df.drop(columns=['prev_address_months_count','intended_balcon_amount'], inplace=True)## on supprimes les deux colonnes avec 75% de valeurs manquantes
##on supprime les lignes pour les observations a valeurs manquantes et qui appartiennent a la classe majoritaire pour équilibrer les données
df = df[~((df['session_length_in_minutes'].isna()) & (df['fraud_bool'] == 0))]
df = df[~((df['bank_months_count'].isna()) & (df['fraud_bool'] == 0))]
df = df[~((df['velocity_6h'].isna()) & (df['fraud_bool'] == 0))]
df= df[~((df['device_distinct_emails_8w'].isna()) & (df['fraud_bool'] == 0))]
count=((df['bank_months_count'].isna()) & (df['fraud_bool'] == 1)).sum()
print(count) ##toutes les valeurs nulles correspond a une fraude !!!!!!



### NAIVE BAYES ######################
dfb = df.copy()
## on transforme les colonnes binaires en type category
dfb['email_is_free'] = dfb['email_is_free'].astype('category')
dfb['phone_home_valid'] = dfb['phone_home_valid'].astype('category')
dfb['phone_mobile_valid'] = dfb['phone_mobile_valid'].astype('category')
dfb['has_other_cards'] = dfb['has_other_cards'].astype('category')
dfb['foreign_request'] = dfb['foreign_request'].astype('category')
dfb['keep_alive_session'] = dfb['keep_alive_session'].astype('category')
### on donne la valeur output a y
dfb.drop(columns=['device_fraud_count'], inplace=True)
y = dfb['fraud_bool']
## on separe les données en categoriale et numérique
ds=dfb.copy()
ds.drop(columns=['fraud_bool'],inplace=True)
categorical_columns = dfb.select_dtypes(include=['object', 'category']).columns
numerical_columns = ds.select_dtypes(include=['number']).columns

# Separation
X_categorical = dfb[categorical_columns]
X_numerical = dfb[numerical_columns]

# Encoding
encoder = OneHotEncoder(sparse_output=False)
X_categorical_encoded = encoder.fit_transform(X_categorical)

X_p = np.hstack((X_categorical_encoded, X_numerical))

X_p= np.nan_to_num(X_p, nan=-1)

X = np.array(X_p)  # Full feature set
y = np.array(y)

### augmentation des données
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X = np.array(X_resampled)  
y = np.array(y_resampled)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_index, test_index = next(sss.split(X, y))

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

X_categorical_train, X_categorical_test = X_train[:, :X_categorical_encoded.shape[1]], X_test[:, :X_categorical_encoded.shape[1]]
X_numerical_train, X_numerical_test = X_train[:, X_categorical_encoded.shape[1]:], X_test[:, X_categorical_encoded.shape[1]:]

# normalisation
scaler = MinMaxScaler()
X_numerical_train_scaled = scaler.fit_transform(X_numerical_train)
X_numerical_test_scaled = scaler.transform(X_numerical_test)

# hyperparametre tuning
param_grid_gnb = {'var_smoothing': [1e-9, 1e-7, 1e-5, 1e-3]}
param_grid_mnb = {'alpha': [0.1, 0.5, 1.0, 2.0], 'fit_prior': [True, False]}

# GaussianNB
gnb = GaussianNB()
grid_search_gnb = GridSearchCV(gnb, param_grid_gnb, cv=5, scoring='recall_macro')
grid_search_gnb.fit(X_numerical_train_scaled, y_train)
grid_result=grid_search_gnb.fit(X_categorical_train, y_train)
print(f"Best Parameters: {grid_result.best_params_}")
print(f"Best Score: {grid_result.best_score_:.2f}")
# MultinomialNB
mnb = MultinomialNB()
grid_search_mnb = GridSearchCV(mnb, param_grid_mnb, cv=5, scoring='recall_macro')
grid_search_mnb.fit(X_categorical_train, y_train)
grid_result=grid_search_mnb.fit(X_categorical_train, y_train)
print(f"Best Parameters: {grid_result.best_params_}")
print(f"Best Score: {grid_result.best_score_:.2f}")
best_gnb = grid_search_gnb.best_estimator_
best_mnb = grid_search_mnb.best_estimator_

gnb_log_probs = best_gnb.predict_log_proba(X_numerical_test_scaled)
mnb_log_probs = best_mnb.predict_log_proba(X_categorical_test)
final_log_probs = gnb_log_probs + mnb_log_probs
y_pred = np.argmax(final_log_probs, axis=1)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

### NAIVE BAYES ######################
### MLP ###############################
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from imblearn.over_sampling import SMOTE

dfm = df.copy()
###converstion du type en category
dfm['email_is_free'] = dfm['email_is_free'].astype('category')
dfm['phone_home_valid'] = dfm['phone_home_valid'].astype('category')
dfm['phone_mobile_valid'] = dfm['phone_mobile_valid'].astype('category')
dfm['has_other_cards'] = dfm['has_other_cards'].astype('category')
dfm['foreign_request'] = dfm['foreign_request'].astype('category')
dfm['keep_alive_session'] = dfm['keep_alive_session'].astype('category')

#toutes les valeurs sont 0
dfm.drop(columns=['device_fraud_count'], inplace=True)
###output
y = dfm['fraud_bool']


### input
ds = dfm.drop(columns=['fraud_bool'])

categorical_columns = ds.select_dtypes(include=['object', 'category']).columns
numerical_columns = ds.select_dtypes(include=['number']).columns
### encoding
encoder = OneHotEncoder(sparse_output=False)
X_categorical_encoded = encoder.fit_transform(dfm[categorical_columns])
X_p1 = np.hstack((X_categorical_encoded, ds[numerical_columns]))
X_p1 = np.nan_to_num(X_p1, nan=-1)
X = np.array(X_p1) 
y = np.array(y) 
### augmentation
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X = np.array(X_resampled)
y = np.array(y_resampled)

# Stratified train/test split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_index, test_index = next(sss.split(X, y))

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

# Normalisation
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
def create_model(activation='tanh', optimizer='adam', learning_rate=0.01, units=10):
    model = Sequential()
    model.add(Dense(units, activation=activation, input_dim=X_train_scaled.shape[1]))  # First hidden layer
    model.add(Dense(20, activation=activation))  # Second hidden layer (fixed for simplicity)
    model.add(Dense(1, activation='sigmoid'))  # Output layer (binary classification)
    optimizers = {'adam': Adam}
    if optimizer not in optimizers:#petit test
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    opt = optimizers[optimizer](learning_rate=learning_rate)
    
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model
model = KerasClassifier(build_fn=create_model, verbose=0)

#hyperparametre tuning
param_grid = {
    'activation': ['relu', 'tanh'],       # Activation functions
    'optimizer': ['adam'],                # Optimizer
    'learning_rate': [0.001, 0.01],       # Learning rates
    'batch_size': [32],                   # Batch sizes
    'epochs': [10, 20],                   # Epochs
    'units': [10, 20]                 # Number of neurons in the first hidden layer
}
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3,scoring='recall_macro')
grid_result = grid.fit(X_train_scaled, y_train)
print(f"Best Parameters: {grid_result.best_params_}")
print(f"Best Accuracy: {grid_result.best_score_}")

best_model = grid_result.best_estimator_
y_pred = (best_model.predict(X_test_scaled) > 0.5).astype(int)

print("\nFinal Test Set Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
###MLP#################################################
###K_MEANS#################################################


df.fillna(0, inplace=True)

colonnes_avec_nan = df.columns[df.isna().any()].tolist()

print("Colonnes contenant des valeurs NaN :", colonnes_avec_nan)

print(df.columns)

Y = df['fraud_bool']

X = df.drop(columns=['fraud_bool'], inplace=True)
df.head()

dfb = df
dfb['email_is_free'] = dfb['email_is_free'].astype('category')
dfb['phone_home_valid'] = dfb['phone_home_valid'].astype('category')
dfb['phone_mobile_valid'] = dfb['phone_mobile_valid'].astype('category')
dfb['has_other_cards'] = dfb['has_other_cards'].astype('category')
dfb['foreign_request'] = dfb['foreign_request'].astype('category')
dfb['keep_alive_session'] = dfb['keep_alive_session'].astype('category')

categorical_columns = dfb.select_dtypes(include=['object', 'category']).columns
numerical_columns = dfb.select_dtypes(include=['number']).columns
dfb.drop(columns=['device_fraud_count'], inplace=True)
numerical_columns = dfb.select_dtypes(include=['number']).columns

dfb_encoded = pd.get_dummies(dfb, categorical_columns)

dfb_encoded.head(10)

dfb_encoded.dtypes

scaler = StandardScaler()
dfb_scaled = scaler.fit_transform(dfb_encoded)

# Appliquer K-means avec un nombre K de clusters
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(dfb_scaled)


# Obtenir les centres des clusters et les étiquettes
centers = kmeans.cluster_centers_
labels = kmeans.labels_
inertie = kmeans.inertia_

print(centers)
print(labels)
print(inertie)

dfb_encoded['cluster'] = labels
# Ajouter les etiquettes au jeu de données
for i in range(5):  # Assurez-vous que la taille correspond
    print(dfb_encoded.iloc[i], dfb_scaled[i], Y[i], dfb_encoded['cluster'].iloc[i])

# Calculer le nombre d'éléments dans chaque cluster
unique_labels, counts = np.unique(labels, return_counts=True)

# Afficher le nombre d'éléments dans chaque cluster
for cluster, count in zip(unique_labels, counts):
    print(f"Cluster {cluster}: {count} éléments")

from sklearn.metrics import accuracy_score, recall_score

label_mapping = {}
for cluster in range(num_clusters):
    # Obtenez les indices des points dans le cluster actuel
    cluster_indices = np.where(labels == cluster)[0]

    # Obtenez les étiquettes réelles correspondantes à ces indices
    true_labels_in_cluster = Y.iloc[cluster_indices].values

    # Trouvez l'étiquette réelle la plus fréquente dans le cluster
    most_common_label = np.argmax(np.bincount(true_labels_in_cluster))

    # Associez le numéro du cluster à l'étiquette réelle la plus fréquente
    label_mapping[cluster] = most_common_label

# Appliquez la correspondance au tableau des étiquettes de cluster pour obtenir les étiquettes prédites
predicted_labels = np.vectorize(label_mapping.get)(labels)

# Calculer l'accuracy
accuracy = accuracy_score(Y, predicted_labels)

# Afficher le résultat
print(f'Accuracy : {accuracy}')

# Calculer le recall (rappel)
recall = recall_score(Y, predicted_labels, average='macro')  # 'macro' pour moyenner sur toutes les classes

# Afficher les résultats
print(f'Accuracy : {accuracy}')
print(f'Recall : {recall}')

for i in range(4):
    print (dfb_encoded.iloc[i], Y[i], predicted_labels[i])


Y = pd.DataFrame(Y, columns=['fraud_bool'])

r = pd.concat([dfb_encoded, Y], axis=0, ignore_index=True)

from sklearn.decomposition import PCA

# Utiliser l'analyse en composantes principales (PCA) pour visualiser les clusters dans un espace bidimensionnel
pca = PCA(n_components=2)
X_pca = pca.fit_transform(dfb_scaled)

#grille pour deux graphiques placés côte-à-côte
gs = gridspec.GridSpec(1, 2)

#Premier graphique des clusters originals
ax = plt.subplot(gs[0,0])
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=Y, cmap='viridis', edgecolors='k', s=20)
ax.set_title('Clusters Originals')
plt.xlabel('Première composante principale')
plt.ylabel('Deuxième composante principale')

#Second graphique des clusters identifiés par K-means
ax = plt.subplot(gs[0,1])
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', edgecolors='k', s=20)
ax.set_title('Clusters identifiés par K-Means')
plt.xlabel('Première composante principale')
plt.ylabel('Deuxième composante principale')

plt.show()

# Utiliser la méthode du coude pour trouver le nombre optimal de clusters (k)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(dfb_encoded)
    wcss.append(kmeans.inertia_)


# Tracer le coude (Elbow) pour déterminer k
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Méthode du coude pour trouver le nombre optimal de clusters')
plt.xlabel('Nombre de clusters')
plt.ylabel('WCSS (Somme des carrés intra-cluster)')
plt.show()

for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(dfb_scaled)
    silhouette_avg = silhouette_score(dfb_scaled, cluster_labels)
    print(f"Silhouette Score pour nombre de clusters {i} : {silhouette_avg}")

for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(dfb_scaled)
    db_score = davies_bouldin_score(dfb_scaled, cluster_labels)
    print(f"Coefficient de Davies-Bouldin pour nombre de clusters {i} : {db_score}")
###KMEANS################################
###SVM################################

ds = df.copy()
ds.fillna(0, inplace=True)

colonnes_avec_nan = ds.columns[ds.isna().any()].tolist()

print("Colonnes contenant des valeurs NaN :", colonnes_avec_nan)

ds = ds.apply(pd.to_numeric, errors='coerce')

X = ds.drop('fraud_bool', axis=1)  # Variables indépendantes
y = ds['fraud_bool']  # Cible

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X = X_resampled
y = y_resampled

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.model_selection import GridSearchCV

# Définir les paramètres à tester
param_grid = {
    'C': [1, 10],       # Valeurs possibles pour C
    'gamma': ['auto']  # Valeurs possibles pour gamma
}

classifierSVMrbf = SVC(kernel='rbf')

# Appliquer GridSearchCV pour trouver les meilleurs paramètres
grid_search = GridSearchCV(estimator=classifierSVMrbf, param_grid=param_grid, cv=2, n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train)

print("\nFinal Test Set Performance:")
y_pred = grid_search.best_estimator_.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Génération de la matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)

# Affichage de la matrice de confusion
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Non-Fraud', 'Fraud'])
disp.plot(cmap='Blues')
disp.ax_.set_title("Matrice de confusion SVM")
plt.show()



# algo KNN
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Chargement du dataset
file_path = r'Base.csv'
df = pd.read_csv(file_path)

# Suppression des colonnes catégorielles
categorical_columns = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
df = df.drop(columns=categorical_columns)

# Suppression des lignes avec des valeurs manquantes
df = df.dropna()

# Séparation des caractéristiques (X) et de la cible (y)
X = df.drop(columns=['fraud_bool'])
y = df['fraud_bool']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Suréchantillonnage avec SMOTE pour équilibrer les classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Normalisation des données
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Modèle KNN avec distance euclidienne
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train_resampled, y_train_resampled)

# Prédictions
y_pred = knn.predict(X_test)

# Évaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Calcul de la matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)

# Affichage graphique de la matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Fraud", "Fraud"], yticklabels=["Non-Fraud", "Fraud"])
plt.title("Matrice de Confusion pour KNN")
plt.xlabel("Prédictions")
plt.ylabel("Valeurs Réelles")
plt.show()
