# Import libraries
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyoff
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression

# --------------------- Data Loading ---------------------
# NOTE: Update path based on your local setup
df = pd.read_csv('cell2celltrain.csv')
customer_data = df.copy()

# --------------------- Visualization ---------------------
# Plot histogram of 'MonthsInService' to analyze distribution
pyoff.iplot([go.Histogram(x=customer_data['MonthsInService'])])

# --------------------- Elbow Method for KMeans ---------------------
# Find optimal number of clusters using SSE
sse = {}
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, max_iter=1000, random_state=42)
    kmeans.fit(customer_data[['CustomerID', 'MonthsInService']])
    sse[k] = kmeans.inertia_

# Plot SSE for each value of k
plt.figure(figsize=(8, 5))
sns.pointplot(x=list(sse.keys()), y=list(sse.values()), color='red', markers='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.grid(True)
plt.show()

# --------------------- KMeans Clustering ---------------------
kmeans = KMeans(n_clusters=3, random_state=42)
customer_data['MonthsInServiceCluster'] = kmeans.fit_predict(customer_data[['MonthsInService']])

# Function to reorder cluster labels based on a target variable
def order_cluster(cluster_field_name, target_field_name, df, ascending):
    cluster_avg = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    cluster_avg = cluster_avg.sort_values(by=target_field_name, ascending=ascending).reset_index(drop=True)
    cluster_avg['index'] = cluster_avg.index
    df = pd.merge(df, cluster_avg[[cluster_field_name, 'index']], on=cluster_field_name)
    df = df.drop(columns=[cluster_field_name])
    df = df.rename(columns={"index": cluster_field_name})
    return df

# Reorder clusters
customer_data = order_cluster('MonthsInServiceCluster', 'MonthsInService', customer_data, True)

# Select churn-prone cluster for modeling
df_churn_prone = customer_data[customer_data['MonthsInServiceCluster'] == 0]

# --------------------- Feature Selection ---------------------
# Remove irrelevant or high-cardinality columns
columns_to_drop = ['HandsetPrice', 'HandsetModels', 'Homeownership', 'MaritalStatus', 'UniqueSubs',
                   'PeakCallsInOut', 'OffPeakCallsInOut', 'DroppedBlockedCalls', 'RetentionCalls',
                   'InboundCalls', 'OverageMinutes', 'ReceivedCalls', 'OwnsMotorcycle', 'NonUSTravel',
                   'OwnsComputer', 'RVOwner', 'TruckOwner', 'HandsetRefurbished', 'HandsetWebCapable',
                   'Handsets']

dataset = df_churn_prone.drop(columns=columns_to_drop)

# --------------------- Encoding Categorical Features ---------------------
# Separate binary and multi-value categorical columns
binary_cols = []
multi_value_cols = []

for col in dataset.columns:
    if dataset[col].dtype == 'object':
        if dataset[col].nunique() == 2:
            binary_cols.append(col)
        else:
            multi_value_cols.append(col)

# Encode multi-value columns using LabelEncoder
le = LabelEncoder()
for col in multi_value_cols:
    dataset[col] = le.fit_transform(dataset[col].astype(str))

# One-hot encode binary columns (excluding Churn)
binary_cols_excl_churn = [col for col in binary_cols if col != 'Churn']
dummies = pd.get_dummies(dataset[binary_cols_excl_churn], prefix=binary_cols_excl_churn)
dataset.drop(columns=binary_cols, inplace=True)
dataset = pd.concat([dataset, dummies], axis=1)

# Convert Churn to binary
dataset['Churn'] = dataset['Churn'].map({'Yes': 1, 'No': 0})

# --------------------- Handling Missing Data ---------------------
# Impute missing values using KNN
imputer = KNNImputer(n_neighbors=5)
dataset_imputed = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)

# --------------------- Model Training and Evaluation ---------------------
# Prepare training data
X = dataset_imputed.drop(columns=["CustomerID", "Churn"])
y = dataset_imputed["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "Decision Tree": DecisionTreeClassifier(criterion='entropy', random_state=42),
    "SGD": SGDClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=4000)
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n🔍 {name} Model Results:")
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
