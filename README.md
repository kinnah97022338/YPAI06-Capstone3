# YPAI06-Capstone3
# Capstone 3: Customer Segmentation
### Develop a deep learning model to predict the outcome of bank marketing campaigns.
- Use TensorFlow to build a model with only Dense, Dropout, and Batch Normalization layers.
- Achieve model accuracy above 70%.
- Monitor training loss and accuracy using TensorBoard.
- Modularize your code with classes for repeated functions.
import os
import pandas as pd
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
df = pd.read_csv('train.csv')
# inspect data
df.keys()
# Load and preprocess the data
class BankMarketingData:
    def __init__(self, df):
        self.data = pd.read_csv('train.csv')
dataset_train = pd.read_csv('train.csv')
dataset_train.head()
df.info()
df.describe().T
training_set = dataset_train.iloc[:,1:2].values

print(training_set)
print(training_set.shape)
print("rows : "+str(df.shape[0]))
print(df.dtypes)
data = df.copy()
df = df.drop(['last_contact_duration'], axis=1)
# data cleaning"
df.isnull().sum()
# Box plot
df.boxplot(figsize=(20, 20))
plt.show()
df.drop(columns=['id'], inplace=True)
df.hist(figsize=(20,20), edgecolor='black')
plt.show()
df.isnull().sum()
df = df.drop(columns='days_since_prev_campaign_contact')
df.isnull().sum()
# drop the duplicates
dups = df.duplicated()
print('before are there any duplicates : ', dups.any())
df.drop_duplicates(inplace=True)
# reset indices after dropping rows
df=df.reset_index(drop=True)
print('after are there any duplicates : ', df.duplicated().any())
# Check if the column exists before filling null values
if 'last_contact_duration' in df.columns:
    df['customer_age'].fillna(df['customer_age'].mean(), inplace=True)
    df['marital'].fillna('Unknown', inplace=True)
    df['balance'].fillna(df['balance'].median(), inplace=True)
    df['personal_loan'].fillna('Unknown', inplace=True)
    df['last_contact_duration'].fillna(df['last_contact_duration'].median(), inplace=True)
    df['num_contacts_in_campaign'].fillna(df['num_contacts_in_campaign'].median(), inplace=True)
else:
    print("Column 'last_contact_duration' does not exist in the DataFrame.")

# Check for the correct column names
print(df.columns)  # Check all column names in your DataFrame

# Example: If 'last_contact_duration' is not present, handle the available columns
if 'customer_age' in df.columns:
    df['customer_age'].fillna(df['customer_age'].mean(), inplace=True)
if 'marital' in df.columns:
    df['marital'].fillna('Unknown', inplace=True)
# Handle other columns similarly based on their availability
# ...
else:
    print("Column 'last_contact_duration' (or other relevant columns) not found.")

df.isnull().sum()
df['customer_age'].fillna(df['customer_age'].mean(), inplace=True)
df['marital'].fillna('Unknown', inplace=True)
df['balance'].fillna(df['balance'].median(), inplace=True)
df['personal_loan'].fillna('Unknown', inplace=True)
#df['last_contact_duration'].fillna(df['last_contact_duration'].median(), inplace=True)
df['num_contacts_in_campaign'].fillna(df['num_contacts_in_campaign'].median(), inplace=True)
df.duplicated().sum()
# Specify columns to visualize
cols = ['customer_age', 'balance', 'day_of_month', 'num_contacts_in_campaign', 'num_contacts_prev_campaign']
df[cols]
cat_cols = list(df.drop(columns=cols).columns)
 
df[cat_cols] = df[cat_cols].fillna(method='ffill', axis=0)
 
df.isnull().sum()
for col in cols:
    if df[col].dtype in ['int64', 'float64']:  # Check if the column contains numeric data
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
        fig.suptitle(col)
        
        # Boxplot
        axes[0].boxplot(df[col])
        axes[0].set_title('Boxplot')
        
        # Histogram
        axes[1].hist(df[col])
        axes[1].set_title('Histogram')
        
        # Q-Q plot
        stats.probplot(df[col], dist='norm', plot=axes[2])
        axes[2].set_title('Q-Q Plot')
        
        plt.show()
    else:
        print(f"Column '{col}' is not numeric and cannot be visualized.")

# remove outlier of campaign
fig, axes = plt.subplots(1,2)
df2 = df
col='num_contacts_in_campaign'
print("Before Shape:",df2.shape)
axes[0].title.set_text("Before")
sns.boxplot(df2[col],orient='v',ax=axes[0])
# Removing campaign above 50 
df2 = df2[ (df2[col]<50)]
print("After Shape:",df2.shape)
axes[1].title.set_text("After")
sns.boxplot(df2[col],orient='v',ax=axes[1])
df=df2;
plt.show()
# reset indices after dropping rows
df=df.reset_index(drop=True)
# Correlation
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
 
sns.heatmap(df[cols].corr(),
            annot=True,
            ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
plt.show()
df.describe()
numerical_df = df.select_dtypes(include=['float64','int64'])
numerical_df.head()
categorical_df=df.select_dtypes(include='object')
categorical_df.head()
def count_plot(df,feature):
    sns.set(color_codes = 'Blue', style="whitegrid")
    sns.set_style("whitegrid", {'axes.grid' : False})
    sns.set_context(rc = {'patch.linewidth': 0.0})
    fig = plt.subplots(figsize=(10,3))
    sns.countplot(x=feature, data=df, color = 'yellow') # countplot
    plt.show()
df.hist(figsize=(8,8), edgecolor='black')
plt.show()
for cat_col in categorical_df.columns:
    if cat_col in ['job_type','marital','education', 'default','housing_loan','personal_loan']:
        count_plot(df,cat_col)
# Preprocessing
oe = OrdinalEncoder()
df[cat_cols[0: -1]] = oe.fit_transform(df[cat_cols[0: -1]])
df
# Correlation
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
 
sns.heatmap(df[cols].corr(),
            annot=True,
            ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
plt.show()
df.describe()
numerical_df = df.select_dtypes(include=['float64','int64'])
numerical_df.head()
categorical_df=df.select_dtypes(include='object')
categorical_df.head()
def count_plot(df,feature):
    sns.set(color_codes = 'Blue', style="whitegrid")
    sns.set_style("whitegrid", {'axes.grid' : False})
    sns.set_context(rc = {'patch.linewidth': 0.0})
    fig = plt.subplots(figsize=(10,10))
    sns.countplot(x=feature, data=df, color = 'yellow') # countplot
    plt.show()
df.hist(figsize=(8,8), edgecolor='black')
plt.show()
for cat_col in categorical_df.columns:
    if cat_col in ['job_type','marital','education', 'default','housing_loan','personal_loan']:
        count_plot(df,cat_col)
# Preprocessing
oe = OrdinalEncoder()
df[cat_cols[0: -1]] = oe.fit_transform(df[cat_cols[0: -1]])
df
oe.categories_
X = df.drop(columns='term_deposit_subscribed')
X
y = df[['term_deposit_subscribed']]
y
ohe = OneHotEncoder(sparse=False)
 
y_encoded = ohe.fit_transform(y)
y_encoded
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=13)

# Create a Sequential model
model = Sequential()
 
# Add input layer
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
 
# Add hidden layers
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
 
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
 
# Add output layer
model.add(Dense(y_train.shape[1], activation='softmax'))
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
 
# TensorBoard callback for logging training process
 
PATH = os.getcwd()
logpath = os.path.join(PATH, "tensorboard_log", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard = TensorBoard(logpath)
 
# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, tensorboard]
)
model.summary()
plot_model(model, show_shapes=True, show_layer_names=(True))
# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy*100:.2f}%")
model.save(os.path.join('models','classify_v1.h5'))
