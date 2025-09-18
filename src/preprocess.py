# src/preprocess.py (add)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

NUM_FEATURES = ['age','trestbps','chol','thalach','oldpeak']
CAT_FEATURES = ['sex','cp','fbs','restecg','exang','slope','ca','thal']

def build_preprocessor():
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, NUM_FEATURES),
        ('cat', cat_pipe, CAT_FEATURES)
    ])
    return preprocessor


from sklearn.model_selection import train_test_split
from src.preprocess import load_data, build_preprocessor

df = load_data()
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

preprocessor = build_preprocessor()
X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)
