import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import randint, uniform
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from skimpy import skim
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_selection import SelectKBest, f_regression
import lightgbm as lgb
import xgboost as xgb
import pickle

# función para cargar el dataset
def cargar_dataset(path):
    df = pd.read_csv(path)
    return df

# ==========================================================
# funcion para dividir el dataframe en train y test
# ==========================================================
def dividir_train_test(df):
    X = df.drop(columns = ['Price'])
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)
    return X_train, X_test, y_train, y_test

# ==========================================================
# funcion para escalar features
# ==========================================================
def escalar_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# ==========================================================
# funcion para entrenar el modelo Linear Regression
# ==========================================================
def entrenar_LR(X_train, X_test, y_train, y_test):
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    predictions = lm.predict(X_test)
    
    # Probar identificar outliers y borrarlos
    residuals = y_test - predictions
    std_residuals = np.std(residuals)
    outlier_threshold = 3 * std_residuals
    
    X_train_clean = X_train.copy()
    y_train_clean = y_train.copy()
    
    # Identificar outliers en entrenamiento también
    train_pred = lm.predict(X_train)
    train_residuals = y_train - train_pred
    train_outliers = np.where(np.abs(train_residuals) > outlier_threshold)[0]

    if len(train_outliers) > 0:
        X_train_clean = np.delete(X_train, train_outliers, axis=0)
        y_train_clean = np.delete(y_train, train_outliers, axis=0)
        
        # Reentrenar modelo
        model_clean = LinearRegression()
        model_clean.fit(X_train_clean, y_train_clean)
        pred = model_clean.predict(X_test)
    # guardar el modelo
    #pickle.dump(lm, open("../models/Linear_Regression_model.pkl", "wb"))
    return model_clean, pred

# ==========================================================
# funcion para entrenar Ridge Regression (L2)
# ==========================================================
def entrenar_Ridge(X_train_scaled, X_test_scaled, y_train):
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    pred = ridge.predict(X_test_scaled)
    #pickle.dump(ridge, open("../models/Ridge_model.pkl", "wb"))
    return ridge, pred

# ==========================================================
# funcion para entrenar Lasso Regression (L1)
# ==========================================================
def entrenar_Lasso(X_train_scaled, X_test_scaled, y_train):
    lasso = Lasso(alpha=0.01, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    pred = lasso.predict(X_test_scaled)
    #pickle.dump(lasso, open("../models/Lasso_model.pkl", "wb"))
    return lasso, pred

# ==========================================================
# funcion para entrenar ElasticNet (combinación L1 + L2)
# ==========================================================
def entrenar_elastic(X_train_scaled, X_test_scaled, y_train):
    elastic = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)
    elastic.fit(X_train_scaled, y_train)
    pred = elastic.predict(X_test_scaled)
    #pickle.dump(elastic, open("../models/Elastic_Net_model.pkl", "wb"))
    return elastic, pred

# ==========================================================
# funcion para entrenar Random Forest 
# ==========================================================
def entrenar_RF(X_train, X_test, y_train):
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    #pickle.dump(rf, open("../models/Random_Forest_model.pkl", "wb")) 
    return rf, pred

# ==========================================================
# funcion para entrenar Gradient Boosting
# ==========================================================
def entrenar_GB(X_train, X_test, y_train):
    gb = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    gb.fit(X_train, y_train)
    pred = gb.predict(X_test)
    #pickle.dump(gb, open("../models/Gradient_Boosting_model.pkl", "wb"))
    return gb, pred

# ==========================================================
# funcion para entrenar LightGB
# ==========================================================
def entrenar_lgb(X_train, X_test, y_train, y_test):
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1,
        'random_state': 42
    }
    
    model_lgb = lgb.train(
        params,
        train_data,
        valid_sets=[test_data],
        num_boost_round=3000,
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)]
    )
    pred = model_lgb.predict(X_test)
    #pickle.dump(model_lgb, open("../models/LightGB_model.pkl", "wb"))
    return model_lgb, pred

# ==========================================================
# funcion para entrenar XGBoost
# ==========================================================
def entrenar_XGB(X_train, X_test, y_train, y_test):
    # Convertir a formato DMatrix (óptimo para XGBoost)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1
    }

    model_xgb = xgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        evals=[(dtest, 'test')],
        early_stopping_rounds=200,
        verbose_eval=100
    )
    
    dtest = xgb.DMatrix(X_test)
    pred = model_xgb.predict(dtest)
    #pickle.dump(model_xgb, open("../models/XGBoost_model.pkl", "wb"))
    return model_xgb, pred

# ==========================================================
# funcion para entrenar modelo No Supervisado
# ==========================================================
def entrenar_noSuperv(df, X, y):
    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clustering para crear segmentos
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    df['segmento_mercado'] = kmeans.fit_predict(X_scaled)

    # PCA para reducir ruido
    pca = PCA(n_components=0.9)
    pca_features = pca.fit_transform(X_scaled)

    # Crear nuevo DataFrame con todo
    X_enhanced = pd.DataFrame(X_scaled, columns=X.columns)
    for i in range(pca.n_components_):
        X_enhanced[f'pca_{i}'] = pca_features[:, i]

    X_enhanced['segmento'] = df['segmento_mercado']
    X_enhanced['dist_centroide'] = kmeans.transform(X_scaled).min(axis=1)

    # Calcular distancia a cada centroide
    distances = kmeans.transform(X_scaled)
    for i in range(kmeans.n_clusters):
        X_enhanced[f'dist_centroide_{i}'] = distances[:, i]

    # Modelo supervisado final
    X_train, X_test, y_train, y_test = train_test_split(
        X_enhanced, y, test_size=0.2, random_state=42
    )
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # 6. Parámetros de XGBoost
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1
    }

    model_xgb = xgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=200,
        verbose_eval=100
    )

    pred = model_xgb.predict(dtest)
    #pickle.dump(model_xgb, open("../models/XGB_NoSuperv_model.pkl", "wb"))
    return model_xgb, pred