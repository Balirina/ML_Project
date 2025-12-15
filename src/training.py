import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
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
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import pickle

# Cargar los 2 dataset
df_baratos = pd.read_csv('../data/processed/coches_baratos_procc.csv')
df_caros = pd.read_csv('../data/processed/coches_caros_procc.csv')

def entrnar_modelos(df):
    X = df.drop(columns = ['Price'])
    y = df['Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)
    
    # ==========================================================
    # Probar el modelo Linear Regression
    # ==========================================================
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    predictions = lm.predict(X_test)
    pickle.dump(lm, open("../models/Linear_Regression_model.pkl", "wb"))
    
    # Probar identificar outliers y borrarlos
    # Identificar outliers
    residuals = y_test - predictions
    std_residuals = np.std(residuals)
    outlier_threshold = 3 * std_residuals
    outlier_indices = np.where(np.abs(residuals) > outlier_threshold)[0]
    
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
        y_pred_clean = model_clean.predict(X_test)
    
    # Escalar features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ==========================================================
    # Ridge Regression (L2)
    # ==========================================================
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge.predict(X_test_scaled)
    pickle.dump(ridge, open("../models/Ridge_model.pkl", "wb"))
    
    # ==========================================================
    # Lasso Regression (L1)
    # ==========================================================
    lasso = Lasso(alpha=0.01, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    y_pred_lasso = lasso.predict(X_test_scaled)
    pickle.dump(lasso, open("../models/Lasso_model.pkl", "wb"))
    
    # ==========================================================
    #. ElasticNet (combinación L1 + L2)
    # ==========================================================
    elastic = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)
    elastic.fit(X_train_scaled, y_train)
    y_pred_elastic = elastic.predict(X_test_scaled)
    pickle.dump(elastic, open("../models/Elastic_Net_model.pkl", "wb"))

    # ==========================================================
    # 1. Random Forest
    # ==========================================================
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    pickle.dump(rf, open("../models/Random_Forest_model.pkl", "wb"))
    
    # ==========================================================
    # 2. Gradient Boosting
    # ==========================================================
    gb = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    pickle.dump(gb, open("../models/Gradient_Boosting_model.pkl", "wb"))

    # ==========================================================
    # 3. Decision Tree
    # ==========================================================
    tree = DecisionTreeRegressor(
        max_depth=10,
        min_samples_split=10,
        random_state=42
    )
    tree.fit(X_train, y_train)
    y_pred_tree = tree.predict(X_test)
    pickle.dump(tree, open("../models/Decision_Tree_model.pkl", "wb"))
    
    # ==========================================================
    # Configuración mejorada de Gradient Boosting
    # ==========================================================
    gb_better = GradientBoostingRegressor(
        n_estimators=300,           # ↑ Más árboles
        learning_rate=0.08,         # ↓ Tasa de aprendizaje más baja
        max_depth=6,                # ↑ Profundidad moderada
        min_samples_split=10,       # Evitar sobreajuste
        min_samples_leaf=5,         # Regularización
        subsample=0.8,              # Stochastic Gradient Boosting
        max_features='sqrt',        # Usar sqrt(n_features) en cada split
        random_state=42,
        n_iter_no_change=10,        # Parada temprana
        validation_fraction=0.1     # Fracción para validación
    )
    gb_better.fit(X_train, y_train)
    y_pred_gb = gb_better.predict(X_test)
    pickle.dump(gb_better, open("../models/GB_improved_model.pkl", "wb"))
    
    # ==========================================================
    # Gradient Boosting con GridSearch
    # ==========================================================
    param_dist = {
    'n_estimators': randint(100, 400),
    'learning_rate': uniform(0.01, 0.2),
    'max_depth': randint(3, 8),
    'min_samples_split': randint(10, 30),
    'min_samples_leaf': randint(4, 20),
    'subsample': uniform(0.7, 0.3),
    'max_features': ['sqrt', 'log2', 0.5, 0.7]
    }

    gb = GradientBoostingRegressor(random_state=42)

    random_search = RandomizedSearchCV(
        estimator=gb,
        param_distributions=param_dist,
        n_iter=50,
        scoring='neg_mean_squared_error',
        cv=3,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    bgrs_pred = best_model.predict(X_test)
    pickle.dump(best_model, open("../models/GB_withGS_model.pkl", "wb"))
    
    # ==========================================================
    # Gradient Boosting with Pipe
    # ==========================================================
    pipeline = Pipeline([
    ('feature_selection', SelectKBest(f_regression, k=20)), 
    ('regressor', GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    ))])

    pipe_params = {
        'feature_selection__k': [15, 20, 25],
        'regressor__n_estimators': [100, 150],
        'regressor__max_depth': [3, 4]
    }

    grid_search = GridSearchCV(
        pipeline,
        pipe_params,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    gs_pred = grid_search.predict(X_test)
    pickle.dump(grid_search, open("../models/GB_withPipe_model.pkl", "wb"))
    
    # ==========================================================
    # Light Gradient Boosting
    # ==========================================================
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
    y_pred_lgb = model_lgb.predict(X_test)
    pickle.dump(model_lgb, open("../models/LightGB_model.pkl", "wb"))
    
    # ==========================================================
    # XGBoost
    # ==========================================================
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
    y_pred_xgb = model_xgb.predict(dtest)
    pickle.dump(model_xgb, open("../models/XGBoost_model.pkl", "wb"))
    
    # ==========================================================
    # No Supervisado (PCA)
    # ==========================================================
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

    y_pred = model_xgb.predict(dtest)
    pickle.dump(model_xgb, open("../models/XGB_NoSuperv_model.pkl", "wb"))
        