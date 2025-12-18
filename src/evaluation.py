import training  
from sklearn import metrics
import numpy as np


path_coches_baratos = '../data/processed/coches_baratos_procc.csv'
path_coches_caros = '../data/processed/coches_caros_procc.csv'

def evaluar_modelo_coches_baratos(path):
    df_coches_baratos = training.cargar_dataset(path)
    X_train, X_test, y_train, y_test = training.dividir_train_test(df_coches_baratos)
    modelo, pred = training.entrenar_XGB(X_train, X_test, y_train, y_test)

    xgb_mae = metrics.mean_absolute_error(y_test, pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, pred))
    r2 = metrics.r2_score(y_test, pred)

    precio_min = df_coches_baratos['Price'].min()
    precio_max = df_coches_baratos['Price'].max()

    precio_promedio_estimado = np.mean([precio_min, precio_max])
    error_relativo_porcentaje = (xgb_mae / precio_promedio_estimado) * 100

    print("📊 RESULTADOS del modelo XGBoost para coches baratos:")
    print("="*60)
    print(f"• Rango de precios: €{precio_min:,} - €{precio_max:,}")
    print(f"• Precio promedio estimado: €{precio_promedio_estimado:,.0f}")
    print(f"•  MAE: €{xgb_mae:,.2f}")
    print(f"•  RMSE: €{rmse:,.2f}")
    print(f"• Error relativo: {error_relativo_porcentaje:.1f}% del precio promedio\n\n\n")
    return pred


def evaluar_modelo_coches_caros(path):
    df_coches_caros = training.cargar_dataset(path)
    X_train, X_test, y_train, y_test = training.dividir_train_test(df_coches_caros)
    modelo, pred = training.entrenar_GB(X_train, X_test, y_train)
    
    gb_mae = metrics.mean_absolute_error(y_test, pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, pred))
    r2 = metrics.r2_score(y_test, pred)

    precio_min = df_coches_caros['Price'].min()
    precio_max = df_coches_caros['Price'].max()

    precio_promedio_estimado = np.mean([precio_min, precio_max])
    error_relativo_porcentaje = (gb_mae / precio_promedio_estimado) * 100

    print("📊 RESULTADOS del modelo Gradient Boosting para coches caros:")
    print("="*60)
    print(f"• Rango de precios: €{precio_min:,} - €{precio_max:,}")
    print(f"• Precio promedio estimado: €{precio_promedio_estimado:,.0f}")
    print(f"•  MAE: €{gb_mae:,.2f}")
    print(f"•  RMSE: €{rmse:,.2f}")
    print(f"• Error relativo: {error_relativo_porcentaje:.1f}% del precio promedio")
    return pred

evaluar_modelo_coches_baratos(path_coches_baratos)
evaluar_modelo_coches_caros(path_coches_caros)