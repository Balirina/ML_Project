import variables as vv
import pandas as pd
import pickle
from datetime import datetime
import xgboost as xgb

def cargar_modelo(ruta_modelo):
    with open(ruta_modelo, 'rb') as archivo:
        modelo = pickle.load(archivo)
    return modelo

def predict_precio(datos_coche, modelo, tipo = 'barato'):
    coche = {}
    coche['Brand_c'] = vv.BRAND_ENCODING[datos_coche['Brand']]
    coche['Kilometers'] = int(datos_coche['Kilometers'])
    coche['Seats'] = datos_coche['Seats']
    coche['Doors'] = datos_coche['Doors']
    lista_models = vv.MODEL_ENCODING[datos_coche['Brand']]
    coche['Brand_Model_code'] = lista_models[datos_coche['Model']]
    current_year = datetime.now().year
    coche['Car_Age'] = current_year - datos_coche['Year']
    coche['Country_c'] = vv.COUNTRY[datos_coche['Country']]
    coche['Gearbox_c'] = vv.GEARBOX[datos_coche['Gearbox']]
    coche['Fuel_c'] = vv.FUEL[datos_coche['Fuel']]
    coche['Seller_c'] = vv.SELLER[datos_coche['Seller']]
    coche['Type_c'] = vv.TYPE[datos_coche['Type']]
    coche['Drivetrain_c'] = vv.DRIVETRAIN[datos_coche['Drivetrain']]
    coche['Upholstery_c'] = vv.UPHOLSTERY[datos_coche['Upholstery']]
    coche['Color_c'] = vv.COLOR[datos_coche['Color']]
    coche['Body_Type_c'] = vv.BODY_TYPE[datos_coche['Body Type']]
    df_coche = pd.DataFrame([coche])
    if tipo == 'barato':
        df_coche = xgb.DMatrix(df_coche)
    pred = modelo.predict(df_coche)
    return float(pred[0])
