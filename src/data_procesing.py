import pandas as pd
from datetime import datetime

def normalizar_por_marca(grupo):
    # Normalizar precios por marca
    precio_min = grupo['Price'].min()
    precio_max = grupo['Price'].max()
    rango = precio_max - precio_min
    
    if rango > 0:
        return 0.001 + 0.998 * (grupo['Price'] - precio_min) / rango
    else:
        n = len(grupo)
        return [0.001 + i * (0.998 / (n - 1)) for i in range(n)] if n > 1 else [0.5]

def codificar_Brand_Model(df):
    # borrar las lineas que tienen NaN en modelo
    df = df.dropna(subset = ['Model'])

    # Crear mapeo para la marca del coche: precio medio menos caro → número más bajo
    brand_counts = df.groupby('Brand')['Price'].mean().sort_values(ascending=True)
    frequency_order = brand_counts.index.tolist()
    encoding_dict = {brand: i for i, brand in enumerate(frequency_order)}
    df.insert(1, 'Brand_c' , df['Brand'].map(encoding_dict))

    #Crear mapeo para la marca y el modelo del coche: 
    # Calcular precio promedio por modelo
    precio_promedio_modelo = df.groupby(['Brand_c', 'Model'])['Price'].mean().reset_index()

    precio_promedio_modelo['Model_decimal'] = precio_promedio_modelo.groupby('Brand_c')['Price'].transform(
        lambda x: normalizar_por_marca(pd.DataFrame({'Price': x}))
    )

    # Crear nueva columna de nr. decimales, donde la parte entera es el codigo del Brand y la parte decimal el codigo del Model
    precio_promedio_modelo['Brand_Model_code'] = (precio_promedio_modelo['Brand_c'] + precio_promedio_modelo['Model_decimal'].round(3))

    # Mapear al DataFrame original
    codigo_map = precio_promedio_modelo.set_index(['Brand_c', 'Model'])['Brand_Model_code'].to_dict()
    df['Brand_Model_code'] = df.apply(lambda r: codigo_map[(r['Brand_c'], r['Model'])], axis=1)
    
    print("==="*20)
    print("✅ La marca y el modelo codificados correctamente! ")
    return df

def rellenar_nans(df):
    # borrar las lineas que tienen más de 3 columnas vacias
    df = df[df.isna().sum(axis=1) <= 3]
    
    # cambiar el numero de puertas a 5 donde por error se guardo 35
    df.loc[df['Doors'] == 35, 'Doors'] = 5

    # cambiar el año a 2025 y los kilometros a 0 si no existen y el coche es de tipo nuevo
    df.loc[df['Year'].isna() & (df['Type'] == 'New') & (df['Kilometers'].isna()), 'Kilometers'] = 0
    df.loc[df['Year'].isna() & (df['Type'] == 'New'), 'Year'] = 2025

    # Borrar las lineas que no tienen año ni modelo
    df = df.dropna(subset=['Year'])
    df = df.dropna(subset= ['Model'])

    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    # Si es antiguo (<=2013) y no tiene Gearbox, asumir Manual
    df.loc[df['Gearbox'].isna() & (df['Year'] <= 2013), 'Gearbox'] = 'Manual'

    # Si es moderno (>2013) y no tiene Gearbox, asumir Automatic
    df.loc[df['Gearbox'].isna() & (df['Year'] > 2013), 'Gearbox'] = 'Automatic'

    # Si no tiene Drivetrain y los caballos son inferiores a 200, asumir que el Drivetrain es Front
    df.loc[df['Drivetrain'].isna() & (df['Power'] <= 200), 'Drivetrain'] = 'Front'
    # Si no tiene Drivetrain y los caballos son superiores a 200, asumir que el Drivetrain es 4WD
    df.loc[df['Drivetrain'].isna() & (df['Power'] > 200), 'Drivetrain'] = '4WD'
    # Borrar los registros que quedan vacios
    df = df.dropna(subset= ['Drivetrain'])

    # sustituir NaN con la moda
    df['Fuel'] = df['Fuel'].fillna(df['Fuel'].mode()[0])
    df['Seats'] = df['Seats'].fillna(df['Seats'].mode()[0])
    df['Doors'] = df['Doors'].fillna(df['Doors'].mode()[0])
    df['Upholstery'] = df['Upholstery'].fillna(df['Upholstery'].mode()[0])
    df['Color'] = df['Color'].fillna(df['Color'].mode()[0])

    # cambiar el tipo de la columna km de object a float
    df['Kilometers'] = df['Kilometers'].astype(float)

    # Transformar la columna Year a una columna que represente la edad del coche
    current_year = datetime.now().year
    df['Car_Age'] = current_year - df['Year']
    
    print("==="*20)
    print("✅ Los nullos han sido rellenado correctamente! ")
    return df

def mappear_object_columnas(df):
    mapping = {cat: i for i, cat in enumerate(df['Country'].value_counts().index)}
    df['Country_c'] = df['Country'].map(mapping)

    mapping = {cat: i for i, cat in enumerate(df['Gearbox'].value_counts().index)}
    df['Gearbox_c'] = df['Gearbox'].map(mapping)

    mapping = {cat: i for i, cat in enumerate(df['Fuel'].value_counts().index)}
    df['Fuel_c'] = df['Fuel'].map(mapping)

    mapping = {cat: i for i, cat in enumerate(df['Seller'].value_counts().index)}
    df['Seller_c'] = df['Seller'].map(mapping)

    mapping = {cat: i for i, cat in enumerate(df['Type'].value_counts().index)}
    df['Type_c'] = df['Type'].map(mapping)

    mapping = {cat: i for i, cat in enumerate(df['Drivetrain'].value_counts().index)}
    df['Drivetrain_c'] = df['Drivetrain'].map(mapping)

    mapping = {cat: i for i, cat in enumerate(df['Upholstery'].value_counts().index)}
    df['Upholstery_c'] = df['Upholstery'].map(mapping)

    mapping = {cat: i for i, cat in enumerate(df['Color'].value_counts().index)}
    df['Color_c'] = df['Color'].map(mapping)

    mapping = {cat: i for i, cat in enumerate(df['Body Type'].value_counts().index)}
    df['Body_Type_c'] = df['Body Type'].map(mapping)
    
    print("==="*20)
    print("✅ El mappeo se ha hecho correctamente! ")
    return df
    
def hacer_feature_engineering():
    df = pd.read_csv('../data/raw/car_prices-EU.csv')
    
    df = codificar_Brand_Model(df)
    df = rellenar_nans(df)
    df = mappear_object_columnas(df)
    
    # Borrar las columnas no numericas e innecesarias
    df = df.drop(columns=['Brand', 'Model', 'Country','Cylinders','Gearbox','Fuel','Seller','Body Type', 'Type', 'Drivetrain', 'Color', 'Upholstery', 'Power', 'Year'])
    
    # Hacer un ultimo dropna por si queda algún nulo
    df = df.dropna()
    
    # Guardar el DataSet entero processado
    df.to_csv('../data/processed/car_prices_procc.csv', index = False)
    
    # Guardar DataSet con coches que NO superan los 60000 euros
    df_coches_caros = df[df['Price']>60000]
    df_coches_caros.to_csv('../data/processed/coches_caros_procc.csv', index = False)
    
    # Guardar DataSet con coches que superan los 60000 euros
    df_coches_baratos = df[df['Price']<=60000]
    df_coches_baratos.to_csv('../data/processed/coches_baratos_procc.csv', index = False)
    
    print("==="*20)
    print("DataSets guardados correctamente! 🚀 Listo para entrenar los modelos! 🚀 ")
    print("==="*20)
    
hacer_feature_engineering()