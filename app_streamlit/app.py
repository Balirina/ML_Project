import streamlit as st 
import variables as vv
from model import cargar_modelo, predict_precio

# page config
st.set_page_config(page_title="Predictor de Precios de Coches", page_icon="🚘", layout='wide', initial_sidebar_state="expanded")

st.title("Bienvenido a la pàgina màs eficiente de predecir precios de coches!")
st.image('img/AutoScout.jpg', use_container_width=True)
# Cargar datos y modelo
try:
    modelo_barratos = cargar_modelo('models/best_models/XGB_baratos_model.pkl')
    modelo_carros = cargar_modelo('models/best_models/GB_caros_model.pkl')
except Exception as e:
    st.error(f"Error al el modelo: {e}")
    st.stop()
    
st.sidebar.header("Especificaciones del Coche")

opciones_coche = ["Coche ordinario","Coche de lujo"]
seleccion_usuario = st.sidebar.radio(
        label="Elige una categoría de coche",
        options=opciones_coche,
        index=0, # Selecciona "Coche de Lujo" por defecto
        horizontal=True # Muestra las opciones en formato vertical (True para horizontal)
    )

brand_selected = st.sidebar.selectbox('Marca', vv.BRAND_ENCODING)
if brand_selected:
    if brand_selected in vv.MODEL_ENCODING:
        models_for_brand = list(vv.MODEL_ENCODING[brand_selected].keys())
        model_selected = st.sidebar.selectbox(
            "Modelo",
            models_for_brand
        )
    else:
        st.sidebar.warning(f"⚠️ No hay modelos disponibles para {brand_selected}")

year_selected = st.sidebar.slider('Año', 1946, 2025)
country_selected = st.sidebar.selectbox('Pais', vv.COUNTRY)
km_selected = st.sidebar.text_input('Kilometros', value='0')
gearbox_selected = st.sidebar.selectbox('Transmisión', vv.GEARBOX)
fuel_selected = st.sidebar.selectbox('Combustible', vv.FUEL)
seller_selected = st.sidebar.selectbox('Vendedor', vv.SELLER)
bodytype_selected = st.sidebar.selectbox('Corrocería', vv.BODY_TYPE)
type_selected = st.sidebar.selectbox('Tipo de vehículo', vv.TYPE)
drivetrain_selected = st.sidebar.selectbox('Eje motriz', vv.DRIVETRAIN)
seats_selected = st.sidebar.slider('Asientos',1, 9 )
doors_selected = st.sidebar.slider('Nº de puertas', 2, 7)
color_selected = st.sidebar.selectbox('Color', vv.COLOR)
upholstery_selected = st.sidebar.selectbox('Material', vv.UPHOLSTERY)

if st.sidebar.button('Predecir Precio', type='primary'):
    with st.spinner('Calculando predicción...'):
        try:
            # Crear diccionario con todas las entradas del usuario
            datos_entrada = {
                'Brand': brand_selected,
                'Model': model_selected,
                'Country': country_selected,
                'Kilometers': km_selected,
                'Gearbox': gearbox_selected,
                'Year': year_selected,
                'Fuel': fuel_selected,
                'Seller': seller_selected,
                'Body Type': bodytype_selected,
                'Type': type_selected,
                'Drivetrain': drivetrain_selected,
                'Seats': seats_selected,
                'Doors': doors_selected,
                'Color': color_selected,
                'Upholstery': upholstery_selected
            }
            # Realizar la predicción y mostrar el resultado
            if seleccion_usuario == 'Coche ordinario':
                modelo = modelo_barratos
                tipo = 'barato'
            elif seleccion_usuario == 'Coche de lujo':
                modelo = modelo_carros
                tipo = 'caro'
            precio_predicho = predict_precio(datos_entrada, modelo, tipo)
            st.success(f"### Precio estimado: €{precio_predicho:,.2f}")
            
        except Exception as e:
            st.error(f"Error en la predicción: {str(e)}")

