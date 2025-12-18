import streamlit as st 
import variables as vv
from model import cargar_modelo, predict_precio

# page config
st.set_page_config(page_title="Predictor de Precios de Coches", page_icon="🚘", layout='wide', initial_sidebar_state="expanded")

# Cargar los modelo
try:
    modelo_baratos = cargar_modelo('models/coches_baratos/Best_Model.pkl')
    modelo_caros = cargar_modelo('models/coches_caros/Best_model.pkl')
except Exception as e:
    st.error(f"Error al el modelo: {e}")
    st.stop()

left_spacer, center_col, right_spacer = st.columns([1, 3, 1])
with center_col: 
    with st.container(): 
        st.title("Bienvenido a la pàgina màs eficiente de predecir precios de coches!")
        st.image('img/AutoScout.jpg', width='stretch') 
        st.header("Especificaciones del coche")
        opciones_coche = ["Coche ordinario","Coche de lujo"]
        seleccion_usuario = st.radio(
            label="Elige una categoría de coche",
            options=opciones_coche,
            index=0,
            horizontal=True
        )

        col1, col2 = st.columns(2)
        with col1:
            brand_selected = st.selectbox('Marca', vv.BRAND_ENCODING)
        with col2:
            if brand_selected:
                if brand_selected in vv.MODEL_ENCODING:
                    models_for_brand = list(vv.MODEL_ENCODING[brand_selected].keys())
                    model_selected = st.selectbox(
                        "Modelo",
                        models_for_brand
                    )
                else:
                    st.warning(f"⚠️ No hay modelos disponibles para {brand_selected}")

        year_selected = st.slider('Año', 1946, 2025)

        col3, col4 = st.columns(2)
        with col3:
            km_selected = st.text_input('Kilometros', value='0')
        with col4:
            gearbox_selected = st.selectbox('Transmisión', vv.GEARBOX)

        col5, col6, col7 = st.columns(3)
        with col5:
            fuel_selected = st.selectbox('Combustible', vv.FUEL)
        with col6:
            seller_selected = st.selectbox('Vendedor', vv.SELLER)
        with col7:
            country_selected = st.selectbox('Pais', vv.COUNTRY)
            
        col9, col10, col11 = st.columns(3)
        with col9:
            bodytype_selected = st.selectbox('Corrocería', vv.BODY_TYPE)
        with col10:
            type_selected = st.selectbox('Tipo de vehículo', vv.TYPE)
        with col11:   
            drivetrain_selected = st.selectbox('Eje motriz', vv.DRIVETRAIN)

        col12, col13 = st.columns(2)
        with col12:
            seats_selected = st.slider('Asientos',1, 9 )
        with col13:
            doors_selected = st.slider('Nº de puertas', 2, 7)

        col14, col15 = st.columns(2)
        with col14:
            color_selected = st.selectbox('Color', vv.COLOR)
        with col15:
            upholstery_selected = st.selectbox('Material', vv.UPHOLSTERY)

        col16, col17, col18 = st.columns(3)
        with col17:
            if st.button('Predecir Precio', type='primary'):
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
                            modelo = modelo_baratos
                            tipo = 'barato'
                        elif seleccion_usuario == 'Coche de lujo':
                            modelo = modelo_caros
                            tipo = 'caro'
                        precio_predicho = predict_precio(datos_entrada, modelo, tipo)
                        st.success(f"### Precio estimado: €{precio_predicho:,.2f}")
                        
                    except Exception as e:
                        st.error(f"Error en la predicción: {str(e)}")