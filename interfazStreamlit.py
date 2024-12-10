import streamlit as st
from datetime import datetime
from streamlit_extras.grid import grid
import pandas as pd
import numpy as np



# Configuración de la página
st.set_page_config(
    page_title="Interfaz Proyecto",  # Título de la pestaña
    page_icon="./cerebro.png",       # Ruta al favicon
    layout="wide"
)




# Estilo de la página
st.title("Proyecto de Entrenamiento y Predicción")
st.markdown("---")


# Pestañas: Entrenamiento y Predicción
tabs = st.tabs(["Entrenamiento", "Predicción"])

with tabs[0]:  # Pestaña de Entrenamiento
    st.header("Entrenamiento")
    st.markdown("### Configuración de Entrenamiento")

    random_df = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])

    my_grid = grid([2,1], [2,1], [2,1], 1,[2,1],1,1, vertical_align='center')

    # Row 1:
    file_data_a = my_grid.text_input("Fuente de datos A", placeholder="Ruta")
    uploaded_file_a = my_grid.file_uploader("Subir archivo A", type=["csv", "txt", "xlsx", "json"])
    if uploaded_file_a is not None:
        if uploaded_file_a.name.split('.')[-1]=="csv":
            dataframe = pd.read_csv(uploaded_file_a)
            st.write(dataframe)
            file_data_a = uploaded_file_a.name
    file_data_b = my_grid.text_input("Fuente de datos B", placeholder="Ruta")
    uploaded_file_b = my_grid.file_uploader("Subir archivo B", type=["csv", "txt", "xlsx", "json"])
    
    
    # Row 2:
    #TODO: Guardar el modelo en una ruta específica
    model_path = my_grid.text_input("Ruta para guardar", placeholder="Carpeta1/Carpeta2/Fichero.ext")
    my_grid.button("Guardar Modelo")
    
    
    # Row 3:
    # Selección de algoritmo en una nueva fila
    my_grid.markdown("### Selección de Algoritmo")
    algorithm = my_grid.selectbox("Algoritmo", ["Árbol de decisión", "Regresión logística", "SVM"])

    # Ejecución del modelo
    if my_grid.button("Ejecutar"):
        #TODO: Entrenar el modelo con los datos 
        my_grid.success("¡Entrenamiento completado con éxito!")
    
    # Vista previa de datos
    #TODO: Mostrar una vista previa de los datos
    my_grid.markdown("### Vista Previa")
    my_grid.info(
        f"**Fecha:** {datetime.now().strftime('%d/%m/%Y')}\n\n"
        f"**Ejemplares:** 1,646\n\n"
        f"**Tiempo de entrenamiento:** 00:01:27\n\n"
        f"**Algoritmo seleccionado:** {algorithm}"
    )


    # Mostrar resultados en una cuadrícula
    my_grid.markdown("### Resultados")
    result_cols = grid(2, vertical_align="center")  # Crear cuadrícula de resultados
    

    

with tabs[1]:  # Pestaña de Predicción
    st.header("Predicción")
    st.write("Esta sección estará dedicada a predicciones futuras.")
