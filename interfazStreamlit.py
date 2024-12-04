import streamlit as st
from datetime import datetime

# Configuración de la página
st.set_page_config(page_title="Interfaz ML", layout="wide")
#logo brain icon
st.image("./cerebro.png", width=100)
# Pestañas: Entrenamiento y Predicción
tabs = st.tabs(["Entrenamiento", "Predicción"])

with tabs[0]:  # Pestaña de Entrenamiento
    st.header("Entrenamiento")
    
    # Selección de fuentes de datos
    col1, col2 = st.columns([4, 1])
    with col1:
        file_data_a = st.text_input("Fuente de datos A", placeholder="Ruta")
    with col2:
        st.button("Abrir A")
    
    col3, col4 = st.columns([4, 1])
    with col3:
        file_data_b = st.text_input("Fuente de datos B", placeholder="Ruta")
    with col4:
        st.button("Abrir B")
    
    # Selección de algoritmo
    st.subheader("Seleccionar Algoritmo:")
    algorithm = st.selectbox("Algoritmo", ["Árbol de decisión", "Regresión logística", "SVM"])
    
    # Vista previa
    st.subheader("VISTA PREVIA:")
    st.text(f"Fecha: {datetime.now().strftime('%d/%m/%Y')}")
    st.text(f"Ejemplares: 1,646")
    st.text(f"Tiempo de entrenamiento: 00:01:27")
    st.text(f"Algoritmo seleccionado: {algorithm}")
    
    # Botón de ejecución
    if st.button("Ejecutar"):
        st.success("¡Entrenamiento completado con éxito!")
    
    # Resultado con gráficos
    st.subheader("Resultado:")
    col_result_1, col_result_2 = st.columns([3, 1])
    with col_result_1:
        st.write("Error medio: ???")
    with col_result_2:
        st.image("https://via.placeholder.com/150", caption="Gráfico de resultados")
    
    # Guardar modelo
    st.subheader("Guardar modelo:")
    col_save_1, col_save_2 = st.columns([4, 1])
    with col_save_1:
        model_path = st.text_input("Ruta para guardar", placeholder="Carpeta1/Carpeta2/Fichero.ext")
    with col_save_2:
        st.button("Guardar")

with tabs[1]:  # Pestaña de Predicción
    st.header("Predicción")
    st.write("Esta sección estará dedicada a predicciones futuras.")
