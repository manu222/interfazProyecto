
import streamlit as st
from datetime import datetime
from streamlit_extras.grid import grid
import pandas as pd
import numpy as np
from tkinter import Tk, filedialog


# Configuración de la página
st.set_page_config(
    page_title="Interfaz Proyecto",  # Título de la pestaña
    page_icon="./cerebro.png",       # Ruta al favicon
    layout="wide"
)

def seleccionar_archivo():
    """
    Abre un cuadro de diálogo para seleccionar un archivo y devuelve su ruta completa.
    """
    root = Tk()
    root.withdraw()  # Oculta la ventana principal de Tkinter
    root.attributes('-topmost', True)  # Asegura que el cuadro de diálogo esté al frente
    ruta_archivo = filedialog.askopenfilename()  # Abrir cuadro de diálogo para seleccionar archivo
    root.destroy()  # Cerrar la ventana de Tkinter
    return ruta_archivo

def seleccionar_carpeta():
    """
    Abre un cuadro de diálogo para seleccionar una carpeta y devuelve su ruta completa.
    """
    root = Tk()
    root.withdraw()  # Oculta la ventana principal de Tkinter
    root.attributes('-topmost', True)  # Asegura que el cuadro de diálogo esté al frente
    ruta_carpeta = filedialog.askdirectory()  # Abrir cuadro de diálogo para seleccionar carpeta
    root.destroy()  # Cerrar la ventana de Tkinter
    return ruta_carpeta

# Estilo de la página
st.title("Proyecto de Entrenamiento y Predicción")
st.markdown("---")

if 'ruta_seleccionada' not in st.session_state:
    st.session_state['ruta_seleccionada'] = None
if 'carpeta_seleccionada' not in st.session_state:
    st.session_state['carpeta_seleccionada'] = None


# Pestañas: Entrenamiento y Predicción
tabs = st.tabs(["Entrenamiento", "Predicción"])

with tabs[0]:  # Pestaña de Entrenamiento
    st.header("Entrenamiento")
    st.markdown("### Configuración de Entrenamiento")
    
    my_grid = grid(1,1,1,1,1,1,1,1,1,1,1, vertical_align='center')

    # Row 1:
   
    # uploaded_file_a = my_grid.file_uploader("Subir archivo A", type=["csv", "txt", "xlsx", "json"])
    if my_grid.button("Seleccionar archivo con Tkinter"):
        st.session_state.ruta_seleccionada = seleccionar_archivo()
        if st.session_state.ruta_seleccionada:
            #Text label
            st.session_state.ruta_seleccionada ="Archivo : "+ st.session_state.ruta_seleccionada
    
    if st.session_state.ruta_seleccionada:
        my_grid.text(st.session_state.ruta_seleccionada)
            
    
    
    # Row 2:
    #TODO: Guardar el modelo en una ruta específica
    
    if my_grid.button("Guardar Modelo"):
        st.session_state.carpeta_seleccionada = seleccionar_carpeta()
        if st.session_state.carpeta_seleccionada:
            #Text label
            st.session_state.carpeta_seleccionada = "Carpeta : "+st.session_state.carpeta_seleccionada
    
    if st.session_state.carpeta_seleccionada:
        my_grid.text(st.session_state.carpeta_seleccionada)
        
    
    
    # Row 3:
    # Selección de algoritmo en una nueva fila
    my_grid.markdown("### Selección de Algoritmo")
    algorithm = my_grid.selectbox("Algoritmo", ["Árbol de decisión", "Regresión logística", "SVM"])

    # Ejecución del modelo
    if my_grid.button("Ejecutar"):
        #TODO: Entrenar el modelo con los datos 
        st.session_state["uploaded_file_path_a"] = "adasdsa"
        my_grid.success("¡Entrenamiento completado con éxito!")
    
    
    # Vista previa de datos
    #TODO: Mostrar una vista previa de los datos
    my_grid.markdown("### Vista Previa")
    my_grid.info(
        f"**Fecha:** {datetime.now().strftime('%d/%m/%Y')}\n\n"
        #Rows del dataset
        f"**Número de filas:** 100\n\n"
        f"**Tiempo de entrenamiento:** 00:01:27\n\n"
        f"**Algoritmo seleccionado:** {algorithm}"
    )


    # Mostrar resultados en una cuadrícula
    my_grid.markdown("### Resultados")
    result_cols = grid(2, vertical_align="center")  # Crear cuadrícula de resultados
    

    

with tabs[1]:  # Pestaña de Predicción
    st.header("Predicción")
    st.write("Esta sección estará dedicada a predicciones futuras.")