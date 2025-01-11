import streamlit as st
from datetime import datetime
from tkinter import Tk, filedialog
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd
import numpy as np
import time

# Configuración de la página
st.set_page_config(
    page_title="Interfaz Proyecto",
    page_icon="./cerebro.png",
    layout="wide"
)

# Función para seleccionar solo archivos CSV
def seleccionar_archivo_csv():
    root = Tk()
    root.withdraw()  # Oculta la ventana principal de Tkinter
    root.attributes('-topmost', True)  # Asegura que el cuadro de diálogo esté al frente
    ruta_archivo = filedialog.askopenfilename(filetypes=[("Archivos CSV", "*.csv")])  # Solo permite seleccionar archivos CSV
    root.destroy()  # Cierra la ventana de Tkinter inmediatamente después
    return ruta_archivo

# Función para seleccionar solo archivos .pkl
def seleccionar_archivo_pkl():
    root = Tk()
    root.withdraw()  # Oculta la ventana principal de Tkinter
    root.attributes('-topmost', True)  # Asegura que el cuadro de diálogo esté al frente
    ruta_archivo = filedialog.askopenfilename(filetypes=[("Archivos PKL", "*.pkl")])  # Solo permite seleccionar archivos PKL
    root.destroy()  # Cierra la ventana de Tkinter inmediatamente después
    return ruta_archivo

def seleccionar_carpeta():
    root = Tk()
    root.withdraw()  # Oculta la ventana principal de Tkinter
    root.attributes('-topmost', True)  # Asegura que el cuadro de diálogo esté al frente
    ruta_carpeta = filedialog.askdirectory()  # Abrir cuadro de diálogo para seleccionar carpeta
    root.destroy()  # Cierra la ventana de Tkinter inmediatamente después
    return ruta_carpeta

# Función genérica de entrenamiento
def entrenar_modelo(df, nombre_modelo, algoritmo):
    df = df[['ingredientes', 'nombre', 'supermercado', 'categoria']]
    vectorizer = TfidfVectorizer(max_features=500)
    text_features = vectorizer.fit_transform(df['ingredientes'].astype(str)).toarray()
    numerical_features = df[['nombre', 'supermercado']].apply(pd.to_numeric, errors='coerce').fillna(0)
    X = pd.concat([numerical_features.reset_index(drop=True), pd.DataFrame(text_features)], axis=1)
    X.columns = X.columns.astype(str)  # Convertir nombres de columnas a cadenas
    y = df['categoria']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    start_time = time.time()  # Tiempo inicial
    if algoritmo == "Árbol de Decisión":
        model = DecisionTreeClassifier(random_state=42)
    elif algoritmo == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif algoritmo == "Gradient Boost":
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    else:
        raise ValueError("Algoritmo desconocido.")
    
    model.fit(X_train, y_train)
    end_time = time.time()  # Tiempo final

    model_and_vectorizer = {'model': model, 'vectorizer': vectorizer}
    joblib.dump(model_and_vectorizer, nombre_modelo + '.pkl')

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    training_time = end_time - start_time

    resultados = {
        "accuracy": accuracy,
        "tiempo_ejecucion": training_time,
        "tamano_train": len(X_train),
        "tamano_test": len(X_test),
        "fecha_ejecucion": datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
        "algoritmo": algoritmo
    }

    return resultados

# Inicializar variables en la sesión
if 'ruta_seleccionada' not in st.session_state:
    st.session_state.ruta_seleccionada = None
if 'carpeta_seleccionada' not in st.session_state:
    st.session_state.carpeta_seleccionada = None
if 'dataframe' not in st.session_state:
    st.session_state.dataframe = None
if 'modelo' not in st.session_state:
    st.session_state.modelo = None
if 'vectorizador' not in st.session_state:
    st.session_state.vectorizador = None
if 'df_prediccion' not in st.session_state:
    st.session_state.df_prediccion = None

# Interfaz de usuario
st.title("Proyecto de Entrenamiento y Predicción")
tabs = st.tabs(["Entrenamiento", "Predicción"])

# Pestaña de Entrenamiento
with tabs[0]:  
    st.header("Entrenamiento")
    st.markdown("### Configuración de Entrenamiento")
    if st.button("Seleccionar archivo CSV"):
        st.session_state.ruta_seleccionada = seleccionar_archivo_csv()
        if st.session_state.ruta_seleccionada:
            try:
                st.session_state.dataframe = pd.read_csv(st.session_state.ruta_seleccionada)
                st.success(f"Archivo cargado: {st.session_state.ruta_seleccionada}")
            except Exception as e:
                st.error(f"Error al cargar el archivo: {e}")

    if st.session_state.dataframe is not None:
        st.dataframe(st.session_state.dataframe.head())
        if st.button("Seleccionar carpeta para guardar el modelo"):
            st.session_state.carpeta_seleccionada = seleccionar_carpeta()
            if st.session_state.carpeta_seleccionada:
                st.success(f"Carpeta seleccionada: {st.session_state.carpeta_seleccionada}")

        algorithm = st.selectbox("Seleccionar algoritmo", ["Árbol de Decisión", "Random Forest", "Gradient Boost"])
        if st.button("Entrenar modelo"):
            if st.session_state.carpeta_seleccionada:
                nombre_modelo = f"{st.session_state.carpeta_seleccionada}/modelo_{algorithm.replace(' ', '_').lower()}"
                resultados = entrenar_modelo(st.session_state.dataframe, nombre_modelo, algorithm)

                st.markdown(
                    f"""
                    ### Resultados del Modelo
                    - **Algoritmo Seleccionado:** {resultados['algoritmo']}
                    - **Precisión del Modelo:** {resultados['accuracy']:.2f}
                    - **Tamaño del Conjunto de Entrenamiento:** {resultados['tamano_train']}
                    - **Tamaño del Conjunto de Prueba:** {resultados['tamano_test']}
                    - **Tiempo de Entrenamiento:** {resultados['tiempo_ejecucion']:.2f} segundos
                    - **Fecha y Hora del Entrenamiento:** {resultados['fecha_ejecucion']}
                    """
                )
                st.success("¡Entrenamiento completado con éxito!")
            else:
                st.error("Selecciona una carpeta antes de entrenar.")

# Pestaña de Predicción
with tabs[1]:
    st.header("Predicción")
    st.markdown("### Configuración de Predicción")

    st.markdown("#### Paso 1: Seleccionar archivo del modelo (.pkl)")
    if st.button("Seleccionar archivo del modelo"):
        modelo_path = seleccionar_archivo_pkl()
        if modelo_path:
            try:
                modelo_y_vectorizador = joblib.load(modelo_path)
                st.session_state.modelo = modelo_y_vectorizador['model']
                st.session_state.vectorizador = modelo_y_vectorizador['vectorizer']
                st.success(f"Modelo cargado desde: {modelo_path}")
            except Exception as e:
                st.error(f"Error al cargar el modelo: {e}")
    
    st.markdown("#### Paso 2: Seleccionar archivo para predicción (.csv)")
    if st.button("Seleccionar archivo para predicción"):
        prediccion_path = seleccionar_archivo_csv()
        if prediccion_path:
            try:
                st.session_state.df_prediccion = pd.read_csv(prediccion_path)
                st.success(f"Archivo cargado para predicción: {prediccion_path}")
            except Exception as e:
                st.error(f"Error al cargar el archivo: {e}")

    if 'df_prediccion' in st.session_state and st.session_state.df_prediccion is not None:
        st.markdown("### Vista previa del archivo de predicción")
        st.dataframe(st.session_state.df_prediccion.head())

        if st.button("Ejecutar predicción"):
            try:
                if 'modelo' in st.session_state and 'vectorizador' in st.session_state:
                    ingredientes_features = st.session_state.vectorizador.transform(
                        st.session_state.df_prediccion['ingredientes'].astype(str)
                    ).toarray()

                    if 'nombre' not in st.session_state.df_prediccion:
                        st.session_state.df_prediccion['nombre'] = 0
                    if 'supermercado' not in st.session_state.df_prediccion:
                        st.session_state.df_prediccion['supermercado'] = 0

                    nombre_features = st.session_state.df_prediccion['nombre'].apply(
                        pd.to_numeric, errors='coerce'
                    ).fillna(0)
                    supermercado_features = st.session_state.df_prediccion['supermercado'].apply(
                        pd.to_numeric, errors='coerce'
                    ).fillna(0)

                    X_prediccion = np.hstack([
                        nombre_features.values.reshape(-1, 1),
                        supermercado_features.values.reshape(-1, 1),
                        ingredientes_features
                    ])

                    predicciones = st.session_state.modelo.predict(X_prediccion)

                    st.session_state.df_prediccion['Predicción'] = predicciones
                    st.markdown("### Resultados de la Predicción")
                    st.dataframe(st.session_state.df_prediccion)

                    # Guardar el archivo con las predicciones
                    if st.button("Guardar resultados"):
                        carpeta_destino = seleccionar_carpeta()
                        if carpeta_destino:
                            output_path = f"{carpeta_destino}/resultados_prediccion.csv"
                            st.session_state.df_prediccion.to_csv(output_path, index=False)
                            st.success(f"Resultados guardados en: {output_path}")
                else:
                    st.error("Primero debes cargar un modelo y un vectorizador.")
            except Exception as e:
                st.error(f"Error al realizar la predicción: {e}")
