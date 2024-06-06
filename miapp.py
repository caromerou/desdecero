import streamlit as st
import pandas as pd

# Título de la aplicación
st.title("Carga de Archivo CSV Pesado")

# Instrucción para cargar el archivo CSV
uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"])

# Si se cargó un archivo
if uploaded_file is not None:
    # Intentar cargar el archivo CSV
    try:
        # Leer el archivo CSV
        df = pd.read_csv(uploaded_file)
        
        # Mostrar las primeras filas del DataFrame
        st.write("Primeras filas del DataFrame:")
        st.write(df.head())

        # Mostrar información adicional del DataFrame
        st.write("Información del DataFrame:")
        st.write(df.info())

        # Mostrar estadísticas descriptivas del DataFrame
        st.write("Estadísticas descriptivas del DataFrame:")
        st.write(df.describe())
    except Exception as e:
        # Manejar cualquier error al cargar el archivo
        st.error(f"Ocurrió un error al cargar el archivo: {e}")



