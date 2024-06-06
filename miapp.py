import streamlit as st
import pandas as pd

# Título de la aplicación
st.title("Cargar y visualizar archivos")

# Instrucción para cargar el archivo
uploaded_file = st.file_uploader("Cargar archivo", type=["csv", "xlsx", "txt"])

# Verificar si se ha cargado un archivo
if uploaded_file is not None:
    # Mostrar el nombre del archivo
    st.write(f"Archivo cargado: {uploaded_file.name}")

    # Leer y mostrar el contenido del archivo según el tipo
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        st.write("Contenido del archivo CSV:")
        st.dataframe(df)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
        st.write("Contenido del archivo Excel:")
        st.dataframe(df)
    elif uploaded_file.name.endswith('.txt'):
        content = uploaded_file.read().decode("utf-8")
        st.write("Contenido del archivo de texto:")
        st.text(content)
    else:
        st.write("Tipo de archivo no soportado.")
else:
    st.write("Por favor, cargue un archivo para continuar.")
