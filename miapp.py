import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Título de la aplicación
st.title("Cargar y visualizar archivo Python")

# Instrucción para cargar el archivo
uploaded_file = st.file_uploader("Cargar archivo Python", type=["py"])

if uploaded_file is not None:
    # Mostrar el nombre del archivo
    st.write(f"Archivo cargado: {uploaded_file.name}")
    
    # Leer y mostrar el contenido del archivo Python
    content = uploaded_file.read().decode("utf-8")
    st.write("Contenido del archivo Python:")
    st.code(content, language='python')
    
    # Ejecutar el archivo Python
    try:
        exec_globals = {}
        exec(content, exec_globals)
    except Exception as e:
        st.error(f"Error al ejecutar el archivo: {e}")
    
    # Suponiendo que el archivo define un DataFrame `df`
    if 'df' in exec_globals:
        df = exec_globals['df']  # Obtener el DataFrame `df`
        
        # Botones para mostrar las diferentes instrucciones
        if st.button('Mostrar df.head(10)'):
            st.write("df.head(10):")
            st.write(df.head(10))

        if st.button('Mostrar df.shape'):
            st.write("df.shape:")
            st.write(df.shape)

        if st.button('Mostrar df.info()'):
            st.write("df.info():")
            buffer = io.StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)

        if st.button('Mostrar df.describe()'):
            st.write("df.describe():")
            st.write(df.describe())
    else:
        st.write("El archivo Python no define un DataFrame llamado `df`.")
else:
    st.write("Por favor, cargue un archivo Python para continuar.")
