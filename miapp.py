import streamlit as st

# Título de la aplicación
st.title("Cargar y visualizar archivo Python")

# Instrucción para cargar el archivo
uploaded_file = st.file_uploader("Cargar archivo Python", type=["py"])

# Verificar si se ha cargado un archivo
if uploaded_file is not None:
    # Mostrar el nombre del archivo
    st.write(f"Archivo cargado: {uploaded_file.name}")
    
    # Leer y mostrar el contenido del archivo Python
    content = uploaded_file.read().decode("utf-8")
    st.write("Contenido del archivo Python:")
    st.code(content, language='python')
else:
    st.write("Por favor, cargue un archivo Python para continuar.")

