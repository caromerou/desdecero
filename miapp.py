import streamlit as st

# Título de la aplicación
st.title("Ejecutar archivo Python en Streamlit")

# Instrucción para cargar el archivo Python
uploaded_file = st.file_uploader("Cargar archivo Python", type=["py"])

if uploaded_file is not None:
    # Mostrar el nombre del archivo
    st.write(f"Archivo cargado: {uploaded_file.name}")
    
    # Leer el contenido del archivo Python
    content = uploaded_file.getvalue().decode("utf-8")
    
    # Mostrar el contenido del archivo en un área de texto
    st.write("Contenido del archivo Python:")
    code_input = st.text_area("Código", content, height=300)
    
    # Ejecutar el código Python si se presiona el botón
    if st.button("Ejecutar"):
        try:
            exec(code_input)
        except Exception as e:
            st.error(f"Error al ejecutar el código: {e}")
else:
    st.write("Por favor, cargue un archivo Python para continuar.")
