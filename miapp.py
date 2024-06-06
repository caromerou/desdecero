import streamlit as st
import subprocess

# Título de la aplicación
st.title("Cargar y ejecutar archivo Python en Streamlit")

# Instrucción para cargar el archivo Python
uploaded_py_file = st.file_uploader("Cargar archivo Python", type=["py"])

# Instrucción para cargar el archivo de requisitos
uploaded_txt_file = st.file_uploader("Cargar archivo de requisitos (requirements.txt)", type=["txt"])

if uploaded_py_file is not None:
    # Mostrar el nombre del archivo Python
    st.write(f"Archivo Python cargado: {uploaded_py_file.name}")
    
    # Leer el contenido del archivo Python
    content_py = uploaded_py_file.getvalue().decode("utf-8")
    
    # Mostrar el contenido del archivo en un área de texto
    st.write("Contenido del archivo Python:")
    st.code(content_py, language='python')

    if uploaded_txt_file is not None:
        # Mostrar el nombre del archivo de requisitos
        st.write(f"Archivo de requisitos cargado: {uploaded_txt_file.name}")
        
        # Leer el contenido del archivo de requisitos
        content_txt = uploaded_txt_file.getvalue().decode("utf-8")
        
        # Mostrar el contenido del archivo en un área de texto
        st.write("Contenido del archivo de requisitos:")
        st.text(content_txt)

        # Botón para ejecutar el archivo Python
        if st.button("Ejecutar archivo Python"):
            try:
                # Escribir el contenido del archivo de requisitos en un archivo requirements.txt
                with open("requirements.txt", "w") as f:
                    f.write(content_txt)
                
                # Ejecutar el archivo Python cargado
                result = subprocess.run(["python", "-m", "pip", "install", "-r", "requirements.txt"], capture_output=True, text=True)
                if result.returncode == 0:
                    exec(content_py)
                else:
                    st.error(f"Error al instalar las dependencias del archivo Python: {result.stderr}")
            except Exception as e:
                st.error(f"Error al ejecutar el archivo Python: {e}")
else:
    st.write("Por favor, cargue un archivo Python para continuar.")
