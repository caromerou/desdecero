import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Título de la aplicación
st.title("Visualización de Gráficos de Machine Learning")

# Cargar dataset
uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Datos del archivo cargado:")
    st.dataframe(data)

    # Seleccionar características y etiquetas
    target = st.selectbox("Seleccionar la columna objetivo", data.columns)
    features = st.multiselect("Seleccionar las columnas de características", [col for col in data.columns if col != target])

    if st.button("Entrenar modelo"):
        # Separar datos en características y etiquetas
        X = data[features]
        y = data[target]

        # Dividir el dataset en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Entrenar el modelo
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Realizar predicciones
        y_pred = model.predict(X_test)

        # Mostrar métricas de clasificación
        st.write("Reporte de Clasificación:")
        st.text(classification_report(y_test, y_pred))

        # Matriz de Confusión
        cm = confusion_matrix(y_test, y_pred)
        st.write("Matriz de Confusión:")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Verdadero')
        ax.set_title('Matriz de Confusión')
        st.pyplot(fig)

        # Importancia de las características
        feature_importances = pd.DataFrame(model.feature_importances_,
                                           index = X_train.columns,
                                           columns=['Importancia']).sort_values('Importancia', ascending=False)
        st.write("Importancia de las Características:")
        st.bar_chart(feature_importances)

else:
    st.write("Por favor, cargue un archivo CSV para continuar.")



import streamlit as st
import pandas as pd
import io

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


