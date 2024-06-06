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
