import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# Función para cargar y preprocesar los datos
def load_and_preprocess_data():
    df = pd.read_csv('/kaggle/input/fraude/PS_20174392719_1491204439457_log.csv')
    
    # Codificar variables categóricas
    le = LabelEncoder()
    df['type'] = le.fit_transform(df['type'])

    # Escalar características numéricas
    scaler = StandardScaler()
    df[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']] = scaler.fit_transform(
        df[['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
    )
    
    return df

# Función para entrenar un modelo de árbol de decisión
def train_decision_tree(X_train, y_train):
    # Crear el modelo
    clf = DecisionTreeClassifier(random_state=42)

    # Entrenar el modelo
    clf.fit(X_train, y_train)

    return clf

# Función para entrenar un modelo de Gradient Boosting
def train_gradient_boosting(X_train, y_train):
    # Crear el modelo
    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    # Entrenar el modelo
    gb_model.fit(X_train, y_train)

    return gb_model

# Función para entrenar un modelo de SVM
def train_support_vector_machine(X_train, y_train):
    # Crear el modelo
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')

    # Entrenar el modelo
    svm_model.fit(X_train, y_train)

    return svm_model

# Función principal de la aplicación
def main():
    st.title("Visualización de Modelos de Aprendizaje Automático")

    # Cargar y preprocesar los datos
    df = load_and_preprocess_data()

    # Dividir los datos
    X = df.drop(['isFraud', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
    y = df['isFraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar modelos
    clf = train_decision_tree(X_train, y_train)
    gb_model = train_gradient_boosting(X_train, y_train)
    svm_model = train_support_vector_machine(X_train, y_train)

    # Evaluar modelos
    st.write("**Decision Tree Classifier**")
    evaluate_model(clf, X_test, y_test)

    st.write("**Gradient Boosting Classifier**")
    evaluate_model(gb_model, X_test, y_test)

    st.write("**Support Vector Machine (SVM)**")
    evaluate_model(svm_model, X_test, y_test)

# Función para evaluar el modelo y mostrar resultados
def evaluate_model(model, X_test, y_test):
    # Hacer predicciones
    y_pred = model.predict(X_test)

    # Mostrar resultados
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    st.write("Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()


