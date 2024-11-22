import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Evaluate the model
accuracy = accuracy_score(y_test, model.predict(X_test))

# Streamlit UI
st.title("Iris Flower Classifier")
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length", float(X[:, 0].min()), float(X[:, 0].max()))
sepal_width = st.sidebar.slider("Sepal Width", float(X[:, 1].min()), float(X[:, 1].max()))
petal_length = st.sidebar.slider("Petal Length", float(X[:, 2].min()), float(X[:, 2].max()))
petal_width = st.sidebar.slider("Petal Width", float(X[:, 3].min()), float(X[:, 3].max()))

# Prediction
input_features = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_features)
predicted_class = iris.target_names[prediction[0]]

st.write(f"Predicted Class: **{predicted_class}**")
st.write(f"Model Accuracy: **{accuracy * 100:.2f}%**")
