import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TEST_SIZE = 0.25


def train_testing(x, y):
    """--- División en entrenamiento y prueba ---"""
    # test_size=0.25: el 25% de los datos se reserva para evaluar el modelo
    # El 75% restante se usa para entrenar (evita overfitting al evaluar con datos "vistos")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=42
    )

    # TIP: se puede jugar con el tamaño de prueba

    """ --- Escalado de features (algo un poco opcional en Random Forest pero lo dejo :)  --- """
    # Si un dato tiene valores mucho mayores que otra dominará la distancia.
    # StandardScaler estandariza cada feature: z = (x - media) / desviación_estándar
    sc = StandardScaler()
    # fit_transform: calcula media y std del set de ENTRENAMIENTO y transforma
    x_train = sc.fit_transform(x_train)
    # transform: aplica la misma transformación al set de PRUEBA (no recalcula media/std)
    x_test = sc.transform(x_test)

    """ --- Entrenamiento del modelo Random Forest --- """
    # n_estimators=100 por defecto: cantidad de árboles de decisión en el bosque
    # random_state: un número que actúa como semilla aleatoria
    classifier = RandomForestClassifier(random_state=42)
    # fit: el modelo "aprende" la relación features -> objetivo a partir de los datos
    classifier.fit(x_train, y_train)

    """ --- Evaluación del modelo --- """
    # predict: el modelo predice la clase para cada muestra del set de prueba
    y_pred = classifier.predict(x_test)
    # accuracy_score: proporción de predicciones correctas (0.0 a 1.0, se multiplica por 100)
    print(
        f"Porcentaje de resultados predichos correctamente: {round(accuracy_score(y_test, y_pred) * 100, 2)}%"
    )
    # Ej: Porcentaje de resultados predichos correctamente: 79½

    print("Confusion matrix:")

    # confusion_matrix devuelve [[TN, FP], [FN, TP]]
    # .ravel() lo aplana a [TN, FP, FN, TP]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"""
    RESULTADOS:
    {"-" * 40}
    TN: {tn} verdaderos negativos (murió y se predijo muerte)
    TP: {tp} verdaderos positivos (sobrevivió y se predijo supervivencia)

    FP: {fp} falsos positivos (murió y se predijo supervivencia)
    FN: {fn} falsos negativos (sobrevivió y se predijo muerte)
    {"-" * 40}

    """)


# Carga de datos
df = pd.read_csv("heart_clean.csv")

x = df.drop("has_heart_disease", axis=1)  # todas las columnas excepto has_heart_disease
y = df["has_heart_disease"]  # solo la columna target
train_testing(x, y)
