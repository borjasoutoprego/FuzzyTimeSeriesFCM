import numpy as np
import skfuzzy as fuzz


class FuzzyInformationGranules:

    def __init__(self, window_size, n_clusters, m=2):
        """
        Parámetros
        ----------
        window_size : int
            Longitud L de las ventanas temporales.

        n_clusters : int
            Número de clusters fuzzy.

        m : float
            Índice de fuzziness del FCM.
        """
        self.L = window_size
        self.c = n_clusters
        self.m = m

        self.granules = None
        self.centers = None
        self.memberships = None
        self.rules = None

    # ---------------------------------------------------------
    # PASO 1: SEGMENTAR LA SERIE
    # ---------------------------------------------------------

    def create_windows(self, X):
        """
        Crea ventanas deslizantes de longitud L.

        V1 = X[1], ..., X[L]
        V2 = X[2], ..., X[L+1]
        ...

        En Python:
        V1 = X[0:L]
        V2 = X[1:L+1]
        ...
        """

        X = np.asarray(X, dtype=float)

        if len(X) <= self.L:
            raise ValueError(
                "La serie debe tener más observaciones que la "
                "longitud de la ventana."
            )

        windows = np.array([
            X[i:i + self.L]
            for i in range(len(X) - self.L + 1)
        ])

        return windows

    # ---------------------------------------------------------
    # PASO 2: OBTENER GRÁNULOS DE INFORMACIÓN
    # ---------------------------------------------------------

    def extract_features(self, windows):
        """
        Convierte cada ventana en un vector de características.

        Características utilizadas:
        - mínimo
        - máximo
        - rango
        - media
        - desviación típica
        - pendiente de tendencia
        """

        features = []

        # Posiciones temporales dentro de la ventana
        t = np.arange(self.L)

        for window in windows:

            minimum = np.min(window)
            maximum = np.max(window)
            data_range = maximum - minimum
            mean = np.mean(window)
            std = np.std(window)

            # Pendiente de una regresión lineal
            slope = np.polyfit(t, window, 1)[0]

            features.append([
                minimum,
                maximum,
                data_range,
                mean,
                std,
                slope
            ])

        return np.asarray(features)

    # ---------------------------------------------------------
    # PASO 3: FCM SOBRE LOS GRÁNULOS
    # ---------------------------------------------------------

    def fit(self, X):
        """
        Entrena el modelo completo sobre la serie X.
        """

        X = np.asarray(X, dtype=float)

        # Crear ventanas
        windows = self.create_windows(X)

        # Obtener gránulos
        granules = self.extract_features(windows)

        self.windows = windows
        self.granules = granules

        # FCM necesita:
        # filas = características
        # columnas = observaciones
        data = granules.T

        centers, memberships, _, _, _, _, _ = fuzz.cluster.cmeans(
            data,
            c=self.c,
            m=self.m,
            error=0.005,
            maxiter=1000
        )

        self.centers = centers
        self.memberships = memberships

        # Cluster dominante de cada gránulo
        self.labels = np.argmax(memberships, axis=0)

        # Construir reglas cluster -> futuro
        self.build_rules(X)

        return self

    # ---------------------------------------------------------
    # PASO 4: CONSTRUIR REGLAS FUZZY
    # ---------------------------------------------------------

    def build_rules(self, X):
        """
        Para cada ventana V_j, se observa X(j+L).

        Se agrupan esos valores futuros según el cluster
        al que pertenece V_j.

        Regla:

        Cluster i -> valor futuro asociado
        """

        X = np.asarray(X, dtype=float)

        self.rules = {}

        # La última ventana no tiene observación futura
        # conocida dentro de la muestra.
        n_training_windows = len(self.windows) - 1

        for cluster in range(self.c):

            future_values = []

            for j in range(n_training_windows):

                if self.labels[j] == cluster:

                    future_value = X[j + self.L]

                    future_values.append(future_value)

            if len(future_values) > 0:

                self.rules[cluster] = {
                    "values": future_values,
                    "prediction": np.mean(future_values)
                }

            else:

                self.rules[cluster] = {
                    "values": [],
                    "prediction": None
                }

        return self.rules

    # ---------------------------------------------------------
    # PREDICCIÓN DE UNA NUEVA OBSERVACIÓN
    # ---------------------------------------------------------

    def predict(self, X):
        """
        Predice X(N+1) utilizando los últimos L valores observados.
        """

        X = np.asarray(X, dtype=float)

        if len(X) < self.L:
            raise ValueError(
                "No hay suficientes observaciones para formar "
                "la última ventana."
            )

        # Última ventana observada
        last_window = X[-self.L:]

        # Convertirla en gránulo
        last_granule = self.extract_features(
            last_window.reshape(1, -1)
        )

        # Calcular memberships respecto a los clusters aprendidos
        memberships, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
            last_granule.T,
            self.centers,
            self.m,
            error=0.005,
            maxiter=1000
        )

        memberships = memberships[:, 0]

        # Guardamos memberships de la nueva observación
        self.last_memberships = memberships

        # -----------------------------------------------------
        # Promedio ponderado de las predicciones de los clusters
        # -----------------------------------------------------

        numerator = 0.0
        denominator = 0.0

        for cluster in range(self.c):

            prediction = self.rules[cluster]["prediction"]

            if prediction is not None:

                membership = memberships[cluster]

                numerator += membership * prediction
                denominator += membership

        if denominator == 0:
            raise ValueError(
                "No existen reglas de predicción disponibles."
            )

        prediction = numerator / denominator

        return prediction


# =============================================================
# EJEMPLO DE USO
# =============================================================

if __name__ == "__main__":

    # Serie de ejemplo
    X = np.array([
        13055, 13563, 13867, 14696,
        15460, 15311, 15603, 15861,
        16807, 16919, 16388, 15433,
        15497, 15145, 15163, 15984,
        16859, 18150, 18970, 19328,
        19337, 18876
    ])

    # Parámetros iniciales
    L = 5
    c = 4
    m = 2

    # Crear modelo
    model = FuzzyInformationGranules(
        window_size=L,
        n_clusters=c,
        m=m
    )

    # Entrenamiento
    model.fit(X)

    print("Centros de los clusters:")
    print(model.centers)

    print("\nGránulos:")
    print(model.granules)

    print("\nClusters dominantes:")
    print(model.labels)

    print("\nReglas:")
    for cluster, rule in model.rules.items():
        print(
            f"Cluster {cluster + 1}: "
            f"{rule['values']} -> "
            f"{rule['prediction']}"
        )

    # Predicción X(N+1)
    prediction = model.predict(X)

    print("\nMemberships de la última ventana:")
    print(model.last_memberships)

    print("\nPredicción X(N+1):")
    print(prediction)