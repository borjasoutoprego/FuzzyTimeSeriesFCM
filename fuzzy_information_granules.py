import numpy as np
import skfuzzy as fuzz


class FuzzyInformationGranules:

    def __init__(self, window_size, n_clusters, m=2, random_state=None):
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
        self.random_state = random_state

        self.granules = None
        self.centers = None
        self.memberships = None
        self.rules = None

    def _validate_parameters(self):
        if not isinstance(self.L, (int, np.integer)) or self.L < 2:
            raise ValueError("window_size must be an integer greater than or equal to 2.")
        if not isinstance(self.c, (int, np.integer)) or self.c < 2:
            raise ValueError("n_clusters must be an integer greater than or equal to 2.")
        if not np.isfinite(self.m) or self.m <= 1:
            raise ValueError("m must be finite and greater than 1.")

    @staticmethod
    def _validate_series(X, name="X"):
        values = np.asarray(X, dtype=float)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be a one-dimensional series of finite values.")
        return values

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

        X = self._validate_series(X)

        if len(X) <= self.L:
            raise ValueError(
                "La serie debe tener más observaciones que la "
                "longitud de la ventana."
            )

        windows = np.array([X[i:i + self.L] for i in range(len(X) - self.L)])

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

        windows = np.asarray(windows, dtype=float)
        if windows.ndim != 2 or windows.shape[1] != self.L:
            raise ValueError("windows must have shape (n_windows, window_size).")
        if not np.all(np.isfinite(windows)):
            raise ValueError("windows must contain only finite values.")
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

        self._validate_parameters()
        X = self._validate_series(X)
        if X.size <= self.L:
            raise ValueError("The series must contain more values than window_size.")

        # Crear ventanas
        windows = self.create_windows(X)

        # Obtener gránulos
        granules = self.extract_features(windows)

        if windows.shape[0] < self.c:
            raise ValueError("Not enough target-known windows for n_clusters.")
        self.windows = windows
        self.granules = granules
        self.targets_ = X[self.L:].copy()

        # FCM necesita:
        # filas = características
        # columnas = observaciones
        data = granules.T

        seed = None
        if self.random_state is not None:
            seed = int(np.random.default_rng(self.random_state).integers(0, 2**31 - 1))
        centers, memberships, _, _, _, _, _ = fuzz.cluster.cmeans(
            data,
            c=self.c,
            m=self.m,
            error=0.005,
            maxiter=1000,
            seed=seed,
        )

        self.centers = centers
        self.memberships = memberships

        # Cluster dominante de cada gránulo
        self.labels = np.argmax(memberships, axis=0)

        # Rules use only target-known windows, never the final inference window.
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

        X = self._validate_series(X)

        self.rules = {}

        # La última ventana no tiene observación futura
        # conocida dentro de la muestra.
        n_training_windows = len(self.windows)

        for cluster in range(self.c):

            future_values = []

            for j in range(n_training_windows):

                if self.labels[j] == cluster:

                    future_value = self.targets_[j]

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

        if self.centers is None or self.rules is None:
            raise RuntimeError("Fit the model before predicting.")
        X = self._validate_series(X)

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
