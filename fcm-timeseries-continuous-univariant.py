from pyexpat import model

import numpy as np
import skfuzzy as fuzz
from sklearn.neural_network import MLPClassifier


class FuzzyTimeSeriesFCM:
    
    def __init__(self, n_clusters, m=2):
        self.c = n_clusters
        self.m = m  # fuzziness
        self.centers = None
        self.u = None  # matriz de memberships
        self.labels = None  # serie fuzzy
    
    def fit_fcm(self, X):
        """
        X: array 1D (serie temporal)
        """
        X = np.array(X).reshape(1, -1)  # formato requerido
        
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            X, 
            c=self.c, 
            m=self.m, 
            error=0.005, 
            maxiter=1000
        )
        
        self.centers = cntr.flatten()
        self.u = u

    def fuzzify(self):
        # Ordenar centroides
        order = np.argsort(self.centers)
        self.centers = self.centers[order]
        self.u = self.u[order, :]
        
        # Obtener etiquetas (1,...,c)
        self.labels = np.argmax(self.u, axis=0) + 1
        
        return self.labels
    
    def build_rules(self):
        # XF(t-1) → XF(t)
        self.rules = {i: set() for i in range(1, self.c + 1)}
        
        for t in range(1, len(self.labels)):
            prev_label = self.labels[t-1]
            curr_label = self.labels[t]
            
            self.rules[prev_label].add(curr_label)
        
        return self.rules
    
    def predict_cheng(self, last_label):
    
        next_labels = self.rules.get(last_label, set())
        
        # Caso 1: conjunto vacío
        if len(next_labels) == 0:
            return self.centers[last_label - 1]
        
        # Caso 2: solo una etiqueta
        elif len(next_labels) == 1:
            label = list(next_labels)[0]
            return self.centers[label - 1]
        
        # Caso 3: varias etiquetas → promedio
        else:
            centers = [self.centers[l - 1] for l in next_labels]
            return np.mean(centers)
        
    def train_nn(self):
    
        X = self.labels[:-1].reshape(-1, 1)
        y = self.labels[1:]
        
        self.nn = MLPClassifier(
            hidden_layer_sizes=(10,),
            activation='logistic',
            solver='lbfgs',
            max_iter=1000
        )
        
        self.nn.fit(X, y)

    def predict_nn(self, last_label):
    
        pred_label = self.nn.predict([[last_label]])[0]
        
        # Defuzzificación: centro
        return self.centers[pred_label - 1]
    

def grid_search_fcm(
        X,
        c_values=range(2, 8),
        m_values=[1.5, 1.8, 2.0, 2.2, 2.5],
        method="cheng"  # o "nn"
    ):
    """Grid search para optimizar c y m"""
    
    results = []
    
    for c in c_values:
        for m in m_values:
            
            model = FuzzyTimeSeriesFCM(n_clusters=c, m=m)
            
            # Paso 1 y 2
            model.fit_fcm(X)
            labels = model.fuzzify()
            
            # Train/test simple (1-step ahead)
            preds = []
            real = X[1:]
            
            if method == "cheng":
                model.build_rules()
                
                for t in range(len(labels) - 1):
                    pred = model.predict_cheng(labels[t])
                    preds.append(pred)
            
            elif method == "nn":
                model.train_nn()
                
                for t in range(len(labels) - 1):
                    pred = model.predict_nn(labels[t])
                    preds.append(pred)
            
            # Error (MSE)
            preds = np.array(preds)
            real = np.array(real)
            
            mse = np.mean((real - preds) ** 2)
            
            results.append({
                "c": c,
                "m": m,
                "mse": mse
            })
    
    # Mejor combinación
    best = min(results, key=lambda x: x["mse"])
    
    return best, results
    
# Serie ejemplo
X = [13055,13563,13867,14696,15460,15311,15603]

best_cheng, _ = grid_search_fcm(X, method="cheng")
best_nn, _ = grid_search_fcm(X, method="nn")

print("Selección de parámetros:")
print("Cheng:", best_cheng) # {'c': 7, 'm': 2.0, 'mse': np.float64(0.0)}
print("NN:", best_nn) # {'c': 7, 'm': 2.2, 'mse': np.float64(0.0)}

# Con los mejores parámetros, entrenamos los modelos finales

# Cheng
model_cheng = FuzzyTimeSeriesFCM(n_clusters=best_cheng["c"])
model_cheng.m = best_cheng["m"]

model_cheng.fit_fcm(X)
labels = model_cheng.fuzzify()

print("Centros:", model_cheng.centers)
print("Labels:", labels)

model_cheng.build_rules()
pred_cheng = model_cheng.predict_cheng(labels[-1])
print("Predicción Cheng:", pred_cheng)

# Egrioglu
model_egrioglu = FuzzyTimeSeriesFCM(n_clusters=best_nn["c"])
model_egrioglu.m = best_nn["m"]

model_egrioglu.fit_fcm(X)
labels = model_egrioglu.fuzzify()

print("Centros:", model_egrioglu.centers)
print("Labels:", labels)

model_egrioglu.train_nn()
pred_egrioglu = model_egrioglu.predict_nn(labels[-1])
print("Predicción Egrioglu:", pred_egrioglu)