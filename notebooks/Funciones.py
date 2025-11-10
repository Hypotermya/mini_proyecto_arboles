import matplotlib.pyplot as plt
import time
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report,roc_curve,auc
def crear_pipeline_preprocesamiento(modelo,
                                    usar_smote=False,
                                    escalar=True):
    # Detectar columnas automáticamente
    def seleccionar_columnas(X):
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        return num_cols, cat_cols

    # Construcción del pipeline cuando ya se conocen las columnas
    def construir_pipeline(X):
        num_cols, cat_cols = seleccionar_columnas(X)

        # Pipeline para variables numéricas: Imputación + Escalado (opcional)
        pasos_num = []
        pasos_num.append(('imputer', SimpleImputer(strategy='median')))
        if escalar:
            pasos_num.append(('scaler', StandardScaler()))

        transformer_num = Pipeline(steps=pasos_num)

        # Pipeline para categóricas: Imputación + OneHotEncoder
        transformer_cat = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocesador = ColumnTransformer(
            transformers=[
                ('num', transformer_num, num_cols),
                ('cat', transformer_cat, cat_cols)
            ]
        )

        # Construcción final con o sin SMOTE
        if usar_smote:
            pipeline = ImbPipeline(steps=[
                ('preprocesamiento', preprocesador),
                ('smote', SMOTE()),
                ('modelo', modelo)
            ])
        else:
            pipeline = Pipeline(steps=[
                ('preprocesamiento', preprocesador),
                ('modelo', modelo)
            ])

        return pipeline

    return construir_pipeline
def evaluar_modelo_clasificacion(modelo, X_test, y_test, clase_positiva=1):
    # --- Predicciones ---
    y_pred = modelo.predict(X_test)
    
    # --- Cálculo de métricas ---
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred,average='weighted',pos_label=clase_positiva, zero_division=0)
    recall = recall_score(y_test, y_pred,average='weighted',pos_label=clase_positiva, zero_division=0)
    f1 = f1_score(y_test, y_pred,average='weighted',pos_label=clase_positiva, zero_division=0)
    
    # --- Reporte ---
    print("\n Reporte de Clasificación:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # --- Matriz de Confusión ---
    etiquetas = sorted(list(set(y_test)))
    matriz_conf = confusion_matrix(y_test, y_pred, labels=etiquetas)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(matriz_conf, annot=True, fmt='d', cmap='Blues',
                xticklabels=etiquetas, yticklabels=etiquetas)
    plt.title("Matriz de Confusión (Heatmap)")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.show()

    # --- Curva ROC ---
    if hasattr(modelo, "predict_proba"):
        # Detectar correctamente el índice de la clase positiva
        pos_index = np.where(modelo.classes_ == clase_positiva)[0][0]
        y_pred_proba = modelo.predict_proba(X_test)[:, pos_index]
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba, pos_label=clase_positiva)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.title("Curva ROC - Clasificación Binaria")
        plt.xlabel("Tasa de Falsos Positivos (FPR)")
        plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.show()
    else:
        print("\n El modelo no soporta 'predict_proba', no se puede graficar la curva ROC.")
    
    # --- Retornar métricas ---
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    }
def realizar_gridsearch(modelo, param_grid, X_train, y_train, 
                        scoring, cv=5):
    inicio = time.time()
    
    print(f"\n Iniciando GridSearch para: {modelo.__class__.__name__} ...")
    grid_search = GridSearchCV(modelo, param_grid=param_grid, scoring=scoring, 
                               cv=cv)
    grid_search.fit(X_train, y_train)
    
    duracion = time.time() - inicio
    print(f" GridSearch finalizado en {duracion:.2f} segundos")
    print(f"  Mejor score ({scoring}): {grid_search.best_score_:.4f}")
    print(f"  Mejores parámetros: {grid_search.best_params_}\n")

    return {
        'best_model': grid_search.best_estimator_,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_
    }
def obtener_importancias_y_features(pipeline_entrenado):
    # Si tiene SMOTE, entra al pipeline interno
    if 'smote' in pipeline_entrenado.named_steps:
        modelo = pipeline_entrenado.named_steps['modelo']
        preprocesador = pipeline_entrenado.named_steps['preprocesamiento']
    else:
        modelo = pipeline_entrenado.named_steps['modelo']
        preprocesador = pipeline_entrenado.named_steps['preprocesamiento']

    # Obtener nombres de features procesadas
    try:
        feature_names = preprocesador.get_feature_names_out()
    except:
        # Si falla, intenta construirlos manualmente
        feature_names = []
        if hasattr(preprocesador, 'transformers_'):
            for name, trans, cols in preprocesador.transformers_:
                if hasattr(trans, 'get_feature_names_out'):
                    nombres = trans.get_feature_names_out(cols)
                else:
                    nombres = cols
                feature_names.extend(nombres)

    # Obtener importancias según el tipo de modelo
    if hasattr(modelo, 'get_feature_importance'):
        importancias = modelo.get_feature_importance()
    elif hasattr(modelo, 'feature_importances_'):
        importancias = modelo.feature_importances_
    else:
        raise AttributeError("El modelo no tiene atributo de importancia disponible.")

    return modelo, importancias, feature_names
def mostrar_importancias(modelo_nombre, importancias, features, top_n=15):
    df = pd.DataFrame({
        'Característica': features,
        'Importancia': importancias
    }).sort_values(by='Importancia', ascending=False)

    print(f"\n{'='*60}")
    print(f"🔹 Top {top_n} características más importantes - {modelo_nombre}")
    print(f"{'='*60}")
    print(df.head(top_n).to_string(index=False))

    # Gráfico
    plt.figure(figsize=(14, 10))
    plt.barh(df['Característica'][:top_n][::-1],
             df['Importancia'][:top_n][::-1])
    plt.title(f'Top {top_n} características más importantes - {modelo_nombre}')
    plt.xlabel('Importancia')
    plt.ylabel('Característica')
    plt.tight_layout()
    plt.show()