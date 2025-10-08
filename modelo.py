import pandas as pd  
import matplotlib.pyplot as plt  
from statsmodels.tsa.api import VAR  
from statsmodels.tsa.stattools import adfuller, grangercausalitytests  
import numpy as np

# =============================================================================
# **1. Carga y preparación de datos para período 2020-2022**  
# =============================================================================

def load_data_2020_2022():
    """
    Carga datos específicos para el período 2020-2022
    """
    print("Cargando datos para período 2020-2022...")
    
    # Cargar datos CPB World 2020-2022
    cpb_2020_2022 = pd.read_csv('cpb_world_2020_2022.csv', parse_dates=[0], index_col=0)
    
    # Cargar datos del petróleo 2020-2022
    oil_2020 = pd.read_csv('DCOILWTICO-2020.csv', parse_dates=['observation_date'], index_col='observation_date')
    
    # Cargar datos del dólar 2020-2022
    dollar_2020 = pd.read_csv('DTWEXBGS-2020.csv', parse_dates=['observation_date'], index_col='observation_date')
    
    # Cargar datos de tasas de interés 2020-2022
    fed_2020 = pd.read_csv('FEDFUNDS-2020.csv', parse_dates=['observation_date'], index_col='observation_date')
    
    # Cargar datos de balance de la Fed 2020-2022
    walcl_2020 = pd.read_csv('WALCL-2020.csv', parse_dates=['observation_date'], index_col='observation_date')
    
    # Combinar todos los datasets
    df = cpb_2020_2022.join(oil_2020, how='inner')
    df = df.join(dollar_2020, how='inner')
    df = df.join(fed_2020, how='inner')
    df = df.join(walcl_2020, how='inner')
    
    # Renombrar columnas
    df.columns = [
        'World_Trade', 'World_Imports', 'World_IP',
        'Oil_Price', 'Dollar_Index', 'Fed_Funds_Rate', 'Fed_Balance_Sheet'
    ]
    
    # Filtrar solo el período 2020-2022
    df_period = df['2020-01-01':'2022-03-31'].dropna()
    
    print(f"Datos cargados: {df_period.shape[0]} observaciones (2020-2022)")
    print(f"Período: {df_period.index.min()} a {df_period.index.max()}")
    return df_period

# Cargar datos
df_2020 = load_data_2020_2022()

# =============================================================================
# **2. Visualización inicial de las series 2020-2022**  
# =============================================================================
plt.figure(figsize=(12, 10))
for i, col in enumerate(df_2020.columns, 1):
    plt.subplot(4, 2, i)
    plt.plot(df_2020.index, df_2020[col])
    plt.title(f'{col} - Período 2020-2022')
    plt.grid(True)
    plt.xticks(rotation=45)
plt.tight_layout()
plt.suptitle('Series Originales - Período Pandemia 2020-2022', y=1.02)
plt.show()

# =============================================================================
# **3. Prueba de estacionariedad (Dickey-Fuller Aumentado - ADF)**  
# =============================================================================
def adf_test(series, title=''):  
    result = adfuller(series.dropna())  
    print(f'ADF Test para {title}')  
    print(f'  Estadístico: {result[0]:.4f}')  
    print(f'  p-valor: {result[1]:.4f}')  
    print('  => Estacionaria' if result[1] < 0.05 else '  => NO estacionaria')  
    print()  

print("PRUEBAS DE ESTACIONARIEDAD - PERÍODO 2020-2022:")
print("="*50)
for column in df_2020.columns:
    adf_test(df_2020[column], column)

# =============================================================================
# **4. Transformación a estacionariedad (diferenciación)**  
# =============================================================================
df_2020_diff = df_2020.diff().dropna()  

# =============================================================================
# **5. Verificación de estacionariedad post-diferenciación**  
# =============================================================================
print("\nVERIFICACIÓN DESPUÉS DE DIFERENCIAR - 2020-2022:")
print("="*50)
for column in df_2020_diff.columns:
    adf_test(df_2020_diff[column], f'{column} (diferenciada)')

# =============================================================================
# **6. Visualización de series diferenciadas 2020-2022**  
# =============================================================================
plt.figure(figsize=(12, 10))
for i, col in enumerate(df_2020_diff.columns, 1):
    plt.subplot(4, 2, i)
    plt.plot(df_2020_diff.index, df_2020_diff[col])
    plt.title(f'{col} (Δ) - Período 2020-2022')
    plt.grid(True)
    plt.xticks(rotation=45)
plt.tight_layout()
plt.suptitle('Series Diferenciadas - Período Pandemia 2020-2022', y=1.02)
plt.show()

# =============================================================================
# **7. Modelado VAR para período 2020-2022
# =============================================================================
print("\nMODELADO VAR - PERÍODO 2020-2022:")
print("="*50)

n_obs = len(df_2020_diff)
n_vars = len(df_2020_diff.columns)
max_possible_lags = (n_obs - 1) // (n_vars + 1)  # Fórmula conservadora

print(f"Observaciones disponibles: {n_obs}")
print(f"Número de variables: {n_vars}")
print(f"Máximo lags teóricamente posible: {max_possible_lags}")


max_lags_to_try = min(3, max_possible_lags) 
print(f"Probando con máximo: {max_lags_to_try} lags")

try:
    model_2020 = VAR(df_2020_diff)  
    results_2020 = model_2020.fit(maxlags=max_lags_to_try, ic='aic')  
    print(f"Rezagos seleccionados: {results_2020.k_ar}")
    print(results_2020.summary())  
    
except Exception as e:
    print(f"Error al ajustar el modelo: {e}")
    print("Probando con modelo más simple...")

    key_vars = ['World_Trade', 'World_IP', 'Fed_Funds_Rate', 'Fed_Balance_Sheet']
    df_2020_reduced = df_2020_diff[key_vars]
    
    model_2020_simple = VAR(df_2020_reduced)
    max_lags_simple = min(2, (len(df_2020_reduced) - 1) // (len(df_2020_reduced.columns) + 1))
    results_2020 = model_2020_simple.fit(maxlags=max_lags_simple, ic='aic')
    print(f"Modelo reducido - Rezagos seleccionados: {results_2020.k_ar}")
    print(results_2020.summary())

# =============================================================================
# **8. Función de Impulso-Respuesta (IRF) 2020-2022
# =============================================================================
print("\nFUNCIÓN DE IMPULSO-RESPUESTA - 2020-2022:")
print("="*50)

try:
    irf_2020 = results_2020.irf(8)  
    irf_2020.plot(orth=False, figsize=(15, 12))  
    plt.suptitle("Función de Impulso-Respuesta - Período Pandemia 2020-2022")  
    plt.tight_layout()  
    plt.show()  
except Exception as e:
    print(f"Error al calcular IRF: {e}")

# =============================================================================
# **9. Causalidad de Granger 2020-2022 - AJUSTADO
# =============================================================================
print("\nTESTS DE CAUSALIDAD DE GRANGER - 2020-2022:")
print("="*50)

def perform_granger_tests_small_sample(data, maxlag=2): 
    variables = data.columns.tolist()
    significant_causality = []
    
    for i, cause_var in enumerate(variables):
        for j, effect_var in enumerate(variables):
            if cause_var != effect_var:
                print(f"\n{cause_var} → {effect_var}:")
                print("-" * 30)
                try:
                    test_data = data[[effect_var, cause_var]]
                    granger_result = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)
                    
                    p_values = []
                    for lag in range(1, maxlag + 1):
                        p_value = granger_result[lag][0]['ssr_ftest'][1]
                        p_values.append(p_value)
                        significance = '***' if p_value < 0.01 else '**' if p_value < 0.05 else '*' if p_value < 0.1 else ''
                        print(f"  Lag {lag}: p-value = {p_value:.4f} {significance}")
                    
                    min_p_value = min(p_values)
                    if min_p_value < 0.1: 
                        causality_strength = "FUERTE" if min_p_value < 0.05 else "MODERADA"
                        print(f"  => EVIDENCIA {causality_strength} de causalidad de Granger (p-min: {min_p_value:.4f})")
                        significant_causality.append(f"{cause_var} → {effect_var} (p-min: {min_p_value:.4f})")
                    else:
                        print(f"  => SIN evidencia de causalidad de Granger")
                        
                except Exception as e:
                    print(f"  Error en el test: {e}")
    
    return significant_causality


if 'df_2020_reduced' in locals():
    data_for_granger = df_2020_reduced
else:
    data_for_granger = df_2020_diff

significant_2020 = perform_granger_tests_small_sample(data_for_granger, maxlag=2)

# =============================================================================
# **10. Análisis de Descomposición de Varianza 2020-2022
# =============================================================================
print("\nDESCOMPOSICIÓN DE LA VARIANZA - 2020-2022:")
print("="*50)

try:
    fevd_2020 = results_2020.fevd(6)
    fevd_summary_2020 = fevd_2020.summary()
    print(fevd_summary_2020)

    fevd_2020.plot(figsize=(12, 10))
    plt.suptitle('Descomposición de Varianza - Período Pandemia 2020-2022')
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Error en descomposición de varianza: {e}")


