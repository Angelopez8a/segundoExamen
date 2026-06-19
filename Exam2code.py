# %% [markdown]
# # Examen 2 — Detección de Fraude en Transacciones con Tarjeta
# **Rolando Oviedo Quezada**
#
# Objetivo: identificar transacciones fraudulentas sobre el dataset de IBM (~24M filas).
# Se trabaja con las 3 tablas: Clientes, Tarjetas y Transacciones.
# %%
import pandas as pd
import numpy as np
import gc
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score, precision_recall_curve,
    classification_report, PrecisionRecallDisplay
)
warnings.filterwarnings('ignore')

# ── Configuracion ──────────────────────────────────────────────────────────────
# NROWS = None  → lee toda la base (24.3M filas, recomendado ≥16 GB RAM)
# NROWS = 15_000_000  → version aligerada si la RAM es limitada
NROWS    = None
RUTA     = 'data/'
SALIDA   = './'
RANDOM   = 42
TARGET   = 'Is Fraud?'
# ───────────────────────────────────────────────────────────────────────────────
print('Configuracion lista.')
# %% [markdown]
# ## 1. Carga de datos
# %%
df_users = pd.read_csv(RUTA + 'sd254_users.csv')
df_cards = pd.read_csv(RUTA + 'sd254_cards.csv')

df_trans = pd.read_csv(
    RUTA + 'credit_card_transactions-ibm_v2.csv',
    nrows=1_000_000,
    dtype={
        'User':  'int32',
        'Card':  'int32',
        'Year':  'int16',
        'Month': 'int8',
        'Day':   'int8',
        'MCC':   'int32',
    }
)

print('Shapes:', df_users.shape, df_cards.shape, df_trans.shape)
print(df_trans[TARGET].value_counts())
print('Fraude:', round(df_trans[TARGET].eq('Yes').mean() * 100, 3), '%')
# %% [markdown]
# ## 2. Limpieza de tablas
# %%
# ── Funcion reutilizable para columnas con $ ────────────────────────────────
def limpiar_dinero(col):
    return col.str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)

# ── Tabla Usuarios ─────────────────────────────────────────────────────────
df_users['Per Capita Income - Zipcode'] = limpiar_dinero(df_users['Per Capita Income - Zipcode'])
df_users['Yearly Income - Person']      = limpiar_dinero(df_users['Yearly Income - Person'])
df_users['Total Debt']                  = limpiar_dinero(df_users['Total Debt'])
df_users['Apartment']                   = df_users['Apartment'].fillna('')
df_users.reset_index(inplace=True)
df_users.rename(columns={'index': 'User'}, inplace=True)

# ── Tabla Tarjetas ─────────────────────────────────────────────────────────
df_cards['Credit Limit']       = limpiar_dinero(df_cards['Credit Limit'])
df_cards['Has Chip']           = (df_cards['Has Chip'].str.upper() == 'YES').astype(int)
df_cards['Card on Dark Web']   = (df_cards['Card on Dark Web'].str.strip() == 'Yes').astype(int)
df_cards['Expires']            = pd.to_datetime(df_cards['Expires'],        format='%m/%Y')
df_cards['Acct Open Date']     = pd.to_datetime(df_cards['Acct Open Date'], format='%m/%Y')
df_cards.rename(columns={'CARD INDEX': 'CARD_INDEX'}, inplace=True)

# ── Tabla Transacciones ────────────────────────────────────────────────────
df_trans['Amount']   = df_trans['Amount'].str.replace('$', '', regex=False).astype(float).astype('float32')
df_trans[TARGET]     = (df_trans[TARGET] == 'Yes').astype('int8')
df_trans['Errors?']  = df_trans['Errors?'].fillna('No Error')
df_trans['Zip']      = df_trans['Zip'].fillna(0).astype(int).astype(str)
df_trans.loc[df_trans['Zip'] == '0', 'Zip'] = ''

print('Usuarios:', df_users.shape)
print('Tarjetas:', df_cards.shape)
print('Transacciones:', df_trans.shape)
print('Fraudes:', df_trans[TARGET].sum(), '/', len(df_trans))
# %% [markdown]
# ## 3. Merge de las 3 tablas
# %%
df = df_trans.merge(
    df_cards,
    left_on=['User', 'Card'],
    right_on=['User', 'CARD_INDEX'],
    how='left'
).merge(
    df_users,
    on='User',
    how='left'
)

del df_trans, df_cards, df_users
gc.collect()

# Columnas que nunca usaremos (IDs, texto libre, PII)
# Apartment se conserva aqui para calcular tiene_depto mas adelante
cols_inutiles = ['Person', 'Address', 'Card Number', 'CVV',
                 'Merchant Name', 'CARD_INDEX', 'Card on Dark Web']
df.drop(columns=[c for c in cols_inutiles if c in df.columns], inplace=True)

# Strings con pocos valores unicos → category ahorra hasta 90% de RAM
for col in ['Use Chip', 'Merchant State', 'Merchant City', 'Gender',
            'City', 'State', 'Card Brand', 'Card Type', 'Errors?']:
    if col in df.columns:
        df[col] = df[col].astype('category')

print(f'Shape: {df.shape}')
print(f'Memoria: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB')
# %% [markdown]
# ## 4. Feature Engineering
#
# Se crean **todas** las variables que tienen los notebooks del equipo. Después, en la sección de validación (lift), se identifican las que no discriminan fraude y se eliminan antes del modelado.
# %%
# ── 4.1 Monto ──────────────────────────────────────────────────────────────
df['abs_amount'] = df['Amount'].abs().astype('float32')

# Reembolso (monto negativo)
df['is_refund'] = (df['Amount'] < 0).astype('int8')

# Patrones de monto sospechosos
df['monto_redondo']      = (df['Amount'] % 1 == 0).astype('int8')
df['monto_psicologico']  = ((df['Amount'] % 1) >= 0.95).astype('int8')
df['monto_multiplo_100'] = ((df['Amount'] % 100 == 0) & (df['Amount'] > 0)).astype('int8')
df['monto_multiplo_50']  = ((df['Amount'] % 50  == 0) & (df['Amount'] > 0)).astype('int8')

print('Features de monto OK')
df[['Amount', 'abs_amount', 'is_refund', 'monto_redondo', 'monto_psicologico',
    'monto_multiplo_100', 'monto_multiplo_50']].describe()
# %%
# ── 4.2 Tiempo ─────────────────────────────────────────────────────────────
df['hour']   = df['Time'].str.split(':').str[0].astype('int8')
df['minute'] = df['Time'].str.split(':').str[1].astype('int8')

df['is_madrugada']    = (df['hour'].between(0, 5)).astype('int8')
df['night_transaction']= ((df['hour'] >= 22) | (df['hour'] <= 5)).astype('int8')
df['working_hours']   = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype('int8')

# Fecha para calcular edad de tarjeta y dia de la semana
df['fecha'] = pd.to_datetime(
    df[['Year', 'Month', 'Day']].rename(columns={'Year': 'year', 'Month': 'month', 'Day': 'day'})
)
df['dia_semana']     = df['fecha'].dt.dayofweek.astype('int8')   # lunes=0, domingo=6
df['is_weekend']     = (df['dia_semana'] >= 5).astype('int8')
df['hora_de_semana'] = (df['hour'] + df['dia_semana'] * 24).astype('int16')  # 0-167
df['es_q4']          = df['Month'].isin([10, 11, 12]).astype('int8')

df['turno_dia'] = pd.cut(
    df['hour'],
    bins=[-1, 5, 11, 17, 23],
    labels=['madrugada', 'manana', 'tarde', 'noche']
)

print('Features de tiempo OK')
df[['hour', 'is_madrugada', 'night_transaction', 'working_hours',
    'is_weekend', 'es_q4', 'dia_semana']].head()
# %%
# ── 4.3 Geografia ──────────────────────────────────────────────────────────
df['different_state'] = (df['Merchant State'].astype(str) != df['State'].astype(str)).astype('int8')
df['different_city']  = (df['Merchant City'].astype(str)  != df['City'].astype(str)).astype('int8')

# Compras online (el Merchant City dice literalmente ONLINE)
df['is_online'] = (df['Merchant City'].astype(str).str.upper() == 'ONLINE').astype('int8')

# Fuera del estado del cliente (comercio con codigo de estado valido de 2 letras)
df['is_out_of_state'] = (
    (df['Merchant State'].astype(str).str.len() == 2) &
    (df['Merchant State'].astype(str) != df['State'].astype(str))
).astype('int8')

estados_usa = {
    'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA',
    'KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
    'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT',
    'VA','WA','WV','WI','WY','DC'
}
df['is_extranjero'] = (
    ~df['Merchant State'].astype(str).isin(estados_usa) & df['Merchant State'].notna()
).astype('int8')

print('Features geograficas OK')
df[['different_state','different_city','is_online','is_out_of_state','is_extranjero']].value_counts().head(8)
# %%
# ── 4.4 Tipo de transaccion y errores ──────────────────────────────────────
df['swipe_transaction']  = (df['Use Chip'] == 'Swipe Transaction').astype('int8')
df['chip_transaction']   = (df['Use Chip'] == 'Chip Transaction').astype('int8')
df['online_transaction'] = (df['Use Chip'] == 'Online Transaction').astype('int8')

df['has_error']        = (df['Errors?'] != 'No Error').astype('int8')
df['auth_error']       = df['Errors?'].str.contains('Bad PIN',              na=False).astype('int8')
df['financial_error']  = df['Errors?'].str.contains('Insufficient Balance', na=False).astype('int8')
df['error_cvv']        = df['Errors?'].str.contains('Bad CVV',              na=False).astype('int8')
df['error_expiracion'] = df['Errors?'].str.contains('Bad Expiration',       na=False).astype('int8')
df['technical_error']  = df['Errors?'].str.contains('Technical Glitch',     na=False).astype('int8')

print('Features de transaccion/errores OK')
df[['has_error','auth_error','financial_error','error_cvv',
    'error_expiracion','technical_error']].sum()
# %%
# ── 4.5 Caracteristicas de tarjeta ─────────────────────────────────────────
df['has_chip'] = df['Has Chip'].astype('int8')
df.drop(columns=['Has Chip'], inplace=True)

# Tarjeta reemitida (sustituye a una anterior, puede indicar robo previo)
df['card_reissued'] = (df['Cards Issued'] > 1).astype('int8')

df['amount_credit_ratio'] = (
    df['Amount'] / df['Credit Limit'].replace(0, np.nan)
).fillna(0).clip(0, 5).astype('float32')

df['cerca_del_limite'] = (df['amount_credit_ratio'] > 0.8).astype('int8')

df['saldo_disponible_ratio'] = (
    (df['Credit Limit'] - df['Amount']) / df['Credit Limit'].replace(0, np.nan)
).fillna(0).clip(-1, 1).astype('float32')

df['dias_para_vencer'] = (df['Expires'] - df['fecha']).dt.days.clip(lower=-365).astype('float32')
df['tarjeta_vencida']  = (df['dias_para_vencer'] < 0).astype('int8')

df['card_age_months'] = (
    (df['fecha'] - df['Acct Open Date']).dt.days / 30
).clip(lower=0).astype('float32')

df['card_age_years'] = (df['Year'] - df['Acct Open Date'].dt.year).clip(lower=0).astype('float32')

df['young_card'] = (df['card_age_years'] <= 1).astype('int8')

df['pin_age'] = (df['Year'] - df['Year PIN last Changed']).clip(lower=0).astype('float32')

df.drop(columns=['Expires', 'Acct Open Date'], inplace=True)
gc.collect()
print('Features de tarjeta OK')
# %%
# ── 4.6 Perfil financiero del cliente ──────────────────────────────────────
income = df['Yearly Income - Person'].replace(0, np.nan)
pc_inc = df['Per Capita Income - Zipcode'].replace(0, np.nan)
credit = df['Credit Limit'].replace(0, np.nan)

df['debt_income_ratio']         = (df['Total Debt'] / income).fillna(0).astype('float32')
df['years_to_retirement']       = (df['Retirement Age'] - df['Current Age']).clip(lower=0).astype('float32')
df['is_retired']                = (df['Current Age'] >= df['Retirement Age']).astype('int8')
df['ingreso_vs_zona']           = (df['Yearly Income - Person'] / pc_inc).fillna(1).astype('float32')
df['amount_daily_income_ratio'] = (df['Amount'] / (income / 365)).fillna(0).clip(0, 100).astype('float32')
df['amount_income_ratio']       = (df['Amount'] / income).fillna(0).clip(0, 1).astype('float32')
df['amount_to_monthly_income']  = (df['abs_amount'] / (income / 12)).fillna(0).clip(0, 100).astype('float32')
df['credit_limit_to_income']    = (df['Credit Limit'] / income).fillna(0).clip(0, 10).astype('float32')
df['debt_to_credit_limit']      = (df['Total Debt'] / credit).fillna(0).clip(0, 50).astype('float32')
df['low_fico']                  = (df['FICO Score'] < 650).astype('int8')
df['high_debt']                 = (df['debt_income_ratio'] > 2).astype('int8')
df['many_cards']                = (df['Num Credit Cards'] >= 5).astype('int8')

# Vive en edificio (zona urbana)
df['tiene_depto'] = (df['Apartment'].astype(str).str.strip() != '').astype('int8')
df.drop(columns=['Apartment'], inplace=True)

# FICO en categorias
df['fico_tier'] = pd.cut(
    df['FICO Score'],
    bins=[0, 579, 669, 739, 799, 850],
    labels=['malo', 'regular', 'bueno', 'muy_bueno', 'excelente']
)

gc.collect()
print('Features de perfil financiero OK')
df[['debt_income_ratio', 'ingreso_vs_zona', 'low_fico', 'high_debt',
    'many_cards', 'is_retired', 'tiene_depto']].describe()
# %% [markdown]
# ### 4.7 Features históricas (sin fuga de información)
#
# Para cada transacción se computan estadísticas **solo con transacciones anteriores** del mismo usuario.
# Se usa `cumsum` vectorizado (O(N)) en vez de `groupby.apply` para soportar 24M filas eficientemente.
# %%
# Guardamos orden original para restaurarlo al final
df['_orden_original'] = np.arange(len(df))

# Ordenamos cronologicamente por usuario para calcular historicos correctamente
df = df.sort_values(['User', 'Year', 'Month', 'Day', 'hour', 'minute', 'Card']).reset_index(drop=True)

g_user = df.groupby('User', sort=False)

# Numero de transacciones previas del usuario
df['txns_prev_usuario'] = g_user.cumcount().astype('int32')

# ── Gasto historico (media, std, max) usando cumsum vectorizado ────────────
cum_abs    = g_user['abs_amount'].cumsum()
df['_abs_sq'] = df['abs_amount'] ** 2
cum_abs_sq = g_user['_abs_sq'].cumsum()

n_prev      = df['txns_prev_usuario'].replace(0, np.nan)
sum_prev    = cum_abs    - df['abs_amount']
sum_sq_prev = cum_abs_sq - df['_abs_sq']

df['gasto_promedio_usuario_hist'] = (sum_prev / n_prev).fillna(0).astype('float32')
var_hist = (sum_sq_prev / n_prev) - df['gasto_promedio_usuario_hist'] ** 2
df['gasto_std_usuario_hist']      = np.sqrt(var_hist.clip(lower=0)).fillna(0).astype('float32')
df['gasto_max_usuario_hist']      = (
    g_user['abs_amount'].cummax()
    .groupby(df['User']).shift(1).fillna(0).astype('float32')
)
df['z_score_monto_hist'] = (
    (df['abs_amount'] - df['gasto_promedio_usuario_hist']) /
    df['gasto_std_usuario_hist'].replace(0, np.nan)
).fillna(0).clip(-10, 10).astype('float32')

df['supera_maximo_historico'] = (
    (df['txns_prev_usuario'] > 0) & (df['abs_amount'] > df['gasto_max_usuario_hist'])
).astype('int8')

# ── Comportamiento online historico ────────────────────────────────────────
prev_online = g_user['online_transaction'].cumsum() - df['online_transaction']
df['online_ratio_usuario_hist'] = (prev_online / n_prev).fillna(0).astype('float32')

# Compra online inusual: usuario que casi nunca compra online (minimo 5 txns previas)
df['online_inusual'] = (
    (df['online_transaction'] == 1) &
    (df['txns_prev_usuario'] >= 5) &
    (df['online_ratio_usuario_hist'] < 0.05)
).astype('int8')

# ── Diversidad de MCC historica ────────────────────────────────────────────
primer_mcc_usuario = ~df.duplicated(['User', 'MCC'])
df['mcc_diversity_usuario_hist'] = (
    primer_mcc_usuario.astype('int16').groupby(df['User']).cumsum()
    .groupby(df['User']).shift(1).fillna(0).astype('int16')
)

df.drop(columns=['_abs_sq'], inplace=True)
gc.collect()

print('Features historicas de usuario OK')
df[['txns_prev_usuario','gasto_promedio_usuario_hist','z_score_monto_hist',
    'supera_maximo_historico','online_inusual','mcc_diversity_usuario_hist']].describe()
# %%
# ── 4.8 Velocidad (sin fuga: acumulado hasta la transaccion actual) ─────────
g_card_day = df.groupby(['User', 'Card', 'Year', 'Month', 'Day'], sort=False)

# Transacciones previas de la misma tarjeta en el mismo dia (sin incluir la actual)
df['txns_prev_mismo_dia'] = g_card_day.cumcount().astype('int16')
df['velocidad_alta']      = (df['txns_prev_mismo_dia'] > 5).astype('int8')

# Estados distintos visitados por la misma tarjeta en el dia HASTA esta transaccion
primer_estado_dia = ~df.duplicated(['User', 'Card', 'Year', 'Month', 'Day', 'Merchant State'])
keys_dia = [df['User'], df['Card'], df['Year'], df['Month'], df['Day']]
df['estados_distintos_dia_hasta_ahora'] = (
    primer_estado_dia.astype('int16').groupby(keys_dia).cumsum().astype('int16')
)
df['tarjeta_en_varios_estados'] = (df['estados_distintos_dia_hasta_ahora'] > 1).astype('int8')

gc.collect()
print('Features de velocidad OK')
df[['txns_prev_mismo_dia','velocidad_alta','estados_distintos_dia_hasta_ahora',
    'tarjeta_en_varios_estados']].describe()
# %%
# ── 4.9 Variables compuestas ────────────────────────────────────────────────
# triple_riesgo: madrugada + online + diferente estado → muy sospechoso
df['triple_riesgo'] = (
    df['is_madrugada'] & df['online_transaction'] & df['different_state']
).astype('int8')

# score_riesgo_manual: suma ponderada de señales individuales
df['score_riesgo_manual'] = (
    df['is_extranjero']    * 2 +
    df['different_state']  * 1 +
    df['is_madrugada']     * 1 +
    df['has_error']        * 2 +
    df['auth_error']       * 1 +
    df['error_cvv']        * 1 +
    df['cerca_del_limite'] * 1 +
    df['velocidad_alta']   * 2
).astype('int8')

# Eliminamos is_madrugada del set de features (lift ~1.15, solo se uso para compuestas)
# y columnas temporales que ya cumplieron su funcion
df.drop(columns=['fecha'], inplace=True)

# Restauramos orden original
df = df.sort_values('_orden_original').drop(columns=['_orden_original']).reset_index(drop=True)

gc.collect()
print(f'Shape tras FE completo: {df.shape}')
print(f'Memoria: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB')
# %% [markdown]
# ## 5. Validación — Lift de variables binarias
#
# El **lift** mide cuánto más frecuente es el fraude cuando la variable vale 1 vs cuando vale 0.
# Lift = 1 → la variable no discrimina. Lift > 1 → señal positiva de fraude. Lift < 1 → señal negativa.
# Se eliminarán del modelo las variables con lift ≤ 1.5.
# %%
vars_binarias_validar = [
    # tiempo
    'is_madrugada', 'night_transaction', 'working_hours', 'is_weekend', 'es_q4',
    # geografia
    'different_state', 'different_city', 'is_online', 'is_out_of_state', 'is_extranjero',
    # tipo de transaccion
    'swipe_transaction', 'chip_transaction', 'online_transaction',
    # errores
    'has_error', 'auth_error', 'financial_error', 'error_cvv', 'error_expiracion', 'technical_error',
    # tarjeta
    'has_chip', 'card_reissued', 'young_card', 'cerca_del_limite', 'tarjeta_vencida',
    # velocidad / patron
    'velocidad_alta', 'tarjeta_en_varios_estados', 'triple_riesgo',
    # monto
    'is_refund', 'monto_redondo', 'monto_psicologico', 'monto_multiplo_100', 'monto_multiplo_50',
    # comportamiento historico de usuario
    'online_inusual', 'supera_maximo_historico',
    # perfil del cliente
    'is_retired', 'low_fico', 'high_debt', 'many_cards', 'tiene_depto',
]

resultados = []
for var in vars_binarias_validar:
    if var not in df.columns:
        continue
    tasa_0 = df.loc[df[var] == 0, TARGET].mean()
    tasa_1 = df.loc[df[var] == 1, TARGET].mean()
    lift   = tasa_1 / tasa_0 if tasa_0 > 0 else 0
    resultados.append({'variable': var, 'tasa_si': round(tasa_1, 6),
                       'tasa_no': round(tasa_0, 6), 'lift': round(lift, 2)})

res = pd.DataFrame(resultados).sort_values('lift', ascending=False)
print(res.to_string(index=False))
# %%
colors = ['#e74c3c' if x > 1.5 else '#95a5a6' for x in res['lift']]
plt.figure(figsize=(9, 6))
plt.barh(res['variable'], res['lift'], color=colors)
plt.axvline(x=1.5, color='black', linestyle='--', linewidth=1, label='umbral 1.5')
plt.xlabel('Lift (tasa fraude con flag=1 / tasa fraude con flag=0)')
plt.title('Discriminacion de cada variable binaria')
plt.legend()
plt.tight_layout()
plt.show()
# %% [markdown]
# ## 6. Preparación para modelado
# %%
# Variables que NO entran al modelo
# Basado en el lift de la validacion anterior (consistente en los 3 notebooks del equipo)
cols_drop_modelo = [
    # IDs y texto operativo
    'Time', 'Errors?', 'Use Chip', 'User',

    # ── Binarias con lift <= 1.5 ──────────────────────────────────────────
    # Tiempo (lift ~0.68-1.18)
    'is_madrugada',      # 1.15 — solo sirvio para triple_riesgo y score_riesgo_manual
    'night_transaction', # 0.68
    'is_weekend',        # 1.06
    'es_q4',             # 1.07

    # Geografia (lift < 0.51)
    'is_out_of_state',   # 0.47-0.51 — redundante con different_state

    # Tipo de transaccion (lift < 0.57)
    'swipe_transaction', # 0.17-0.19
    'chip_transaction',  # 0.55-0.57
    'is_online',         # redundante con online_transaction (misma informacion)

    # Errores (lift < 1.43)
    'has_chip',          # 1.02-1.05
    'financial_error',   # 1.41-1.43
    'error_expiracion',  # lift bajo, similar a financial_error
    'technical_error',   # lift bajo

    # Tarjeta (lift < 0.93)
    'card_reissued',     # 1.01-1.04
    'young_card',        # 0.92-0.93
    'tarjeta_vencida',   # 0.45-0.56 — conservamos dias_para_vencer (continua)

    # Monto (lift < 1.05)
    'is_refund',         # 0.67-0.72
    'monto_redondo',     # 0.56-0.57
    'monto_psicologico', # 1.02-1.05
    'monto_multiplo_100',# 0.05-0.08
    'monto_multiplo_50', # 0.11-0.15

    # Perfil cliente (lift < 1.40)
    'is_retired',        # 1.29-1.40
    'low_fico',          # 0.86-0.90
    'high_debt',         # 0.82-0.91
    'many_cards',        # 1.46-1.53
    'tiene_depto',       # no validada en notebooks anteriores, sin señal clara

    # ── Redundantes con otras variables mas informativas ──────────────────
    'amount_income_ratio',      # = amount_daily_income_ratio / 365
    'amount_to_monthly_income', # = amount_daily_income_ratio / 30.4
    'card_age_years',           # = card_age_months / 12
    'fico_tier',                # version categorica de FICO Score (ya esta numerico)
    'turno_dia',                # capturado por working_hours + hour
]

cols_drop_modelo = [c for c in cols_drop_modelo if c in df.columns]
df_model = df.drop(columns=cols_drop_modelo)

print(f'Variables eliminadas: {len(cols_drop_modelo)}')
print(f'Shape final: {df_model.shape}')
print(f'Fraudes: {df_model[TARGET].sum():,}  ({df_model[TARGET].mean()*100:.3f}%)')
print(f'Memoria: {df_model.memory_usage(deep=True).sum() / 1e9:.2f} GB')
print('\nColumnas categoricas restantes:')
print(df_model.select_dtypes(include='category').columns.tolist())
# %%
# Guardamos en parquet para reiniciar desde el modelado sin repetir FE
df_model.to_parquet(RUTA + 'transactions_clean.parquet', index=False)
del df
gc.collect()
print('Guardado en parquet OK')
# %% [markdown]
# ## 7. Modelado
#
# **Metrica principal**: PR-AUC (Area bajo la curva Precision-Recall).
# Con ~0.12% de fraudes, Accuracy es engañosa (99.88% diciendo "todo legítimo").
# PR-AUC captura el tradeoff entre detectar fraudes y no alarmar transacciones legitimas.
#
# **Modelos del curriculo implementados**:
# - 2.1 Regresión Logística (modelo lineal base)
# - 2.4 Análisis Discriminante Lineal (LDA)
# - 2.8 Naive Bayes Gaussiano
# - 2.9 Árbol de Decisión
# - 2.11 Random Forest (ensamble)
# - 2.11 Gradient Boosting con histogramas (ensamble, más rápido para datos grandes)
# - 2.10 Red Neuronal (MLP)
#
# > **SVM (2.6) y KNN (2.7)** se omiten: su complejidad O(N²) los hace computacionalmente inviables para millones de filas.
# %%
import pandas as pd
import numpy as np
import gc

df_model = pd.read_parquet(RUTA + 'transactions_clean.parquet')

cat_cols = ['Merchant City', 'Merchant State', 'Card Brand', 'Card Type',
            'Gender', 'City', 'State']
for col in cat_cols:
    if col in df_model.columns:
        df_model[col] = df_model[col].astype('category')

print(df_model.shape)
print(df_model[TARGET].value_counts())
# %%
# ── Split temporal ─────────────────────────────────────────────────────────
# Entrenamos en el pasado y evaluamos en lo mas reciente.
# Esto simula el uso real: aprender con historia y predecir transacciones nuevas.
year_split = int(df_model['Year'].quantile(0.85))
print(f'Entrenando hasta {year_split}, evaluando {year_split+1} en adelante')

X = df_model.drop(columns=[TARGET])
y = df_model[TARGET]

train_mask = df_model['Year'] <= year_split
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

n_neg = int((y_train == 0).sum())
n_pos = int((y_train == 1).sum())

print(f'Train: {X_train.shape}  fraudes: {y_train.sum():,}')
print(f'Test:  {X_test.shape}   fraudes: {y_test.sum():,}')
# %%
# ── Preparar features para sklearn ────────────────────────────────────────
# Las columnas de alta cardinalidad (Merchant City, City, State, Merchant State, Zip)
# ya estan resumidas en las variables engineeradas (different_city, is_extranjero, etc.)
# Se dropean para evitar memoria excesiva con one-hot encoding.
# Las de baja cardinalidad (Card Brand, Card Type, Gender) se codifican con cat.codes.

cols_alta_card = ['Merchant City', 'Merchant State', 'City', 'State', 'Zip']
X_train_sk = X_train.drop(columns=[c for c in cols_alta_card if c in X_train.columns])
X_test_sk  = X_test.drop(columns=[c for c in cols_alta_card if c in X_test.columns])

for col in ['Card Brand', 'Card Type', 'Gender']:
    if col in X_train_sk.columns:
        X_train_sk[col] = X_train_sk[col].cat.codes
        X_test_sk[col]  = X_test_sk[col].cat.codes

# Por si queda alguna columna object
obj_cols = X_train_sk.select_dtypes(include='object').columns.tolist()
if obj_cols:
    X_train_sk = X_train_sk.drop(columns=obj_cols)
    X_test_sk  = X_test_sk.drop(columns=obj_cols)

print(f'Features para sklearn: {X_train_sk.shape[1]}')
print(X_train_sk.dtypes.value_counts())
# %%
# ── Funcion de evaluacion unificada ────────────────────────────────────────
from sklearn.metrics import average_precision_score, precision_recall_curve, classification_report

resultados_modelos = {}

def evaluar(nombre, y_true, y_proba):
    pr_auc = average_precision_score(y_true, y_proba)
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    best_t = thresholds[f1.argmax()]
    y_pred = (y_proba >= best_t).astype(int)
    print(f'\n=== {nombre} ===')
    print(f'PR-AUC: {pr_auc:.4f}  |  threshold optimo: {best_t:.4f}')
    print(classification_report(y_true, y_pred, digits=4))
    resultados_modelos[nombre] = {'pr_auc': pr_auc, 'threshold': best_t, 'y_proba': y_proba}
    return pr_auc, best_t, y_proba
# %% [markdown]
# ### Baseline: Dummy Classifier
# Con 0.12% de fraudes, un modelo aleatorio es el piso mínimo a superar.
# %%
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy='stratified', random_state=RANDOM)
dummy.fit(X_train_sk, y_train)
y_proba_dummy = dummy.predict_proba(X_test_sk)[:, 1]
pr_auc_dummy, _, _ = evaluar('Dummy (baseline)', y_test, y_proba_dummy)
# %% [markdown]
# ### 2.1 Regresión Logística
# Modelo lineal base. `class_weight='balanced'` ajusta el peso de los fraudes para compensar el desbalanceo. `solver='saga'` es el más eficiente para datasets grandes.
# %%
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_sk.fillna(0))
X_test_sc  = scaler.transform(X_test_sk.fillna(0))

lr = LogisticRegression(
    class_weight='balanced',
    solver='saga',
    max_iter=300,
    C=0.1,
    random_state=RANDOM
)
lr.fit(X_train_sc, y_train)

y_proba_lr = lr.predict_proba(X_test_sc)[:, 1]
pr_auc_lr, t_lr, _ = evaluar('Logistic Regression', y_test, y_proba_lr)
# %% [markdown]
# ### 2.4 Análisis Discriminante Lineal (LDA)
# Asume que las clases tienen la misma covarianza. `solver='lsqr'` con regularización de Ledoit-Wolf es estable y escalable para datasets grandes.
# %%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# priors balanceados para compensar el desbalanceo de clases
lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=[0.5, 0.5])
lda.fit(X_train_sc, y_train)

y_proba_lda = lda.predict_proba(X_test_sc)[:, 1]
pr_auc_lda, t_lda, _ = evaluar('LDA', y_test, y_proba_lda)
# %% [markdown]
# ### 2.8 Naive Bayes Gaussiano
# Asume independencia entre features y distribución Gaussiana. Muy rápido, sirve como referencia de modelo probabilístico simple.
# %%
from sklearn.naive_bayes import GaussianNB

# priors balanceados para compensar el desbalanceo
gnb = GaussianNB(priors=[0.5, 0.5])
gnb.fit(X_train_sc, y_train)

y_proba_gnb = gnb.predict_proba(X_test_sc)[:, 1]
pr_auc_gnb, t_gnb, _ = evaluar('Naive Bayes', y_test, y_proba_gnb)
# %% [markdown]
# ### 2.9 Árbol de Decisión
# Reglas interpretables. `max_depth=12` evita que memorice el entrenamiento. `class_weight='balanced'` ajusta los pesos.
# %%
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(
    max_depth=12,
    min_samples_leaf=50,
    class_weight='balanced',
    random_state=RANDOM
)
dt.fit(X_train_sk.fillna(-1), y_train)

y_proba_dt = dt.predict_proba(X_test_sk.fillna(-1))[:, 1]
pr_auc_dt, t_dt, _ = evaluar('Decision Tree', y_test, y_proba_dt)
# %% [markdown]
# ### 2.11 Random Forest
# Ensamble de árboles independientes. `max_samples=0.5` reduce RAM y tiempo sin perder rendimiento. `balanced_subsample` ajusta pesos por árbol.
# %%
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=50,
    max_samples=0.5,
    class_weight='balanced_subsample',
    n_jobs=-1,
    random_state=RANDOM
)
rf.fit(X_train_sk.fillna(-1), y_train)

y_proba_rf = rf.predict_proba(X_test_sk.fillna(-1))[:, 1]
pr_auc_rf, t_rf, _ = evaluar('Random Forest', y_test, y_proba_rf)
# %% [markdown]
# ### 2.11 Gradient Boosting con Histogramas
# Ensamble secuencial optimizado para datos grandes. `HistGradientBoostingClassifier` maneja NaN nativamente y usa histogramas (igual que LightGBM). Early stopping evita sobreajuste.
# %%
from sklearn.ensemble import HistGradientBoostingClassifier

hgb = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.05,
    max_depth=8,
    min_samples_leaf=50,
    class_weight='balanced',
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=RANDOM
)
hgb.fit(X_train_sk, y_train)

y_proba_hgb = hgb.predict_proba(X_test_sk)[:, 1]
pr_auc_hgb, t_hgb, _ = evaluar('Gradient Boosting', y_test, y_proba_hgb)
# %% [markdown]
# ### 2.10 Red Neuronal (MLP)
# Perceptrón multicapa. Arquitectura (64→32) con ReLU y Adam. Early stopping controla sobreajuste.
# %%
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    max_iter=100,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=RANDOM,
    verbose=False
)
mlp.fit(X_train_sc, y_train)

y_proba_mlp = mlp.predict_proba(X_test_sc)[:, 1]
pr_auc_mlp, t_mlp, _ = evaluar('MLP Neural Network', y_test, y_proba_mlp)
# %% [markdown]
# ## 8. Comparación y reporte de desempeño
#
# Sección 2.12: Reportes de estabilidad y desempeño de modelos supervisados.
# %%
from sklearn.metrics import PrecisionRecallDisplay

fig, ax = plt.subplots(figsize=(10, 6))

modelos_plot = [
    ('Dummy (baseline)',   y_proba_dummy, pr_auc_dummy),
    ('Logistic Regression', y_proba_lr, pr_auc_lr),
    ('LDA',               y_proba_lda, pr_auc_lda),
    ('Naive Bayes',       y_proba_gnb, pr_auc_gnb),
    ('Decision Tree',     y_proba_dt,  pr_auc_dt),
    ('Random Forest',     y_proba_rf,  pr_auc_rf),
    ('Gradient Boosting', y_proba_hgb, pr_auc_hgb),
    ('MLP Neural Network',y_proba_mlp, pr_auc_mlp),
]

for nombre, proba, auc in modelos_plot:
    PrecisionRecallDisplay.from_predictions(
        y_test, proba,
        name=f'{nombre} (PR-AUC={auc:.3f})',
        ax=ax
    )

ax.set_title('Curva Precision-Recall — Comparacion de modelos')
ax.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.show()
# %%
resumen_final = pd.DataFrame([
    {'modelo': nombre, 'PR-AUC': round(auc, 4)}
    for nombre, _, auc in modelos_plot
]).sort_values('PR-AUC', ascending=False).reset_index(drop=True)

print('\n=== RESUMEN FINAL ===')
print(resumen_final.to_string(index=False))
# %% [markdown]
# ### Importancia de features (modelo ganador)
# %%
# Importancia del mejor modelo basado en arboles (RF o HGB)
mejor_nombre = resumen_final.iloc[0]['modelo']
print(f'Mejor modelo: {mejor_nombre}')

# Usamos RF para importancia de features (siempre disponible y directamente interpretable)
feat_imp = pd.Series(
    rf.feature_importances_,
    index=X_train_sk.fillna(-1).columns
).nlargest(20).sort_values()

plt.figure(figsize=(8, 7))
plt.barh(feat_imp.index, feat_imp.values, color='#3498db')
plt.title('Top 20 features por importancia — Random Forest')
plt.xlabel('Importancia (reduccion de impureza Gini)')
plt.tight_layout()
plt.show()

print('\nTop 20 features:')
print(feat_imp.sort_values(ascending=False).to_string())
# %% [markdown]
# ## 9. Guardar mejor modelo
# %%
import joblib

# Mapa nombre → objeto modelo
mapa_modelos = {
    'Dummy (baseline)':    dummy,
    'Logistic Regression': lr,
    'LDA':                 lda,
    'Naive Bayes':         gnb,
    'Decision Tree':       dt,
    'Random Forest':       rf,
    'Gradient Boosting':   hgb,
    'MLP Neural Network':  mlp,
}

# Guardamos el mejor modelo (excluyendo el dummy baseline)
resumen_sin_dummy = resumen_final[resumen_final['modelo'] != 'Dummy (baseline)']
nombre_ganador = resumen_sin_dummy.iloc[0]['modelo']
modelo_ganador = mapa_modelos[nombre_ganador]
pr_auc_ganador = resumen_sin_dummy.iloc[0]['PR-AUC']

nombre_archivo = nombre_ganador.replace(' ', '')
ruta_modelo = SALIDA + f'OviedoQuezadaRolando_{nombre_archivo}.joblib'
joblib.dump(modelo_ganador, ruta_modelo)

print(f'Modelo guardado: {nombre_ganador}')
print(f'PR-AUC:          {pr_auc_ganador:.4f}')
print(f'Archivo:         {ruta_modelo}')