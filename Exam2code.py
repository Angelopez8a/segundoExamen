import gc

import joblib
import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (
    PrecisionRecallDisplay,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
)
from sklearn.tree import DecisionTreeClassifier

matplotlib.use('Agg')

# ─── constantes ───────────────────────────────────────────────────────────────

RUTA_BASE   = 'data/'
RUTA_SALIDA = './'

ESTADOS_USA = {
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID',
    'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS',
    'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
    'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV',
    'WI', 'WY', 'DC',
}

CAT_COLS = [
    'Merchant City', 'Merchant State', 'Zip', 'Card Brand', 'Card Type',
    'Gender', 'City', 'State', 'turno_dia', 'fico_tier',
]

# ─── helpers ──────────────────────────────────────────────────────────────────

def limpiar_dinero(series):
    return series.str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)


def evaluar(nombre, y_true, y_proba):
    pr_auc = average_precision_score(y_true, y_proba)
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    best_t = thresholds[f1.argmax()]
    y_pred = (y_proba >= best_t).astype(int)
    print(f'=== {nombre} ===')
    print(f'PR-AUC: {pr_auc:.4f}  |  threshold optimo: {best_t:.4f}')
    print(classification_report(y_true, y_pred, digits=4))
    return pr_auc, best_t


# ─── carga de datos ───────────────────────────────────────────────────────────

df_users = pd.read_csv(RUTA_BASE + 'sd254_users.csv')
df_cards = pd.read_csv(RUTA_BASE + 'sd254_cards.csv')

# 14M filas: buen balance entre cobertura de fraudes y RAM disponible
# los dtypes reducen el consumo de memoria desde el momento de la lectura
df_trans = pd.read_csv(
    RUTA_BASE + 'credit_card_transactions-ibm_v2.csv',
    nrows=1_000_000,
    dtype={
        'User':  'int32',
        'Card':  'int32',
        'Year':  'int16',
        'Month': 'int8',
        'Day':   'int8',
        'MCC':   'int32',
    },
)

print(df_users.shape, df_cards.shape, df_trans.shape)
print(df_trans['Is Fraud?'].value_counts())
print(df_trans['Is Fraud?'].value_counts(normalize=True))

# ─── limpieza: usuarios ───────────────────────────────────────────────────────

df_users['Per Capita Income - Zipcode'] = limpiar_dinero(df_users['Per Capita Income - Zipcode'])
df_users['Yearly Income - Person']      = limpiar_dinero(df_users['Yearly Income - Person'])
df_users['Total Debt']                  = limpiar_dinero(df_users['Total Debt'])
df_users['Apartment']                   = df_users['Apartment'].fillna('')
df_users.reset_index(inplace=True)
df_users.rename(columns={'index': 'User'}, inplace=True)

# ─── limpieza: tarjetas ───────────────────────────────────────────────────────

df_cards['Credit Limit']     = limpiar_dinero(df_cards['Credit Limit'])
df_cards['Has Chip']         = (df_cards['Has Chip'].str.upper() == 'YES').astype(int)
df_cards['Card on Dark Web'] = (df_cards['Card on Dark Web'].str.strip() == 'Yes').astype(int)
df_cards['Expires']          = pd.to_datetime(df_cards['Expires'],        format='%m/%Y')
df_cards['Acct Open Date']   = pd.to_datetime(df_cards['Acct Open Date'], format='%m/%Y')
df_cards.rename(columns={'CARD INDEX': 'CARD_INDEX'}, inplace=True)
# evita expansion de filas por tarjetas duplicadas en el catalogo
df_cards.drop_duplicates(subset=['User', 'CARD_INDEX'], keep='last', inplace=True)

# ─── limpieza: transacciones ──────────────────────────────────────────────────

df_trans['Amount']    = df_trans['Amount'].str.replace('$', '', regex=False).astype(float).astype('float32')
df_trans['Is Fraud?'] = (df_trans['Is Fraud?'] == 'Yes').astype('int8')
df_trans['Errors?']   = df_trans['Errors?'].fillna('No Error')
df_trans['Zip']       = df_trans['Zip'].fillna(0).astype(int).astype(str)
df_trans.loc[df_trans['Zip'] == '0', 'Zip'] = ''

print('fraudes totales:', df_trans['Is Fraud?'].sum())
print('porcentaje de fraude:', round(df_trans['Is Fraud?'].mean() * 100, 3), '%')

# ─── merge ────────────────────────────────────────────────────────────────────

df = df_trans.merge(
    df_cards,
    left_on=['User', 'Card'],
    right_on=['User', 'CARD_INDEX'],
    how='left',
)
df = df.merge(df_users, on='User', how='left')

del df_trans, df_cards, df_users
gc.collect()

print(df.shape)
print(f'memoria: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB')

cols_inutiles = ['Person', 'Address', 'Card Number', 'CVV', 'Merchant Name', 'CARD_INDEX', 'Card on Dark Web']
df.drop(columns=[c for c in cols_inutiles if c in df.columns], inplace=True)

for col in ['Use Chip', 'Merchant State', 'Merchant City', 'Gender',
            'City', 'State', 'Card Brand', 'Card Type', 'Errors?']:
    if col in df.columns:
        df[col] = df[col].astype('category')

print(f'memoria despues de limpiar: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB')

# ─── feature engineering ──────────────────────────────────────────────────────

df['abs_amount'] = df['Amount'].abs().astype('float32')
df['is_refund']  = (df['Amount'] < 0).astype('int8')

df['hour']   = df['Time'].str.split(':').str[0].astype(int)
df['minute'] = df['Time'].str.split(':').str[1].astype(int)

df['is_madrugada']      = df['hour'].between(0, 5).astype('int8')
df['night_transaction'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype('int8')
df['working_hours']     = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype('int8')
df['turno_dia']         = pd.cut(
    df['hour'],
    bins=[-1, 5, 11, 17, 23],
    labels=['madrugada', 'manana', 'tarde', 'noche'],
)

df['fecha']          = pd.to_datetime(
    df[['Year', 'Month', 'Day']].rename(columns={'Year': 'year', 'Month': 'month', 'Day': 'day'})
)
df['dia_semana']     = df['fecha'].dt.dayofweek
df['is_weekend']     = (df['dia_semana'] >= 5).astype('int8')
df['hora_de_semana'] = (df['hour'] + df['dia_semana'] * 24).astype('int16')
df['es_q4']          = df['Month'].isin([10, 11, 12]).astype('int8')

merchant_state_str = df['Merchant State'].astype(str)
merchant_city_str  = df['Merchant City'].astype(str)

df['different_state'] = (merchant_state_str != df['State'].astype(str)).astype('int8')
df['different_city']  = (merchant_city_str  != df['City'].astype(str)).astype('int8')
df['is_online']       = (merchant_city_str.str.upper() == 'ONLINE').astype('int8')
df['is_out_of_state'] = (
    (merchant_state_str.str.len() == 2) & (merchant_state_str != df['State'].astype(str))
).astype('int8')
df['is_extranjero'] = (
    ~merchant_state_str.isin(ESTADOS_USA) & df['Merchant State'].notna()
).astype('int8')

df['swipe_transaction']  = (df['Use Chip'] == 'Swipe Transaction').astype('int8')
df['chip_transaction']   = (df['Use Chip'] == 'Chip Transaction').astype('int8')
df['online_transaction'] = (df['Use Chip'] == 'Online Transaction').astype('int8')

df['has_error']        = (df['Errors?'] != 'No Error').astype('int8')
df['auth_error']       = df['Errors?'].str.contains('Bad PIN',              na=False).astype('int8')
df['financial_error']  = df['Errors?'].str.contains('Insufficient Balance', na=False).astype('int8')
df['error_cvv']        = df['Errors?'].str.contains('Bad CVV',              na=False).astype('int8')
df['error_expiracion'] = df['Errors?'].str.contains('Bad Expiration',       na=False).astype('int8')
df['technical_error']  = df['Errors?'].str.contains('Technical Glitch',     na=False).astype('int8')

df['has_chip']      = df['Has Chip'].astype('int8')
df['card_reissued'] = (df['Cards Issued'] > 1).astype('int8')
df.drop(columns=['Has Chip'], inplace=True)

credit_limit = df['Credit Limit'].replace(0, np.nan)
df['amount_credit_ratio']    = (df['Amount'] / credit_limit).fillna(0).clip(0, 5).astype('float32')
df['cerca_del_limite']       = (df['amount_credit_ratio'] > 0.8).astype('int8')
df['saldo_disponible_ratio'] = (
    (df['Credit Limit'] - df['Amount']) / credit_limit
).fillna(0).clip(-1, 1).astype('float32')

df['dias_para_vencer'] = (df['Expires'] - df['fecha']).dt.days.clip(lower=-365).astype('float32')
df['tarjeta_vencida']  = (df['dias_para_vencer'] < 0).astype('int8')
df['card_age_months']  = ((df['fecha'] - df['Acct Open Date']).dt.days / 30).clip(lower=0).astype('float32')
df['card_age_years']   = (df['Year'] - df['Acct Open Date'].dt.year).clip(lower=0).astype('float32')
df['young_card']       = (df['card_age_years'] <= 1).astype('int8')
df['pin_age']          = (df['Year'] - df['Year PIN last Changed']).clip(lower=0).astype('float32')

# conservamos 'fecha' para los historicos de abajo; Expires y Acct Open Date ya no se necesitan
df.drop(columns=['Expires', 'Acct Open Date'], inplace=True)

yearly_income = df['Yearly Income - Person'].replace(0, np.nan)
df['debt_income_ratio']         = (df['Total Debt'] / yearly_income).fillna(0).astype('float32')
df['years_to_retirement']       = (df['Retirement Age'] - df['Current Age']).clip(lower=0).astype('float32')
df['is_retired']                = (df['Current Age'] >= df['Retirement Age']).astype('int8')
df['ingreso_vs_zona']           = (yearly_income / df['Per Capita Income - Zipcode'].replace(0, np.nan)).fillna(1).astype('float32')
df['amount_daily_income_ratio'] = (df['Amount'] / (yearly_income / 365)).fillna(0).clip(0, 100).astype('float32')
df['amount_income_ratio']       = (df['Amount'] / yearly_income).fillna(0).clip(0, 1).astype('float32')
df['amount_to_monthly_income']  = (df['abs_amount'] / (yearly_income / 12)).fillna(0).clip(0, 100).astype('float32')
df['credit_limit_to_income']    = (df['Credit Limit'] / yearly_income).fillna(0).clip(0, 10).astype('float32')
df['debt_to_credit_limit']      = (df['Total Debt'] / credit_limit).fillna(0).clip(0, 50).astype('float32')
df['low_fico']                  = (df['FICO Score'] < 650).astype('int8')
df['high_debt']                 = (df['debt_income_ratio'] > 2).astype('int8')
df['many_cards']                = (df['Num Credit Cards'] >= 5).astype('int8')
df['tiene_depto']               = (df['Apartment'].str.strip() != '').astype('int8')
df['fico_tier']                 = pd.cut(
    df['FICO Score'],
    bins=[0, 579, 669, 739, 799, 850],
    labels=['malo', 'regular', 'bueno', 'muy_bueno', 'excelente'],
)
df.drop(columns=['Apartment'], inplace=True)

# ─── features historicas (sin data leakage) ───────────────────────────────────
# ordenamos cronologicamente para que cumsum solo vea transacciones anteriores

df['_orden_original'] = np.arange(len(df))
df.sort_values(['User', 'Year', 'Month', 'Day', 'hour', 'minute', 'Card'], inplace=True)
df.reset_index(drop=True, inplace=True)

g_user = df.groupby('User', sort=False)
df['txns_prev_usuario'] = g_user.cumcount().astype('int32')

cum_amount        = g_user['abs_amount'].cumsum()
df['_abs_sq']     = df['abs_amount'] ** 2
cum_amount_sq     = g_user['_abs_sq'].cumsum()

prev_count   = df['txns_prev_usuario'].replace(0, np.nan)
prev_sum     = cum_amount - df['abs_amount']
prev_sum_sq  = cum_amount_sq - df['_abs_sq']

df['gasto_promedio_usuario_hist'] = (prev_sum / prev_count).fillna(0).astype('float32')
var_prev = (prev_sum_sq / prev_count) - (df['gasto_promedio_usuario_hist'] ** 2)
df['gasto_std_usuario_hist']      = np.sqrt(var_prev.clip(lower=0)).fillna(0).astype('float32')
df['gasto_max_usuario_hist']      = (
    g_user['abs_amount'].cummax().groupby(df['User']).shift(1).fillna(0).astype('float32')
)

df['z_score_monto_hist'] = (
    (df['abs_amount'] - df['gasto_promedio_usuario_hist']) /
    df['gasto_std_usuario_hist'].replace(0, np.nan)
).fillna(0).clip(-10, 10).astype('float32')

df['supera_maximo_historico'] = (
    (df['txns_prev_usuario'] > 0) & (df['abs_amount'] > df['gasto_max_usuario_hist'])
).astype('int8')

prev_online = g_user['online_transaction'].cumsum() - df['online_transaction']
df['online_ratio_usuario_hist'] = (prev_online / prev_count).fillna(0).astype('float32')
df['online_inusual'] = (
    (df['online_transaction'] == 1) &
    (df['txns_prev_usuario'] > 0) &
    (df['online_ratio_usuario_hist'] < 0.05)
).astype('int8')

primer_mcc_usuario = ~df.duplicated(['User', 'MCC'])
df['mcc_diversity_usuario_hist'] = (
    primer_mcc_usuario.astype('int16').groupby(df['User']).cumsum()
    .groupby(df['User']).shift(1).fillna(0).astype('int16')
)

df.drop(columns=['_abs_sq'], inplace=True)

# transacciones previas de la misma tarjeta en el mismo dia
g_card_day = df.groupby(['User', 'Card', 'Year', 'Month', 'Day'], sort=False)
df['txns_prev_mismo_dia'] = g_card_day.cumcount().astype('int16')
df['velocidad_alta']      = (df['txns_prev_mismo_dia'] > 5).astype('int8')

df['triple_riesgo'] = (
    df['is_madrugada'] & df['is_online'] & df['different_state']
).astype('int8')

df['score_riesgo_manual'] = (
    df['is_extranjero']    * 2 +
    df['different_state']  * 1 +
    df['is_madrugada']     * 1 +
    df['has_error']        * 2 +
    df['auth_error']       * 1 +
    df['error_cvv']        * 1 +
    df['cerca_del_limite'] * 1 +
    df['tarjeta_vencida']  * 1 +
    df['velocidad_alta']   * 2
).astype('int8')

df['monto_redondo']      = (df['Amount'] % 1 == 0).astype('int8')
df['monto_psicologico']  = ((df['Amount'] % 1) >= 0.95).astype('int8')
df['monto_multiplo_100'] = ((df['Amount'] % 100 == 0) & (df['Amount'] > 0)).astype('int8')
df['monto_multiplo_50']  = ((df['Amount'] % 50  == 0) & (df['Amount'] > 0)).astype('int8')

# estados distintos visitados por la tarjeta durante el dia hasta esta transaccion
primer_estado_dia = ~df.duplicated(['User', 'Card', 'Year', 'Month', 'Day', 'Merchant State'])
keys_dia = [df['User'], df['Card'], df['Year'], df['Month'], df['Day']]
df['estados_distintos_dia_hasta_ahora'] = (
    primer_estado_dia.astype('int16').groupby(keys_dia).cumsum().astype('int16')
)
df['tarjeta_en_varios_estados'] = (df['estados_distintos_dia_hasta_ahora'] > 1).astype('int8')
gc.collect()

# restauramos el orden original
df.sort_values('_orden_original', inplace=True)
df.drop(columns=['_orden_original'], inplace=True)
df.reset_index(drop=True, inplace=True)

# ─── validacion de features binarias ─────────────────────────────────────────

vars_binarias = [
    'is_madrugada', 'night_transaction', 'working_hours', 'is_weekend', 'es_q4',
    'different_state', 'different_city', 'is_online', 'is_out_of_state', 'is_extranjero',
    'swipe_transaction', 'chip_transaction', 'online_transaction',
    'has_error', 'auth_error', 'financial_error', 'error_cvv',
    'has_chip', 'card_reissued', 'young_card', 'cerca_del_limite', 'tarjeta_vencida',
    'velocidad_alta', 'tarjeta_en_varios_estados', 'triple_riesgo',
    'is_refund', 'monto_redondo', 'monto_psicologico', 'monto_multiplo_100', 'monto_multiplo_50',
    'online_inusual', 'supera_maximo_historico',
    'is_retired', 'low_fico', 'high_debt', 'many_cards',
]

resultados = []
for var in vars_binarias:
    if var in df.columns:
        tasa_0 = df[df[var] == 0]['Is Fraud?'].mean()
        tasa_1 = df[df[var] == 1]['Is Fraud?'].mean()
        lift   = tasa_1 / tasa_0 if tasa_0 > 0 else 0
        resultados.append({'variable': var, 'tasa_si': tasa_1, 'tasa_no': tasa_0, 'lift': lift})

res = pd.DataFrame(resultados).sort_values('lift', ascending=False)
print(res.to_string())

colors = ['#e74c3c' if x > 1.5 else '#95a5a6' for x in res['lift']]
plt.figure(figsize=(9, 6))
plt.barh(res['variable'], res['lift'], color=colors)
plt.axvline(x=1, color='black', linestyle='--', linewidth=1)
plt.xlabel('lift (tasa fraude con flag=1 vs flag=0)')
plt.title('que tanto distingue cada variable')
plt.tight_layout()
plt.savefig(RUTA_SALIDA + 'lift_features.png', dpi=150)
plt.close()
print('grafica guardada: lift_features.png')

# ─── preparar dataset para modelado ──────────────────────────────────────────
# se excluyen: IDs, texto operativo, y variables con fuga (tasa_fraude_* usan el target)

cols_drop = ['Time', 'Errors?', 'Use Chip', 'User', 'fecha',
             'tasa_fraude_mcc', 'tasa_fraude_estado']
df.drop(columns=[c for c in cols_drop if c in df.columns], inplace=True)
df_model = df
del df
gc.collect()

print(f'shape final: {df_model.shape}')
print(f'memoria: {df_model.memory_usage(deep=True).sum() / 1e9:.2f} GB')
print(f'fraudes: {df_model["Is Fraud?"].sum():,}  ({df_model["Is Fraud?"].mean() * 100:.3f}%)')

for col in CAT_COLS:
    if col in df_model.columns:
        df_model[col] = df_model[col].astype('category')

TARGET = 'Is Fraud?'
X = df_model.drop(columns=[TARGET])
y = df_model[TARGET]

print(f'features disponibles antes del split: {X.shape[1]}')

# ─── split temporal ───────────────────────────────────────────────────────────
# entrenamos en el pasado y evaluamos en lo mas reciente para simular uso real

year_split = int(df_model['Year'].quantile(0.85))
print(f'entrenando hasta {year_split}, evaluando {year_split + 1} en adelante')

train_mask = df_model['Year'] <= year_split
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

print(f'train: {X_train.shape}  fraudes: {y_train.sum():,}')
print(f'test:  {X_test.shape}   fraudes: {y_test.sum():,}')

n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()

# ─── preparar features para sklearn ──────────────────────────────────────────
# alta cardinalidad se dropea; ya tenemos features geograficas que la resumen

cols_drop_cats = ['Merchant City', 'Merchant State', 'City', 'State', 'Zip']
X_train_sk = X_train.drop(columns=cols_drop_cats)
X_test_sk  = X_test.drop(columns=cols_drop_cats)

for col in ['Card Brand', 'Card Type', 'Gender', 'turno_dia', 'fico_tier']:
    X_train_sk[col] = X_train_sk[col].cat.codes
    X_test_sk[col]  = X_test_sk[col].cat.codes

obj_cols = X_train_sk.select_dtypes(include='object').columns.tolist()
if obj_cols:
    X_train_sk = X_train_sk.drop(columns=obj_cols)
    X_test_sk  = X_test_sk.drop(columns=obj_cols)

print(X_train_sk.shape)

# ─── baseline ─────────────────────────────────────────────────────────────────

dummy = DummyClassifier(strategy='stratified', random_state=42)
dummy.fit(X_train_sk, y_train)
y_proba_dummy = dummy.predict_proba(X_test_sk)[:, 1]
print('PR-AUC dummy:', round(average_precision_score(y_test, y_proba_dummy), 4))
print('F1 dummy:    ', round(f1_score(y_test, dummy.predict(X_test_sk)), 4))

# ─── decision tree ────────────────────────────────────────────────────────────

dt = DecisionTreeClassifier(
    max_depth=12, min_samples_leaf=50, class_weight='balanced', random_state=42
)
dt.fit(X_train_sk.fillna(-1), y_train)
y_proba_dt = dt.predict_proba(X_test_sk.fillna(-1))[:, 1]
pr_auc_dt, _ = evaluar('Decision Tree', y_test, y_proba_dt)

# ─── random forest ────────────────────────────────────────────────────────────

rf = RandomForestClassifier(
    n_estimators=100, max_depth=15, min_samples_leaf=50,
    max_samples=0.5, class_weight='balanced_subsample',
    n_jobs=-1, random_state=42,
)
rf.fit(X_train_sk.fillna(-1), y_train)
y_proba_rf = rf.predict_proba(X_test_sk.fillna(-1))[:, 1]
pr_auc_rf, _ = evaluar('Random Forest', y_test, y_proba_rf)

# ─── xgboost (GPU) ────────────────────────────────────────────────────────────

scale_pw = n_neg / n_pos
hgb = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=50,
    scale_pos_weight=scale_pw,
    early_stopping_rounds=20,
    eval_metric='aucpr',
    device='cuda',
    random_state=42,
)
hgb.fit(
    X_train_sk.fillna(-1), y_train,
    eval_set=[(X_test_sk.fillna(-1), y_test)],
    verbose=100,
)
y_proba_hgb = hgb.predict_proba(X_test_sk.fillna(-1))[:, 1]
pr_auc_hgb, _ = evaluar('XGBoost', y_test, y_proba_hgb)

# ─── lightgbm ─────────────────────────────────────────────────────────────────

try:
    X_train_lgbm = X_train.copy()
    X_test_lgbm  = X_test.copy()
    cat_cols_presentes = [c for c in CAT_COLS if c in X_train_lgbm.columns]
    for col in cat_cols_presentes:
        # +1 shifts -1 (NaN code) to 0; LightGBM requires non-negative integer codes
        X_train_lgbm[col] = (X_train_lgbm[col].cat.codes + 1).astype('int16')
        X_test_lgbm[col]  = (X_test_lgbm[col].cat.codes + 1).astype('int16')

    lgbm = lgb.LGBMClassifier(
        objective='binary',
        scale_pos_weight=n_neg / n_pos,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=100,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    lgbm.fit(
        X_train_lgbm, y_train,
        categorical_feature=cat_cols_presentes,
        eval_set=[(X_test_lgbm, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    y_proba_lgbm = lgbm.predict_proba(X_test_lgbm)[:, 1]
    pr_auc_lgbm, _ = evaluar('LightGBM', y_test, y_proba_lgbm)
    lgbm_disponible = True

    feat_imp = pd.Series(lgbm.feature_importances_, index=X_train_lgbm.columns).nlargest(20).sort_values()
    plt.figure(figsize=(8, 6))
    plt.barh(feat_imp.index, feat_imp.values, color='#3498db')
    plt.title('top 20 features - LightGBM')
    plt.tight_layout()
    plt.savefig(RUTA_SALIDA + 'feature_importance_lgbm.png', dpi=150)
    plt.close()
    print('grafica guardada: feature_importance_lgbm.png')

except ImportError:
    print('LightGBM no esta instalado, se omite.')
    lgbm_disponible = False
    pr_auc_lgbm     = np.nan

# ─── comparacion de modelos ───────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))
PrecisionRecallDisplay.from_predictions(y_test, y_proba_dt,  name=f'Decision Tree       (PR-AUC={pr_auc_dt:.3f})',  ax=ax)
PrecisionRecallDisplay.from_predictions(y_test, y_proba_rf,  name=f'Random Forest       (PR-AUC={pr_auc_rf:.3f})',  ax=ax)
PrecisionRecallDisplay.from_predictions(y_test, y_proba_hgb, name=f'XGBoost             (PR-AUC={pr_auc_hgb:.3f})', ax=ax)
if lgbm_disponible:
    PrecisionRecallDisplay.from_predictions(y_test, y_proba_lgbm, name=f'LightGBM            (PR-AUC={pr_auc_lgbm:.3f})', ax=ax)
ax.set_title('Precision-Recall — comparacion de modelos')
plt.tight_layout()
plt.savefig(RUTA_SALIDA + 'precision_recall_modelos.png', dpi=150)
plt.close()
print('grafica guardada: precision_recall_modelos.png')

# ─── resumen y guardar el mejor modelo ───────────────────────────────────────

modelos_entrenados = {
    'DecisionTree': (dt,  pr_auc_dt),
    'RandomForest': (rf,  pr_auc_rf),
    'XGBoost':      (hgb, pr_auc_hgb),
}
if lgbm_disponible:
    modelos_entrenados['LightGBM'] = (lgbm, pr_auc_lgbm)

resumen_final = pd.DataFrame({
    'modelo': list(modelos_entrenados.keys()),
    'PR-AUC': [v[1] for v in modelos_entrenados.values()],
}).dropna().sort_values('PR-AUC', ascending=False).reset_index(drop=True)
print(resumen_final.to_string())

mejor_nombre, (mejor_modelo, mejor_pr_auc) = max(modelos_entrenados.items(), key=lambda x: x[1][1])
ruta_modelo = RUTA_SALIDA + f'OviedoQuezadaRolando_{mejor_nombre}.joblib'
joblib.dump(mejor_modelo, ruta_modelo)
print(f'modelo guardado: {mejor_nombre}  |  PR-AUC: {mejor_pr_auc:.4f}')