import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dash import Dash, html, dcc, Input, Output, State
import base64  # 👈 para codificar el logo

# -----------------------------
# 0. CARGAR LOGO UAX
# -----------------------------
with open("uax.png", "rb") as f:   # cambia el nombre si tu archivo se llama distinto
    uax_base64 = base64.b64encode(f.read()).decode("ascii")

# -----------------------------
# 1. CARGAR MODELO Y PREPROCESADO
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# cargar objetos
scaler = joblib.load("scaler.pkl")
num_medians = joblib.load("num_medians.pkl")
original_num_cols = joblib.load("original_num_cols.pkl")
missing_flag_cols = joblib.load("missing_flag_cols.pkl")
cat_cols = joblib.load("cat_cols.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# rangos aproximados para las numéricas a partir del StandardScaler (media ± 3·std)
num_means = scaler.mean_
num_stds = scaler.scale_
NUM_RANGES = {
    col: (m - 3 * s, m + 3 * s)
    for col, m, s in zip(original_num_cols, num_means, num_stds)
}

# define arquitectura EXACTAMENTE igual que en entrenamiento
n_features = len(original_num_cols) + len(missing_flag_cols) + len(cat_cols)

model = nn.Sequential(
    nn.Linear(n_features, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(32, 2)   # 2 clases
)

state_dict = torch.load("best_model.pt", map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# -----------------------------
# 2. VARIABLES QUE PEDIRÁ LA APP
# -----------------------------
FEATURES_NUM = [
    "edad",
    "glucosa",
    "urea",
    "proteinas_totales",
    "fosforo",
    "eosinofilos",
    "leucocitos",
    "hemoglobina",
    "plaquetas",
    "albumina",
]

FEATURES_CAT = [
    "sexo",
    "tipo_tumor",
    "estadio_ajcc",
    "t_invasivo",
    "n_linfatico",
    "metastasis",
    "grado_histologico",
    "receptor_hormonal",
    "mut_egfr",
    "mut_pi3k",
]

# opciones de cada categórica a partir del LabelEncoder
cat_options = {
    col: sorted(list(label_encoders[col].classes_))
    for col in FEATURES_CAT
}

# helper para mostrar etiqueta con rango
def numeric_label(col):
    if col in NUM_RANGES:
        lo, hi = NUM_RANGES[col]
        return f"{col} (≈ {lo:.1f} – {hi:.1f})"
    return col

# -----------------------------
# 3. FUNCIÓN DE PREDICCIÓN
# -----------------------------
def preprocess_and_predict(input_dict):
    """
    input_dict: diccionario {col: valor} de SOLO las 20 features de la app.
    Devuelve probabilidad de clase 1 (vive).
    """

    # 1) Construir un DataFrame con UNA SOLA FILA para todas las columnas del modelo
    df_num = pd.DataFrame(columns=original_num_cols)
    df_cat = pd.DataFrame(columns=cat_cols)

    df_num.loc[0] = np.nan
    df_cat.loc[0] = "MISSING"

    # 2) Rellenar sólo las columnas que la app pide
    for col, val in input_dict.items():
        if col in original_num_cols:
            df_num.at[0, col] = float(val) if val is not None else np.nan
        elif col in cat_cols:
            df_cat.at[0, col] = str(val) if val not in [None, ""] else "MISSING"

    # 3) NUMÉRICAS: crear *_missing + imputar mediana de TRAIN
    for col in original_num_cols:
        missing_col = col + "_missing"
        df_num[missing_col] = df_num[col].isna().astype(int)
        df_num[col] = df_num[col].fillna(num_medians[col])

    # separar flags
    flag_cols = [c for c in df_num.columns if c.endswith("_missing")]

    # escalar solo las columnas originales
    num_scaled = scaler.transform(df_num[original_num_cols].values.astype("float32"))
    df_num_scaled = pd.DataFrame(num_scaled, columns=original_num_cols)
    # añadir flags sin escalar
    for col in flag_cols:
        df_num_scaled[col] = df_num[col].values

    # 4) CATEGÓRICAS: asegurar MISSING para categorías desconocidas y aplicar encoder
    for col in cat_cols:
        enc = label_encoders[col]
        known = set(enc.classes_)
        val = df_cat.at[0, col]
        if val not in known:
            val = "MISSING"
        df_cat.at[0, col] = val
        df_cat[col] = enc.transform(df_cat[col].astype(str))

    # 5) Unir num + cat y convertir a tensor
    X_all = np.hstack([
        df_num_scaled.values.astype("float32"),
        df_cat.values.astype("float32")
    ])

    x_tensor = torch.from_numpy(X_all).float().to(device)

    # 6) Predicción con el modelo
    with torch.no_grad():
        logits = model(x_tensor)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()[0]  # prob clase 1

    return float(probs)

def texto_recomendacion(prob):
    # prob = probabilidad de clase 1 (vive)
    if prob < 0.2:
        return "Riesgo muy alto de fallecimiento. Probabilidad de supervivencia baja. Priorizar valoración en comité y considerar estrategias terapéuticas intensivas."
    elif prob < 0.5:
        return "Riesgo elevado. Paciente con múltiples factores desfavorables. Recomendable revisión detallada del caso y seguimiento estrecho."
    elif prob < 0.8:
        return "Riesgo intermedio. Combina factores de buen y mal pronóstico. Valorar individualmente según contexto clínico."
    else:
        return "Riesgo bajo. El modelo estima alta probabilidad de supervivencia. Mantener seguimiento estándar y reevaluar ante cambios clínicos."

def categoria_riesgo(prob):
    # riesgo respecto a fallecer (1 - prob de vivir)
    prob_die = 1.0 - prob
    if prob_die < 0.2:
        return "Riesgo de fallecimiento: MUY BAJO", "#198754"  # verde
    elif prob_die < 0.5:
        return "Riesgo de fallecimiento: BAJO-MODERADO", "#0d6efd"  # azul
    elif prob_die < 0.8:
        return "Riesgo de fallecimiento: MODERADO-ALTO", "#fd7e14"  # naranja
    else:
        return "Riesgo de fallecimiento: MUY ALTO", "#dc3545"  # rojo

def barra_riesgo(prob):
    prob_live = prob
    prob_die = 1.0 - prob
    titulo, color = categoria_riesgo(prob)

    width = f"{prob_die * 100:.1f}%"

    return html.Div(
        children=[
            html.Div(
                titulo,
                style={
                    "fontSize": "15px",
                    "fontWeight": "600",
                    "marginBottom": "6px",
                    "color": "#343a40",
                },
            ),
            html.Div(
                style={
                    "width": "100%",
                    "height": "18px",
                    "backgroundColor": "#e9ecef",
                    "borderRadius": "999px",
                    "overflow": "hidden",
                },
                children=[
                    html.Div(
                        style={
                            "width": width,
                            "height": "100%",
                            "backgroundColor": color,
                            "transition": "width 0.4s ease",
                        }
                    )
                ],
            ),
            html.Div(
                f"Prob. supervivencia: {prob_live*100:.1f}%   |   Prob. fallecimiento: {prob_die*100:.1f}%",
                style={
                    "fontSize": "13px",
                    "color": "#6c757d",
                    "marginTop": "6px",
                },
            ),
        ]
    )

# -----------------------------
# 4. APP DASH (ESTILO MÉDICO + LOGO UAX)
# -----------------------------
app = Dash(__name__)

app.layout = html.Div(
    style={
        "minHeight": "100vh",
        "backgroundColor": "#f4f6f9",
        "fontFamily": "Segoe UI, Arial, sans-serif",
        "padding": "20px",
    },
    children=[
        html.Div(
            style={
                "maxWidth": "1100px",
                "margin": "0 auto",
                "backgroundColor": "white",
                "borderRadius": "12px",
                "boxShadow": "0 4px 16px rgba(0,0,0,0.08)",
                "padding": "25px 30px 30px 30px",
            },
            children=[
                # Header con logo
                html.Div([
                    html.Div(
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "space-between",
                            "marginBottom": "10px",
                        },
                        children=[
                            html.Img(
                                src=f"data:image/png;base64,{uax_base64}",
                                style={
                                    "height": "100px",
                                    "marginRight": "30px",
                                },
                            ),
                            html.Div(
                                children=[
                                    html.H1(
                                        "Asistente de riesgo oncológico",
                                        style={
                                            "color": "#0b5ed7",
                                            "marginBottom": "3px",
                                            "fontWeight": "600",
                                            "fontSize": "26px",
                                        },
                                    ),
                                    html.P(
                                        "Universidad Alfonso X el Sabio · Proyecto de apoyo a la decisión clínica de Harold Sebastián",
                                        style={
                                            "color": "#6c757d",
                                            "marginTop": "0",
                                            "fontSize": "14px",
                                        },
                                    ),
                                ]
                            ),
                        ],
                    ),
                    html.Hr(),
                ]),

                # Contenido: lado izquierdo numérico, derecho categórico
                html.Div(
                    style={
                        "display": "flex",
                        "gap": "30px",
                        "flexWrap": "wrap",
                    },
                    children=[
                        # NUMÉRICAS
                        html.Div(
                            style={"flex": "1 1 380px"},
                            children=[
                                html.H2(
                                    "Datos numéricos",
                                    style={
                                        "fontSize": "20px",
                                        "color": "#343a40",
                                        "marginBottom": "10px",
                                    },
                                ),
                                html.Div(
                                    children=[
                                        html.Div(
                                            style={
                                                "marginBottom": "10px",
                                                "display": "flex",
                                                "flexDirection": "column",
                                            },
                                            children=[
                                                html.Label(
                                                    numeric_label(col),
                                                    style={
                                                        "fontSize": "14px",
                                                        "color": "#495057",
                                                        "marginBottom": "2px",
                                                    },
                                                ),
                                                dcc.Input(
                                                    id=f"num-{col}",
                                                    type="number",
                                                    step="any",
                                                    style={
                                                        "width": "100%",
                                                        "padding": "6px 8px",
                                                        "borderRadius": "6px",
                                                        "border": "1px solid #ced4da",
                                                    },
                                                ),
                                            ],
                                        )
                                        for col in FEATURES_NUM
                                    ]
                                ),
                            ],
                        ),

                        # CATEGÓRICAS
                        html.Div(
                            style={"flex": "1 1 380px"},
                            children=[
                                html.H2(
                                    "Datos categóricos",
                                    style={
                                        "fontSize": "20px",
                                        "color": "#343a40",
                                        "marginBottom": "10px",
                                    },
                                ),
                                html.Div(
                                    children=[
                                        html.Div(
                                            style={
                                                "marginBottom": "10px",
                                                "display": "flex",
                                                "flexDirection": "column",
                                            },
                                            children=[
                                                html.Label(
                                                    col,
                                                    style={
                                                        "fontSize": "14px",
                                                        "color": "#495057",
                                                        "marginBottom": "2px",
                                                    },
                                                ),
                                                dcc.Dropdown(
                                                    id=f"cat-{col}",
                                                    options=[
                                                        {"label": v, "value": v}
                                                        for v in cat_options[col]
                                                    ],
                                                    placeholder="Seleccione...",
                                                    style={
                                                        "width": "100%",
                                                    },
                                                ),
                                            ],
                                        )
                                        for col in FEATURES_CAT
                                    ]
                                ),
                            ],
                        ),
                    ],
                ),

                # Botón
                html.Div(
                    style={"marginTop": "20px"},
                    children=[
                        html.Button(
                            "Calcular riesgo",
                            id="btn-predict",
                            n_clicks=0,
                            style={
                                "backgroundColor": "#0b5ed7",
                                "color": "white",
                                "border": "none",
                                "padding": "10px 22px",
                                "borderRadius": "8px",
                                "fontSize": "16px",
                                "cursor": "pointer",
                            },
                        )
                    ],
                ),

                # Resultado
                html.Hr(style={"marginTop": "25px"}),
                html.Div(
                    children=[
                        html.H3(
                            "Resultado",
                            style={"color": "#343a40", "marginBottom": "10px"},
                        ),
                        html.Div(
                            id="output-prob",
                            style={
                                "fontSize": "20px",
                                "fontWeight": "600",
                                "color": "#0b5ed7",
                                "marginBottom": "5px",
                            },
                        ),
                        html.Div(
                            id="risk-bar",
                            style={"marginBottom": "10px"},
                        ),
                        html.Div(
                            id="output-texto",
                            style={
                                "fontSize": "15px",
                                "color": "#495057",
                                "maxWidth": "800px",
                            },
                        ),
                        html.P(
                            "⚠️ Esta herramienta es de apoyo a la decisión clínica. No sustituye el juicio del especialista.",
                            style={
                                "fontSize": "12px",
                                "color": "#adb5bd",
                                "marginTop": "15px",
                            },
                        ),
                    ]
                ),
            ],
        )
    ],
)

# -----------------------------
# 5. CALLBACK
# -----------------------------
@app.callback(
    [Output("output-prob", "children"),
     Output("output-texto", "children"),
     Output("risk-bar", "children")],
    [Input("btn-predict", "n_clicks")],
    [State(f"num-{col}", "value") for col in FEATURES_NUM] +
    [State(f"cat-{col}", "value") for col in FEATURES_CAT]
)
def actualizar_prediccion(n_clicks, *values):
    if n_clicks == 0:
        return "", "", ""

    num_values = values[:len(FEATURES_NUM)]
    cat_values = values[len(FEATURES_NUM):]

    # Si todo vacío → pedir más información
    if all(v is None for v in num_values + cat_values):
        msg = "Por favor, introduzca al menos algunos datos del paciente para obtener una estimación."
        return msg, "", ""

    total_campos = len(num_values) + len(cat_values)
    rellenos = sum(v is not None for v in num_values + cat_values)
    frac = rellenos / total_campos

    if frac < 0.3:
        msg = "Información insuficiente: rellene más campos para que la estimación sea clínicamente útil."
        return msg, "", ""

    input_dict = {}
    for col, val in zip(FEATURES_NUM, num_values):
        input_dict[col] = val

    for col, val in zip(FEATURES_CAT, cat_values):
        input_dict[col] = val

    prob = preprocess_and_predict(input_dict)
    texto = texto_recomendacion(prob)
    barra = barra_riesgo(prob)

    return (
        f"Probabilidad estimada de supervivencia (clase 1): {prob*100:.1f}%",
        texto,
        barra
    )

# -----------------------------
# 6. RUN
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
