#!/usr/bin/env python3
"""
stress_predictor_refined_with_causes.py

Un predictor de estrés estudiantil basado en variables heurísticas,
con clasificación adicional de causas (económica, motivacional, emocional).
Requisitos: pandas, numpy
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Pesos heurísticos (énfasis en GPA y factores financieros)
# -----------------------------------------------------------------------------
WEIGHTS = {
    "GPA_RISK": 0.70,
    "WORST_GPA_RISK": 0.50,
    "ABSENCES_RISK": 0.20,
    "HRS_WEEK_Z": 0.10,
    "TASKS_WEEK_Z": 0.10,
    "CAREER_RISK": 0.05,
    "DOUBLE_DEGREE": 0.15,
    "ON_TRACK_CAREER_RISK": 0.20,
    "COMMUTE_TIME_Z": 0.10,
    "TUITION_Z": 0.60,
    "DEBTS": 0.80,
    "SCHOLARSHIP_RISK": 0.50,
    "MIN_GPA_RISK": 0.40,
    "CREDITS_OVERLOAD": 0.15,
    "ON_TIME_PAY_RISK": 0.60,
    "PROF_RISK": 0.10,
}
BIAS = 1.50

# Umbrales y etiquetas de estrés
THRESHOLDS = [1.5, 2.5, 3.5, 4.5]
LEVEL_LABELS = {
    1: "Ligeramente estresado",
    2: "Moderadamente estresado",
    3: "Estresado",
    4: "Muy estresado",
    5: "Extremadamente estresado",
}

# -----------------------------------------------------------------------------
# Pesos para clasificación de causas (sumados a 1.0 cada grupo)
# -----------------------------------------------------------------------------
ECO_WEIGHTS = {
    "ON_TRACK_CAREER_RISK": 0.20,
    "TUITION_Z":            0.15,
    "DEBTS":                0.20,
    "SCHOLARSHIP_RISK":     0.10,
    "MIN_GPA_RISK":         0.10,
    "CREDITS_OVERLOAD":     0.15,
    "ON_TIME_PAY_RISK":     0.10,
}
MOT_WEIGHTS = {
    "GPA_RISK":             0.40,
    "ABSENCES_RISK":        0.12,
    "TASKS_WEEK_Z":         0.24,
    "CAREER_RISK":          0.16,
    "ON_TRACK_CAREER_RISK": 0.12,
    "CREDITS_OVERLOAD":     0.16,
}
# Umbrales para activar causa
ECO_THRESHOLD = 0.3
MOT_THRESHOLD = 0.3

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(path: Path) -> pd.DataFrame:
    """Carga un CSV o Excel según la extensión."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def compute_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Construye el DataFrame de variables de riesgo en el orden de WEIGHTS."""
    r = pd.DataFrame(index=df.index)
    # académicos
    r["GPA_RISK"]       = 1 - df["GPA"] / 5
    r["WORST_GPA_RISK"] = 1 - df["WORST_GPA"] / 5
    r["ABSENCES_RISK"]  = 1 - df["ATTEND"] / 100
    # z-scores continuas
    for col in ["HRS_WEEK", "TASKS_WEEK", "COMMUTE_TIME", "TUITION"]:
        zcol = f"{col}_Z"
        r[zcol] = (df[col] - df[col].mean()) / df[col].std(ddof=0)
    # booleanos/invertidos
    r["CAREER_RISK"]          = 1 - df["CAREER_PTS"] / 100
    r["DOUBLE_DEGREE"]        = df["DOUBLE_DEGREE"]
    r["ON_TRACK_CAREER_RISK"] = 1 - df["ON_TRACK_CAREER"]
    r["DEBTS"]                = df["DEBTS"]
    r["SCHOLARSHIP_RISK"]     = 1 - df["SCHOLARSHIP"]
    r["MIN_GPA_RISK"]         = np.clip(
        (df["MINIMUM_GPA_FOR_SCHOLARSHIP_MAINTENANCE"] - 3) / 2,
        0, 1
    )
    r["CREDITS_OVERLOAD"]     = df["CREDITS_OVERLOAD"]
    r["ON_TIME_PAY_RISK"]     = 1 - df["ON_TIME_PAY"]
    r["PROF_RISK"]            = 1 - df["PROF_RATING"] / 5
    return r[list(WEIGHTS.keys())]


def compute_stress_scores(risk_df: pd.DataFrame) -> np.ndarray:
    """Calcula la puntuación cruda y aplica saturación."""
    w = np.array(list(WEIGHTS.values()))
    raw = BIAS + risk_df.values.dot(w)
    return np.minimum(raw, 4.0) + 1.0


def classify_scores(scores: np.ndarray) -> np.ndarray:
    """Convierte puntuaciones continuas en niveles 1–5."""
    levels = []
    for s in scores:
        for idx, thr in enumerate(THRESHOLDS, start=1):
            if s <= thr:
                levels.append(idx)
                break
        else:
            levels.append(5)
    return np.array(levels)


def compute_cause_scores(risk_df: pd.DataFrame, weights: dict) -> np.ndarray:
    """Suma ponderada sobre un subconjunto y normaliza a [0,1]."""
    sub = risk_df[list(weights.keys())].values
    w   = np.array(list(weights.values()))
    raw = sub.dot(w)
    return raw / w.sum()


def assign_cause(row, eco_score, mot_score, emo_score):
    """Decide la/s causa/s de estrés para puntajes >=3.5."""
    causes = []
    if eco_score[row.name] >= ECO_THRESHOLD:
        causes.append("económica")
    if mot_score[row.name] >= MOT_THRESHOLD:
        causes.append("motivacional")
    if emo_score[row.name] >= MOT_THRESHOLD and not causes:
        causes.append("emocional")
    return ", ".join(causes)


def main():
    parser = argparse.ArgumentParser(
        description="Predictor de estrés estudiantil con causas."
    )
    parser.add_argument(
        "-i","--input", type=Path, required=False,
        default=Path("estudiantes_con_nombres.xlsx"),
        help="CSV o XLSX con datos de estudiantes."
    )
    parser.add_argument(
        "-o","--output", type=Path, default=Path("estudiantes_con_estres_y_causas.xlsx"),
        help="Archivo de resultados (.xlsx)."
    )
    args = parser.parse_args()

    logger.info("Cargando datos desde %s", args.input)
    df_full = load_data(args.input)
    # Prepara un DataFrame para cálculos sin las columnas nuevas
    df_calc = df_full.drop(columns=["STUDENT_NAME", "INSTITUTIONAL_EMAIL"], errors="ignore")

    logger.info("Calculando variables de riesgo...")
    risk_df = compute_risk(df_calc)

    logger.info("Calculando puntuaciones de estrés...")
    scores = compute_stress_scores(risk_df)
    df_full["STRESS_SCORE"] = np.round(scores, 2)
    df_full["STRESS_LEVEL"] = classify_scores(scores)
    df_full["STRESS_LABEL"] = df_full["STRESS_LEVEL"].map(LEVEL_LABELS)

    # Causas
    eco_scores = compute_cause_scores(risk_df, ECO_WEIGHTS)
    mot_scores = compute_cause_scores(risk_df, MOT_WEIGHTS)
    emo_scores = 1 - np.maximum(eco_scores, mot_scores)
    df_full["ECO_SCORE"] = np.round(eco_scores, 2)
    df_full["MOT_SCORE"] = np.round(mot_scores, 2)
    df_full["EMO_SCORE"] = np.round(emo_scores, 2)

    # Clasificar causas solo para quienes superan umbral
    mask = df_full["STRESS_SCORE"] >= 3.5
    df_full.loc[mask, "STRESS_CAUSE"] = df_full[mask].apply(
        lambda row: assign_cause(row, eco_scores, mot_scores, emo_scores), axis=1
    )
    df_full["STRESS_CAUSE"].fillna("---", inplace=True)

    # Reordenar para que ID, nombre y correo salgan primeros
    cols = df_full.columns.tolist()
    front = ["ID", "STUDENT_NAME", "INSTITUTIONAL_EMAIL"]
    ordered = [c for c in front if c in cols] + [c for c in cols if c not in front]
    df_full = df_full[ordered]

    logger.info("Guardando resultados en %s", args.output)
    df_full.to_excel(args.output, index=False)
    logger.info("¡Proceso completado! Distribución de causas:\n%s", df_full["STRESS_CAUSE"].value_counts())


if __name__ == "__main__":
    main()
