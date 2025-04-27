#!/usr/bin/env python3
"""
personalized_followup_genai_excel.py

Genera acompañamiento personalizado para estudiantes leyendo un Excel
y consultando Gemini vía google-generativeai.
Ahora incluye el nombre del estudiante para respuestas más personales.
Requisitos:
    pip install pandas openpyxl python-dotenv google-generativeai
    Define en un .env o en tu entorno:
        GENAI_API_KEY="TU_API_KEY_DE_GEMINI"
        GEMINI_MODEL_ID="gemini-1.5-flash"
"""

import os
import time
import pandas as pd
import textwrap
from dotenv import load_dotenv
import google.generativeai as genai

# 1. Configuración de entorno y cliente
load_dotenv()
API_KEY = "AIzaSyAqzQbReUQu3OJOWZwulhqzyEcZrq7w_Z8"
MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-1.5-flash")
if not API_KEY:
    raise RuntimeError("Define GENAI_API_KEY en tu .env o en tu entorno")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_ID)

# 2. Carga de datos de estudiantes (solo los 3 primeros)
def load_student_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(path, engine="openpyxl")
        return df.head(3)
    except Exception as e:
        raise RuntimeError(f"Error cargando datos: {e}")

# 3. Mapeo de puntos de carrera a descripción
def map_career(career_pts: float) -> str:
    mappings = [
        (80, "Ingeniería (alta dificultad, alto empleo)"),
        (60, "Administración de Empresas (media-alta dificultad, buena empleabilidad)"),
        (40, "Comunicación Social (media dificultad, empleabilidad moderada)"),
    ]
    for threshold, desc in mappings:
        if career_pts >= threshold:
            return desc
    return "Diseño Gráfico (baja dificultad, empleabilidad variable)"

# 4. Construcción del prompt para cada estudiante, ahora usando su nombre
def build_prompt(row: pd.Series) -> str:
    # Mapas de causa
    stress_map = {
        "económica": [
            "Identificamos desafíos económicos. Recomendaciones:",
            "• Plan financiero con presupuesto detallado",
            "• Búsqueda de becas y ayudas disponibles",
            "• Estrategias para reducir gastos académicos",
        ],
        "motivacional": [
            "Detectamos falta de motivación. Sugerencias:",
            "• Plan de desarrollo profesional claro",
            "• Conexión con egresados de la carrera",
            "• Metas académicas a corto y mediano plazo",
        ],
        "emocional": [
            "Reconocemos dificultades emocionales. Orientación:",
            "• Técnicas de manejo de estrés académico",
            "• Recursos de apoyo psicológico disponibles",
            "• Estrategias para balance vida-estudio",
        ],
    }
    # Selección de causas según columna STRESS_CAUSE
    causes = []
    cause_text = row.STRESS_CAUSE.lower()
    if "económica" in cause_text:
        causes += stress_map["económica"]
    if "motivacional" in cause_text:
        causes += stress_map["motivacional"]
    if not causes or "emocional" in cause_text:
        causes += stress_map["emocional"]

    career_info = map_career(row.CAREER_PTS)

    # Incorporar nombre
    name = row.get("STUDENT_NAME")
    if name:
        intro = f"Hola {name}, como tutor experto, genera un plan de apoyo personalizado para ti con:"
    else:
        intro = "Como tutor experto, genera un plan de apoyo personalizado para un estudiante con:"

    prompt_lines = [
        intro,
        f"- Nivel de estrés: {row.STRESS_LEVEL}/5",
        f"- Carrera: {career_info}",
        f"- Causas principales: {row.STRESS_CAUSE}",
        "",
        "El plan debe incluir:"
    ]
    prompt_lines += causes
    prompt_lines += [
        "",
        "Formato: Lista clara con acciones concretas.",
        "Tono: Empático pero profesional, evitando jerga técnica."
    ]
    return "\n".join(prompt_lines)

# 5. Llamada a la API de Gemini
def get_ai_response(prompt: str, stream: bool = False):
    if stream:
        return model.generate_content(prompt, stream=True)
    return model.generate_content(prompt)

# 6. Procesamiento principal
def main():
    df = load_student_data("estudiantes_con_estres_y_causas.xlsx")
    required = ["ID", "STUDENT_NAME", "STRESS_LEVEL", "STRESS_CAUSE", "CAREER_PTS"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el Excel: {missing}")

    results = []
    total = len(df)
    for idx, row in df.iterrows():
        # Obtener career_info para el resultado
        career_info = map_career(row.CAREER_PTS)
        print(f"\nProcesando {idx+1}/{total} (ID={row.ID})")
        prompt = build_prompt(row)
        print("Prompt:", prompt.replace("\n", " | "))

        response = get_ai_response(prompt)
        text = getattr(response, "text", str(response))
        print("Respuesta Gemini:", text[:200] + ("..." if len(text) > 200 else ""))

        results.append({
            "ID": row.ID,
            "STUDENT_NAME": row.STUDENT_NAME,
            "STRESS_LEVEL": row.STRESS_LEVEL,
            "STRESS_CAUSE": row.STRESS_CAUSE,
            "CAREER_DESC": career_info,
            "FOLLOW_UP": text
        })

        if (idx+1) % 5 == 0:
            pd.DataFrame(results).to_excel(
                "resultados_parciales.xlsx", index=False, engine="openpyxl"
            )
        time.sleep(1.5)

    pd.DataFrame(results).to_excel(
        "resultados_acompanamiento_final.xlsx", index=False, engine="openpyxl"
    )
    print(f"\n✅ Completado {len(results)}/{total}. Resultados en resultados_acompanamiento_final.xlsx")

if __name__ == "__main__":
    main()
