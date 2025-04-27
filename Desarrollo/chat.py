#!/usr/bin/env python3
"""
real_time_gemini_chat.py

Inicia un chat en tiempo real con Gemini, utilizando el plan de acompañamiento
extraído del Excel. Antes del chat, genera aleatoriamente tres consejos nuevos para
una encuesta inicial que el usuario debe puntuar de 1 a 5.

Requisitos:
    pip install pandas openpyxl python-dotenv google-generativeai
    Define en tu entorno o en un archivo .env:
        GENAI_API_KEY="TU_API_KEY_DE_GEMINI"
        GEMINI_MODEL_ID="gemini-1.5-flash"
"""

import os
import random
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import protos

# Lista de consejos genéricos para encuesta
DEFAULT_TIPS = [
    "Organiza tu tiempo con un calendario semanal.",
    "Practica técnicas de respiración profunda cuando te sientas abrumado.",
    "Haz pausas activas cada hora de estudio.",
    "Establece metas académicas claras y alcanzables.",
    "Mantén una alimentación balanceada y duerme al menos 7 horas.",
    "Busca apoyo de compañeros o tutores cuando tengas dudas.",
    "Utiliza métodos de estudio activos como flashcards o mapas mentales.",
    "Alterna entre materias difíciles y más livianas para evitar agotamiento.",
    "Reflexiona al final del día sobre tus logros y desafíos.",
    "Haz ejercicio físico regularmente para reducir el estrés."
]


def init_model():
    load_dotenv()
    api_key = "AIzaSyAqzQbReUQu3OJOWZwulhqzyEcZrq7w_Z8"
    model_id = os.getenv("GEMINI_MODEL_ID", "gemini-2.0-flash")
    if not api_key:
        raise RuntimeError("Define GENAI_API_KEY en tu .env o variable de entorno")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_id)


def load_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl")


def select_student(df: pd.DataFrame, student_id: str) -> pd.Series:
    sel = df.loc[df["ID"].astype(str) == student_id]
    if sel.empty:
        raise ValueError(f"No existe estudiante con ID={student_id}")
    return sel.iloc[0]


def conduct_survey(name: str) -> list[tuple[str,int]]:
    """Genera tres consejos aleatorios y pide al usuario que los puntúe."""
    tips = random.sample(DEFAULT_TIPS, 3)
    print(f"\n{name}, antes de iniciar el chat, por favor puntúa estos 3 consejos (1-5):")
    ratings = []
    for tip in tips:
        while True:
            try:
                score = int(input(f"Puntúa: '{tip}' => "))
                if 1 <= score <= 5:
                    ratings.append((tip, score))
                    break
                print("Debes ingresar un número entre 1 y 5.")
            except ValueError:
                print("Entrada inválida. Ingresa un número entero del 1 al 5.")
    print("\n¡Gracias! Ahora comenzaremos el chat.\n")
    return ratings


def start_chat_for_student(model, name: str, follow_up: str):
    # Encuesta inicial
    conduct_survey(name)

    # Sembrar plan en el historial del chat
    init_text = (
        "Eres un tutor universitario experto en orientación estudiantil. "
        f"Este es tu plan de acompañamiento personalizado, {name}: {follow_up}"
    )
    initial_content = protos.Content(
        role="user",
        parts=[protos.Part(text=init_text)]
    )
    chat = model.start_chat(history=[initial_content])

    print(f"Chat iniciado para {name}. Escribe 'exit' para terminar.\n")
    while True:
        user_input = input(f"{name}: ").strip()
        if user_input.lower() in ("exit", "salir"):
            print("Chat terminado.")
            break
        response = chat.send_message(user_input)
        answer = response.text.strip()
        print(f"Gemini: {answer}\n")


def main():
    model = init_model()
    df = load_data("resultados_acompanamiento_final.xlsx")
    student_id = input("Ingresa el ID del estudiante: ").strip()
    try:
        row = select_student(df, student_id)
    except ValueError as e:
        print(e)
        return

    name = row.get("STUDENT_NAME", f"Estudiante {student_id}")
    follow_up = row.get("FOLLOW_UP", "")
    print(f"\n== Plan personalizado para {name} ==\n{follow_up}\n")

    start_chat_for_student(model, name, follow_up)

if __name__ == "__main__":
    main()
