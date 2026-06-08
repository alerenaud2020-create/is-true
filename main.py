import os
import requests
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="is-true API")

# Permitir conexiones desde cualquier origen (esencial para que el frontend funcione)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar cliente de OpenAI (se configurará la API Key en la plataforma de hosting)
client = OpenAI()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

class NewsCheckRequest(BaseModel):
    text: str

def buscar_en_google(query: str) -> str:
    if not SERPER_API_KEY:
        return "No hay acceso a búsquedas web en vivo configurado."
    
    url = "https://google.serper.dev/search"
    payload = {"q": query, "gl": "es", "hl": "es"}
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=5)
        resultados = response.json()
        contexto = ""
        for item in resultados.get("organic", [])[:4]:
            contexto += f"Título: {item.get('title')}\nEnlace: {item.get('link')}\nResumen: {item.get('snippet')}\n\n"
        return contexto if contexto else "No se encontraron noticias recientes."
    except Exception:
        return "Error al consultar fuentes en internet."

# Ruta principal: Muestra la interfaz gráfica al entrar a la app
@app.get("/", response_class=HTMLResponse)
async def home():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return "HTML no encontrado. Asegúrate de tener index.html en la misma carpeta."

# Ruta de procesamiento de la noticia
@app.post("/verify")
async def verify_news(request: NewsCheckRequest):
    texto_usuario = request.text.strip()
    if not texto_usuario:
        raise HTTPException(status_code=400, detail="El texto está vacío.")

    try:
        # 1. Extraer palabras clave para Google
        res_query = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Genera términos de búsqueda eficaces para Google basados en el texto. Sé breve."},
                {"role": "user", "content": texto_usuario}
            ],
            temperature=0.0
        )
        query = res_query.choices[0].message.content.strip()
        
        # 2. Buscar en vivo en la web
        contexto_web = buscar_en_google(query)
        
        # 3. Analizar y dar veredicto
        system_instruction = (
            "Eres un analista de datos de fact-checking. Contrasta el texto del usuario "
            "con la información real de internet provista. Responde estrictamente en formato JSON."
        )
        
        user_prompt = f"""
        TEXTO DEL USUARIO: "{texto_usuario}"
        INFORMACIÓN EN INTERNET:
        {contexto_web}
        
        Devuelve un JSON con:
        - "verdict": "VERDADERO", "FALSO", "ENGAÑOSO" o "FALTA CONTEXTO"
        - "confidence_score": (0 a 100)
        - "explanation": (Explicación clara en español, máximo 3 frases)
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        return json.loads(response.choices[0].message.content)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)}
