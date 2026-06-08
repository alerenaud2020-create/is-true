import os
import requests
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from google import genai

# Inicializamos el servidor con el nombre oficial de tu app
app = FastAPI(title="Is-True API - Buscador de la Verdad")

# Permitir conexiones desde la interfaz gráfica de Is-True
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializamos el cliente de Google Gemini (requiere GEMINI_API_KEY en Render)
client = genai.Client()
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

@app.get("/", response_class=HTMLResponse)
async def home():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return "HTML no encontrado. Asegúrate de tener index.html en la misma carpeta."

@app.post("/verify")
async def verify_news(request: NewsCheckRequest):
    texto_usuario = request.text.strip()
    if not texto_usuario:
        raise HTTPException(status_code=400, detail="El texto está vacío.")

    try:
        # 1. Is-True extrae las palabras clave con Gemini
        prompt_query = f"Genera términos de búsqueda eficaces para Google basados en este texto. Sé muy breve y directo: {texto_usuario}"
        res_query = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt_query,
        )
        query = res_query.text.strip()
        
        # 2. Is-True busca pruebas en internet en tiempo real
        contexto_web = buscar_en_google(query)
        
        # 3. Veredicto final de Is-True
        user_prompt = f"""
        Eres el analista principal de 'Is-True', una aplicación avanzada de fact-checking. 
        Contrasta el texto del usuario con la información real de internet provista para determinar si es verdadero o falso.
        
        TEXTO DEL USUARIO: "{texto_usuario}"
        
        INFORMACIÓN REAL EN INTERNET:
        {contexto_web}
        
        Debes responder ESTRICTAMENTE con un objeto JSON que tenga exactamente esta estructura:
        {{
          "verdict": "VERDADERO" o "FALSO" o "ENGAÑOSO" o "FALTA CONTEXTO",
          "confidence_score": (un número de 0 a 100),
          "explanation": "Una explicación clara, objetiva y en español de máximo 3 frases explicando por qué Is-True llegó a este veredicto."
        }}
        """
        
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=user_prompt,
            config={
                'response_mime_type': 'application/json'
            }
        )
        
        return json.loads(response.text.strip())
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
