import streamlit as st
import requests

# Configuración de la página
st.set_page_config(page_title="Traductor/Generador RNN", layout="centered")

st.title("🤖 Interfaz RNN: Encoder-Decoder")
st.write("Ingresa el texto para ser procesado por el modelo.")

# -------------------------------------------------------------------
# CONFIGURACIÓN DE RED
# Cambia esta URL por la que te dé Render cuando subas la API
# -------------------------------------------------------------------
API_URL = "http://localhost:8000/predict" 

# Formulario de entrada
texto_usuario = st.text_area("Texto de entrada:", height=100)

if st.button("Generar Respuesta", type="primary"):
    if texto_usuario.strip() == "":
        st.warning("⚠️ Por favor, ingresa algún texto antes de procesar.")
    else:
        with st.spinner("Procesando en el backend..."):
            try:
                # Hacer la petición POST a FastAPI
                payload = {"text": texto_usuario}
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    datos = response.json()
                    st.success("¡Completado!")
                    
                    st.subheader("Resultado:")
                    st.info(datos["prediction"])
                else:
                    st.error(f"Error en la API (Código {response.status_code}): {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ No se pudo conectar con el backend. Asegúrate de que FastAPI esté corriendo (uvicorn api.main:app --reload).")
            except Exception as e:
                st.error(f"❌ Ocurrió un error inesperado: {e}")
