import streamlit as st
import requests

# Configuración de la página
st.set_page_config(page_title="Traductor IA (Seq2Seq)", page_icon="🌍")

st.title("🌍 Traductor Neuronal: Español a Inglés")
st.markdown("""
Esta aplicación utiliza un modelo **Encoder-Decoder con Atención de Bahdanau** entrenado para traducir oraciones del español al inglés.
""")

# URL de tu API (Ajustar cuando despliegues en Render)
# Cuando pruebes en tu PC, usa 'http://localhost:8000/traducir'
API_URL = "https://api-rnn.onrender.com/traducir"

# Área de texto para el usuario
texto_espanol = st.text_area("Ingresa el texto en español:", placeholder="Ejemplo: ¿Cómo estás hoy?")

if st.button("Traducir"):
    if texto_espanol.strip() == "":
        st.warning("Por favor, ingresa un texto para traducir.")
    else:
        with st.spinner('Traduciendo con el modelo...'):
            try:
                # Petición POST a la API
                response = requests.post(API_URL, json={"texto": texto_espanol})
                
                if response.status_code == 200:
                    data = response.json()
                    traduccion = data.get("traduccion", "")
                    
                    st.success("¡Traducción completada!")
                    st.info(f"**Traducción:** {traduccion}")
                else:
                    st.error(f"Error en la API: {response.json().get('detail', 'Desconocido')}")
            
            except requests.exceptions.ConnectionError:
                st.error("No se pudo conectar con el backend. ¿Está corriendo FastAPI en el puerto 8000?")
