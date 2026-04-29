from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import re
import os
import glob

app = FastAPI(title="API de Traducción Español-Inglés (Seq2Seq)")

# --- HIPERPARÁMETROS (Mismos que en el entrenamiento) ---
UNITS = 512
EMBEDDING_DIM = 256
BATCH_SIZE = 64
MAX_EXAMPLES = 100000

# Variables globales para los tokenizadores y longitudes
inp_lang = None
targ_lang = None
max_length_targ = None
max_length_inp = None
encoder = None
decoder = None

# --- 1. DEFINICIÓN DE LA ARQUITECTURA ---
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(attention_weights * values, axis=1)
        return context_vector, attention_weights

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super().__init__()
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
    def call(self, x, hidden):
        x = self.embedding(x)
        return self.gru(x, initial_state=hidden)

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super().__init__()
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)
    def call(self, x, hidden, enc_output):
        context_vector, _ = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        return self.fc(tf.reshape(output, (-1, output.shape[2]))), state

# --- 2. FUNCIONES DE PREPROCESAMIENTO ---
def preprocess_sentence(w):
    w = w.lower().strip()
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" ]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.strip()
    return f'<start> {w} <end>'

def load_dataset(path, num_examples):
    lines = open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')[:2]] for l in lines[:num_examples]]
    return zip(*word_pairs)

def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer

# --- 3. INICIALIZACIÓN DEL SERVIDOR ---
@app.on_event("startup")
async def startup_event():
    global inp_lang, targ_lang, max_length_inp, max_length_targ, encoder, decoder
    print("Descargando y procesando dataset para reconstruir tokenizadores...")
    
    path_to_zip = tf.keras.utils.get_file(
        'spa-eng.zip', 
        origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip', 
        extract=True
    )
    path_to_file = glob.glob(os.path.join(os.path.dirname(path_to_zip), '**/spa.txt'), recursive=True)[0]
    
    en_raw, sp_raw = load_dataset(path_to_file, MAX_EXAMPLES)
    input_tensor, inp_lang = tokenize(sp_raw)
    target_tensor, targ_lang = tokenize(en_raw)
    
    max_length_inp = input_tensor.shape[1]
    max_length_targ = target_tensor.shape[1]
    
    print("Instanciando modelos...")
    vocab_inp_size = len(inp_lang.word_index) + 1
    vocab_tar_size = len(targ_lang.word_index) + 1
    
    encoder = Encoder(vocab_inp_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
    decoder = Decoder(vocab_tar_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
    
    # Hacer una pasada dummy para inicializar las variables antes de cargar pesos
    dummy_inp = tf.zeros((1, max_length_inp))
    dummy_hidden = [tf.zeros((1, UNITS))]
    enc_out, enc_hidden = encoder(dummy_inp, dummy_hidden)
    dummy_dec_inp = tf.expand_dims([targ_lang.word_index['<start>']], 0)
    decoder(dummy_dec_inp, enc_hidden, enc_out)
    
    print("Cargando pesos (.h5)...")
    # Asegúrate de que las rutas relativas apunten a la carpeta 'modelos'
    encoder.load_weights('./modelos/encoder_weights.weights.h5')
    decoder.load_weights('./modelos/decoder_weights.weights.h5')
    print("¡Modelo cargado y listo para traducir!")

# --- 4. ENDPOINT DE LA API ---
class TranslationRequest(BaseModel):
    texto: str

@app.post("/traducir")
async def translate_api(request: TranslationRequest):
    if not request.texto:
        raise HTTPException(status_code=400, detail="El texto está vacío")
        
    sentence = preprocess_sentence(request.texto)
    
    # Filtrar palabras que el modelo no conoce para evitar errores
    inputs = []
    for i in sentence.split(' '):
        if i in inp_lang.word_index:
            inputs.append(inp_lang.word_index[i])
        else:
            # Ignorar palabras desconocidas (o podrías usar un token <UNK> si lo tuvieras)
            pass
            
    if not inputs:
         raise HTTPException(status_code=400, detail="El modelo no reconoce ninguna palabra de esta frase.")

    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    
    result = ''
    hidden = [tf.zeros((1, UNITS))]
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)
    
    for t in range(max_length_targ):
        predictions, dec_hidden = decoder(dec_input, dec_hidden, enc_out)
        predicted_id = tf.argmax(predictions[0]).numpy()
        word = targ_lang.index_word.get(predicted_id, '')
        
        if word == '<end>': 
            break
            
        result += word + ' '
        dec_input = tf.expand_dims([predicted_id], 0)
        
    return {"original": request.texto, "traduccion": result.strip()}
