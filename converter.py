import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import os
import traceback
import cv2  # <--- NOVA IMPORTAÇÃO
import glob # <--- NOVA IMPORTAÇÃO
import numpy as np # <--- NOVA IMPORTAÇÃO

print("--- 1. Definições Customizadas (para carregar) ---")

def euclidean_distance(vectors):
    x, y = vectors
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def contrastive_loss(y_true, y_pred, margin=1.0):
    y_true = tf.cast(y_true, 'float32')
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    loss = K.mean(y_true * margin_square + (1 - y_true) * square_pred)
    return loss

MODEL_PATH = "best_siamese_model.keras"
BASE_MODEL_SAVE_PATH = "base_network.keras"
TFLITE_MODEL_PATH = "base_network_int8.tflite" # <-- Novo nome do arquivo
SAMPLE_DIR = "data_samples"
MODEL_PATCH_SIZE = (224, 224)

# --- NOVA FUNÇÃO: Gerador de Dataset Representativo ---
def representative_dataset_gen():
    """Gera dados de amostra para quantização."""
    print(f"Carregando dados de amostra de '{SAMPLE_DIR}'...")
    image_paths = glob.glob(os.path.join(SAMPLE_DIR, '*.[jp][pn]g')) # Pega .jpg, .png
    if not image_paths:
        print(f"ERRO: Nenhum arquivo .jpg ou .png encontrado em '{SAMPLE_DIR}'.")
        print("Por favor, adicione 50-100 imagens de amostra e tente novamente.")
        return

    # Limita a 100 amostras para velocidade
    for img_path in image_paths[:100]:
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img = cv2.resize(img, MODEL_PATCH_SIZE)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Pré-processamento FLOAT32 (como o modelo foi treinado)
        img_normalized = (img_rgb / 255.0).astype(np.float32)
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # 'yield' fornece um item de cada vez
        yield [img_batch]

# --- Bloco 2 e 3 (Extração do Modelo) ---
# (Idêntico ao anterior, apenas verifica se o .keras já existe)

if not os.path.exists(BASE_MODEL_SAVE_PATH):
    print(f"Arquivo '{BASE_MODEL_SAVE_PATH}' não encontrado. Gerando agora...")
    if not os.path.exists(MODEL_PATH):
        print(f"ERRO: Modelo '{MODEL_PATH}' não encontrado. Abortando.")
    else:
        print(f"--- 2. Carregando Modelo Siamês Completo de '{MODEL_PATH}' ---")
        full_siamese_model = load_model(
            MODEL_PATH, 
            custom_objects={
                "contrastive_loss": contrastive_loss,
                "euclidean_distance": euclidean_distance 
            }
        )
        print("--- 3. Extraindo Rede Base (do índice [2]) ---")
        try:
            base_network = full_siamese_model.layers[2]
            base_network.save(BASE_MODEL_SAVE_PATH)
            print(f"Rede base salva em '{BASE_MODEL_SAVE_PATH}'")
        except Exception as e:
            print(f"ERRO: Falha ao extrair ou salvar a rede base: {e}")
            exit()
else:
    print(f"Usando '{BASE_MODEL_SAVE_PATH}' existente.")

# --- 4. CONVERSÃO TFLITE (Quantização INT8) ---
print("\n--- 4. Convertendo Rede Base para TFLite (INT8) ---")

try:
    print(f"[Debug] Carregando a rede base de '{BASE_MODEL_SAVE_PATH}'...")
    loaded_base_network = load_model(BASE_MODEL_SAVE_PATH)
    
    print("[Debug] Carregando o conversor...")
    converter = tf.lite.TFLiteConverter.from_keras_model(loaded_base_network)
    
    print("[Debug] Configurando otimizações (DEFAULT)...")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    print("[Debug] Fornecendo o dataset representativo...")
    converter.representative_dataset = representative_dataset_gen
    
    print("[Debug] Forçando apenas operações INT8 (para máxima compatibilidade)...")
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # Define os tipos de entrada e saída do modelo TFLite como INT8
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    print("[Debug] Iniciando converter.convert()... (Isto PODE demorar)")
    tflite_model = converter.convert()
    print("[Debug] Conversão INT8 CONCLUÍDA.")

    print(f"[Debug] Salvando modelo em '{TFLITE_MODEL_PATH}'...")
    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)
    
    print("\n" + "="*30)
    print("🎉 CONVERSÃO INT8 CONCLUÍDA 🎉")
    print(f"Modelo TFLite salvo em: '{TFLITE_MODEL_PATH}'")
    print(f"Tamanho do arquivo: {os.path.getsize(TFLITE_MODEL_PATH) / 1024:.2f} KB")
    print("Copie este NOVO arquivo para a sua placa Labrador!")
    print("="*30)

except Exception as e:
    print("\n" + "!"*30)
    print("ERRO DURANTE A CONVERSÃO TFLITE")
    print(e)
    traceback.print_exc()
    print("!"*30)
