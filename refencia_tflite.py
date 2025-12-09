import tensorflow as tf
import os

# --- Configurações ---
BASE_MODEL_PATH = "base_network.keras"
TFLITE_MODEL_FILE = "model_quantized.tflite"

def converter_modelo():
    if not os.path.exists(BASE_MODEL_PATH):
        print(f"ERRO: Não encontrei o arquivo {BASE_MODEL_PATH}")
        return

    print(f"Carregando modelo original: {BASE_MODEL_PATH}")
    model = tf.keras.models.load_model(BASE_MODEL_PATH)

    # Converter para TFLite
    print("Iniciando conversão para TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # --- OTIMIZAÇÕES ---
    # Aplica otimizações padrão (fusão de camadas, etc)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Opcional: Use float16 para reduzir tamanho pela metade com perda mínima
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    # Salvar o arquivo
    with open(TFLITE_MODEL_FILE, "wb") as f:
        f.write(tflite_model)

    print(f"\nSucesso! Modelo salvo como: {TFLITE_MODEL_FILE}")
    print(f"Tamanho original: {os.path.getsize(BASE_MODEL_PATH) / 1024:.2f} KB")
    print(f"Tamanho TFLite:   {os.path.getsize(TFLITE_MODEL_FILE) / 1024:.2f} KB")

if __name__ == "__main__":
    converter_modelo()
