import tensorflow as tf
import numpy as np
import cv2
import os
import time

print("--- Gerando Gabarito (Versão TFLite) ---")

# --- Configurações ---
TFLITE_MODEL_PATH = "model_quantized.tflite" # Usamos o novo modelo
PATH_IMG_REFERENCIA = "Detector-em-imagem-web3.png" 
OUTPUT_FILE = "ref_vectors_tflite.npy" # Novo nome para não confundir
MODEL_PATCH_SIZE = (224, 224) 
GRID_SIZE = (7, 8) 

def create_patches(image, grid_size, patch_img_size):
    num_rows, num_cols = grid_size
    patch_h, patch_w = patch_img_size
    full_h = num_rows * patch_h
    full_w = num_cols * patch_w
    resized_image = cv2.resize(image, (full_w, full_h))
    patches = []
    for h_patch in np.vsplit(resized_image, num_rows):
        for w_patch in np.hsplit(h_patch, num_cols):
            patches.append(w_patch)
    return patches

# 1. Carregar TFLite
print(f"Carregando TFLite: {TFLITE_MODEL_PATH}")
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 2. Carregar Imagem
img_ref_original = cv2.imread(PATH_IMG_REFERENCIA)
if img_ref_original is None:
    print("ERRO: Imagem de referência não encontrada.")
    exit()

patches = create_patches(img_ref_original, GRID_SIZE, MODEL_PATCH_SIZE)
batch_input = np.array([p / 255.0 for p in patches], dtype=np.float32)

# 3. Redimensionar o Tensor para processar TUDO de uma vez
# Isso evita o loop for lento do Python
num_patches = len(batch_input)
interpreter.resize_tensor_input(input_details[0]['index'], [num_patches, 224, 224, 3])
interpreter.allocate_tensors() # Aloca memória RAM necessária

print(f"Processando {num_patches} patches de referência...")
start_time = time.time()

# 4. Inferência
interpreter.set_tensor(input_details[0]['index'], batch_input)
interpreter.invoke()
ref_vectors = interpreter.get_tensor(output_details[0]['index'])

print(f"Gabarito gerado em {time.time() - start_time:.2f}s")
np.save(OUTPUT_FILE, ref_vectors)
print(f"Salvo em: {OUTPUT_FILE}")
