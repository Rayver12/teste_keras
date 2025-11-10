import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import time

print("--- Script de Geração de Gabarito (PC) ---")

# --- Configurações ---
BASE_MODEL_PATH = "base_network.keras" # O modelo que o converter.py criou
PATH_IMG_REFERENCIA = "referencia.jpeg" 

OUTPUT_FILE = "ref_vectors_pc.npy" # Arquivo de saída

MODEL_PATCH_SIZE = (224, 224) 
GRID_SIZE = (7, 8) 

# --- Funções (Idênticas aos scripts anteriores) ---

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

# --- Lógica Principal ---

print(f"Carregando rede base de: {BASE_MODEL_PATH}")
tf.get_logger().setLevel('ERROR')
model = load_model(BASE_MODEL_PATH) # Carrega a rede base
print("Modelo carregado.")

print(f"Carregando imagem de referência: {PATH_IMG_REFERENCIA}...")
img_ref_original = cv2.imread(PATH_IMG_REFERENCIA)
if img_ref_original is None:
    print("ERRO: Imagem de referência não encontrada.")
    exit()

print("Criando patches de referência...")
ref_patches_list = create_patches(img_ref_original, GRID_SIZE, MODEL_PATCH_SIZE)

# Cria o lote (batch) de referência
ref_batch = np.array([p / 255.0 for p in ref_patches_list]).astype(np.float32)

print(f"Processando {len(ref_batch)} patches (usando múltiplos núcleos)...")
start_time = time.time()

# Gera os vetores de features (float32)
# batch_size=8 para usar seus núcleos
ref_vectors = model.predict(ref_batch, batch_size=8, verbose=1)

print(f"Gabarito gerado em {time.time() - start_time:.2f}s")

# Salva os vetores em um arquivo .npy
np.save(OUTPUT_FILE, ref_vectors)

print("\n" + "="*30)
print("🎉 GABARITO DE REFERÊNCIA DO PC CRIADO! 🎉")
print(f"Vetor[{ref_vectors.shape}] salvo em: {OUTPUT_FILE}")
print("="*30)
