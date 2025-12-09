import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import time

print("--- Script de Inspeção OTIMIZADO (TFLite) ---")

# --- 1. Configurações ---
TFLITE_MODEL_PATH = "model_quantized.tflite"
REF_VECTORS_PATH = "ref_vectors_tflite.npy" # Usa o gabarito novo
PATH_IMG_INSPECAO = "Detector-em-imagem-web5.png"
PATH_IMG_REFERENCIA = "Detector-em-imagem-web3.png" 
THRESHOLD = 0.08 
MODEL_PATCH_SIZE = (224, 224) 
GRID_SIZE = (7, 8) 

# --- 2. Função de Criação de Patches ---
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

# --- 3. Carregamento do Motor TFLite (Muito mais leve) ---
print("Inicializando Interpreter TFLite...")
if not os.path.exists(TFLITE_MODEL_PATH):
    print("ERRO: Modelo TFLite não encontrado. Rode o 'Passo 1'.")
    exit()

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Pré-alocação: Ajustamos o tensor para receber TODOS os patches de uma vez (Batching)
total_patches = GRID_SIZE[0] * GRID_SIZE[1] # 56
interpreter.resize_tensor_input(input_details[0]['index'], [total_patches, 224, 224, 3])
interpreter.allocate_tensors()
print("Motor TFLite pronto.")

# --- 4. Carregar Gabarito ---
if not os.path.exists(REF_VECTORS_PATH):
    print("ERRO: Gabarito TFLite não encontrado. Rode o 'Passo 2'.")
    exit()
ref_matrix = np.load(REF_VECTORS_PATH)

# --- 5. Lógica Principal (Produção) ---

# Carregar imagem
img_insp_original = cv2.imread(PATH_IMG_INSPECAO)
if img_insp_original is None:
    print("ERRO: Imagem de inspeção não encontrada.")
    exit()

# Prepara dados
insp_patches_list = create_patches(img_insp_original, GRID_SIZE, MODEL_PATCH_SIZE)
insp_batch = np.array([p / 255.0 for p in insp_patches_list], dtype=np.float32)

print(f"Processando {len(insp_batch)} patches...")

# --- MEDIÇÃO DE TEMPO CRÍTICA ---
start_time = time.time()

# 1. Enviar dados para o TFLite
interpreter.set_tensor(input_details[0]['index'], insp_batch)

# 2. Rodar a rede (C++ otimizado)
interpreter.invoke()

# 3. Pegar resultados
insp_vectors = interpreter.get_tensor(output_details[0]['index'])

inspection_time = time.time() - start_time
# --------------------------------

print(f"Inspeção concluída em {inspection_time:.4f}s") # Ex: 0.15s

# Cálculo de distâncias (NumPy puro)
distances = np.linalg.norm(ref_matrix - insp_vectors, axis=1)

# --- 6. Análise e Visualização ---
max_distance = np.max(distances)
max_dist_index = np.argmax(distances)

veredito_label = 1 if max_distance > THRESHOLD else 0
veredito_str = "DEFEITO DETECTADO" if veredito_label == 1 else "Bom (Sem Defeito)"
cor_titulo = 'red' if veredito_label == 1 else 'green'

print(f"Distância MÁXIMA: {max_distance:.4f}")
print(f"VEREDITO: {veredito_str}")

# Plotagem
img_ref_original = cv2.imread(PATH_IMG_REFERENCIA)
ref_patches_list = create_patches(img_ref_original, GRID_SIZE, MODEL_PATCH_SIZE)
worst_ref_patch = ref_patches_list[max_dist_index]
worst_insp_patch = insp_patches_list[max_dist_index]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(cv2.cvtColor(worst_ref_patch, cv2.COLOR_BGR2RGB))
ax1.set_title(f"Ref (Q{max_dist_index+1})")
ax1.axis('off')
ax2.imshow(cv2.cvtColor(worst_insp_patch, cv2.COLOR_BGR2RGB))
ax2.set_title(f"Insp (Q{max_dist_index+1})")
ax2.axis('off')

fig.suptitle(f"Veredito: {veredito_str}\nDistância: {max_distance:.4f} | Tempo: {inspection_time:.3f}s", 
             fontsize=14, color=cor_titulo, fontweight='bold')

plt.show()
