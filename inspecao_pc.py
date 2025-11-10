import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import time

print("--- Script de Inspeção Otimizado (PC) ---")

# --- 1. Configurações ---
BASE_MODEL_PATH = "base_network.keras"
REF_VECTORS_PATH = "ref_vectors_pc.npy"
PATH_IMG_INSPECAO = "teste.jpeg"
PATH_IMG_REFERENCIA = "referencia.jpeg" 
THRESHOLD = 0.08 
MODEL_PATCH_SIZE = (224, 224) 
# Certifique-se que o GRID_SIZE aqui é o mesmo usado
# para gerar o 'ref_vectors_pc.npy'
GRID_SIZE = (7, 8) # Ex: (7,8) = 56, (8,8) = 64
BATCH_SIZE = 8 

# --- 2. Função de Criação de Patches ---

def create_patches(image, grid_size, patch_img_size):
    """Divide uma imagem (numpy array) numa grelha de patches."""
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

# --- 3. Lógica Principal (Produção) ---

# ETAPA 1: Carregar o modelo e o gabarito
print(f"Carregando rede base de: {BASE_MODEL_PATH}")
tf.get_logger().setLevel('ERROR')
model = load_model(BASE_MODEL_PATH)
print("Modelo carregado.")

print(f"Carregando gabarito de referência de: {REF_VECTORS_PATH}...")
if not os.path.exists(REF_VECTORS_PATH):
    print("ERRO: Gabarito não encontrado. Rode 'gerar_referencia_pc.py' primeiro.")
    exit()
ref_matrix = np.load(REF_VECTORS_PATH)

# ETAPA 2: Carregar e processar a imagem de inspeção
print(f"Carregando imagem de inspeção: {PATH_IMG_INSPECAO}...")
img_insp_original = cv2.imread(PATH_IMG_INSPECAO)
if img_insp_original is None:
    print("ERRO: Imagem de inspeção não encontrada.")
    exit()

print("Criando patches de inspeção...")
insp_patches_list = create_patches(img_insp_original, GRID_SIZE, MODEL_PATCH_SIZE)
insp_batch = np.array([p / 255.0 for p in insp_patches_list]).astype(np.float32)

# Verifica se o grid do gabarito e da inspeção batem
if ref_matrix.shape[0] != len(insp_batch):
    print(f"ERRO DE DIMENSÃO: O gabarito tem {ref_matrix.shape[0]} patches,")
    print(f"mas a inspeção atual tem {len(insp_batch)} patches.")
    print("Por favor, re-gere o gabarito com o GRID_SIZE correto.")
    exit()

print(f"Processando {len(insp_batch)} patches de inspeção (usando múltiplos núcleos)...")
start_time = time.time()

insp_vectors = model.predict(insp_batch, batch_size=BATCH_SIZE, verbose=1)

# ----------------------------------------------------
# --- MUDANÇA 1: Capturar o tempo de inspeção ---
# ----------------------------------------------------
# Armazena o tempo em uma variável
inspection_time = time.time() - start_time
print(f"Inspeção concluída em {inspection_time:.2f}s")
# ----------------------------------------------------

# ETAPA 3: Calcular distâncias (em Python/NumPy, super rápido)
print("Calculando distâncias...")
distances = np.linalg.norm(ref_matrix - insp_vectors, axis=1)

# --- 4. Análise dos Resultados ---
# (Idêntico ao anterior)
max_distance = np.max(distances)
min_distance = np.min(distances)
avg_distance = np.mean(distances)
max_dist_index = np.argmax(distances)

veredito_label = 1 if max_distance > THRESHOLD else 0
veredito_str = "DEFEITO DETECTADO" if veredito_label == 1 else "Bom (Sem Defeito)"
cor_titulo = 'red' if veredito_label == 1 else 'green'

print("\n" + "="*40)
print("--- RESULTADO DA COMPARAÇÃO (OTIMIZADO) ---")
print(f"Distâncias (Float32):")
print(np.round(distances, 4)) 
print("-" * 40)
print(f"Distância MÍNIMA:     {min_distance:.4f}")
print(f"Distância MÉDIA:      {avg_distance:.4f}")
print(f"Distância MÁXIMA:     {max_distance:.4f} (no quadrante {max_dist_index + 1})")
print(f"Limiar (Threshold):   {THRESHOLD}")
print(f"\nVEREDITO FINAL:     {veredito_str}")
print("="*40 + "\n")

# --- 5. Visualização (Mostra o Pior Quadrante) ---
print("Exibindo o quadrante com a MAIOR diferença...")

img_ref_original = cv2.imread(PATH_IMG_REFERENCIA)
if img_ref_original is None:
    worst_ref_patch = np.zeros((MODEL_PATCH_SIZE[0], MODEL_PATCH_SIZE[1], 3), dtype=np.uint8)
else:
    ref_patches_list = create_patches(img_ref_original, GRID_SIZE, MODEL_PATCH_SIZE)
    # Verifica se o índice é válido (caso o grid da ref no disco seja diferente)
    if max_dist_index < len(ref_patches_list):
        worst_ref_patch = ref_patches_list[max_dist_index]
    else:
        worst_ref_patch = np.zeros((MODEL_PATCH_SIZE[0], MODEL_PATCH_SIZE[1], 3), dtype=np.uint8)

worst_insp_patch = insp_patches_list[max_dist_index]

worst_ref_patch_rgb = cv2.cvtColor(worst_ref_patch, cv2.COLOR_BGR2RGB)
worst_insp_patch_rgb = cv2.cvtColor(worst_insp_patch, cv2.COLOR_BGR2RGB)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(worst_ref_patch_rgb)
ax1.set_title(f"Quadrante {max_dist_index + 1} (Referência)")
ax1.axis('off')
ax2.imshow(worst_insp_patch_rgb)
ax2.set_title(f"Quadrante {max_dist_index + 1} (Inspeção)")
ax2.axis('off')

# ------------------------------------------------------------
# --- MUDANÇA 2: Adicionar a variável 'inspection_time' ---
# ------------------------------------------------------------
fig.suptitle(f"Veredito: {veredito_str} | Pior Distância: {max_distance:.4f} | Tempo: {inspection_time:.2f}s", 
             fontsize=16, 
             color=cor_titulo,
             fontweight='bold')
# ------------------------------------------------------------

plt.show()
