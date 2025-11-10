# SCRIPT 1: create_siamese_dataset.py
#
# Modifica o seu 'create_robust_dataset.py'
# para criar PARES de imagens e um ficheiro 'labels.csv'
#

import os
import cv2
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# --- Configuração ---
# 1. Coloque as suas 1096 imagens de referência aqui
SOURCE_REF_DIR = Path("real_references/") 

# 2. Onde o dataset siamês será salvo
OUTPUT_DIR = Path("data_siamese/")
IMG_DIR = OUTPUT_DIR / "images"

# 3. Parâmetros
NUM_PAIRS_PER_REF = 20   # Quantos pares gerar para CADA imagem de referência
IMG_SIZE = (224, 224)    # Tamanho padrão para redes neurais (diferente do 504)
DEFECT_PROB = 0.5        # 50% de chance de criar um par com defeito
# --------------------

IMG_DIR.mkdir(parents=True, exist_ok=True)

# --- Funções de Augmentação (Reutilizadas do seu script) ---
#
def apply_light_augmentation(img):
    """Aplica leves variações de rotação, escala e brilho."""
    rows, cols, _ = img.shape
    
    # Rotação e Escala
    angle = random.uniform(-2.0, 2.0)
    scale = random.uniform(0.98, 1.02)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)
    
    # Translação
    M[0, 2] += random.uniform(-5, 5) # shift x
    M[1, 2] += random.uniform(-5, 5) # shift y
    
    augmented = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
    
    # Brilho
    hsv = cv2.cvtColor(augmented, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, random.randint(-10, 10))
    v = np.clip(v, 0, 255)
    final_hsv = cv2.merge((h, s, v))
    augmented = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    
    return augmented

def add_synthetic_defect(img):
    """Adiciona um defeito 'copy-paste' ou 'cutout'."""
    #
    h, w, _ = img.shape
    defect_type = random.choice(["paste", "cutout", "stain"])
    
    dw = random.randint(20, 80)
    dh = random.randint(20, 80)
    x1 = random.randint(0, w - dw)
    y1 = random.randint(0, h - dh)
    
    if defect_type == "paste":
        sx1 = random.randint(0, w - dw)
        sy1 = random.randint(0, h - dh)
        patch = img[sy1:sy1+dh, sx1:sx1+dw].copy()
        patch = cv2.GaussianBlur(patch, (5, 5), 0)
        img[y1:y1+dh, x1:x1+dw] = patch
        
    elif defect_type == "cutout":
        img[y1:y1+dh, x1:x1+dw] = (0, 0, 0)
        
    elif defect_type == "stain":
        color = [random.randint(0, 255) for _ in range(3)]
        overlay = np.full((dh, dw, 3), color, dtype=np.uint8)
        patch = img[y1:y1+dh, x1:x1+dw]
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, patch, 1 - alpha, 0, patch)
        img[y1:y1+dh, x1:x1+dw] = patch

    return img
# --- Fim das Funções Reutilizadas ---

def main():
    source_images = list(SOURCE_REF_DIR.glob("*.jpg")) + list(SOURCE_REF_DIR.glob("*.png"))
    if not source_images:
        print(f"Erro: Nenhuma imagem encontrada em {SOURCE_REF_DIR.resolve()}")
        return

    generated_pairs = [] # Lista para guardar os dados do CSV

    print(f"Gerando dataset de pares a partir de {len(source_images)} imagens...")

    for ref_base_path in tqdm(source_images, desc="Processando refs"):
        try:
            base_img = cv2.imread(str(ref_base_path))
            base_img = cv2.resize(base_img, IMG_SIZE)
        except Exception as e:
            print(f"Aviso: Falha ao ler {ref_base_path}. Pulando. Erro: {e}")
            continue
            
        for i in range(NUM_PAIRS_PER_REF):
            
            # --- Cria o par ---
            ref_aug = apply_light_augmentation(base_img.copy())
            
            # Decide se este par será "bom" (match) ou "ruim" (defeito)
            if random.random() < DEFECT_PROB:
                # 1 = DEFEITO (não são iguais)
                label = 1
                insp_aug = apply_light_augmentation(base_img.copy())
                insp_aug = add_synthetic_defect(insp_aug)
            else:
                # 0 = MATCH (são iguais, apenas com aug leve)
                label = 0
                insp_aug = apply_light_augmentation(base_img.copy())

            # Define nomes de ficheiro únicos
            base_name = f"{ref_base_path.stem}_{i:03d}"
            ref_filename = f"{base_name}_ref.jpg"
            insp_filename = f"{base_name}_insp.jpg"
            
            # Salva as imagens na *mesma* pasta
            cv2.imwrite(str(IMG_DIR / ref_filename), ref_aug)
            cv2.imwrite(str(IMG_DIR / insp_filename), insp_aug)
            
            # Adiciona ao nosso dataframe de labels
            generated_pairs.append({
                "img_ref": ref_filename,
                "img_insp": insp_filename,
                "label": label # 0 = match, 1 = defect
            })

    # Salva o CSV final
    csv_path = OUTPUT_DIR / "labels.csv"
    df = pd.DataFrame(generated_pairs)
    df = df.sample(frac=1).reset_index(drop=True) # Embaralha
    df.to_csv(csv_path, index=False)
        
    print("-" * 30)
    print(f"Dataset Siamês criado com sucesso!")
    print(f"  Imagens:  {IMG_DIR.resolve()}")
    print(f"  Labels:   {csv_path.resolve()}")
    print(f"  Total de Pares: {len(df)}")
    print("-" * 30)

if __name__ == "__main__":
    main()
