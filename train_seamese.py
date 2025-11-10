# CÉLULA 4: Treino (Otimizada com Fine-Tuning e LR Scheduler)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
# --- NOVAS IMPORTAÇÕES ---
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model 
# -------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
import os

# --- Configuração ---
IMG_SIZE = (224, 224)
IMG_SHAPE = (224, 224, 3)
BATCH_SIZE = 64 # Com o fine-tuning, 32 é mais seguro para a VRAM do T4
EPOCHS = 50     # O seu novo total de épocas

DATA_DIR = Path("data_siamese/")
CSV_PATH = DATA_DIR / "labels.csv"
IMG_DIR = DATA_DIR / "images"
#IMG_DIR = Path("/home/rafael/Documentos/desafio02/dataset_treinamento")

BEST_MODEL_SAVE_PATH = "best_siamese_model.keras" 
# ---------------------------

# --- INÍCIO: CÓDIGO DO SCRIPT 2 (siamese_model.py) ---
# (Modificado para Fine-Tuning)

def build_base_network(input_shape):
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    
    # --- OTIMIZAÇÃO: FINE-TUNING ---
    # 1. Permite que o modelo base seja treinável
    base_model.trainable = True
    
    # 2. Vamos congelar as primeiras camadas e treinar apenas as últimas.
    #    O MobileNetV2 tem 154 camadas. Vamos congelar as primeiras 100.
    fine_tune_at_layer = 100
    
    print(f"Congelando todas as camadas antes da camada {fine_tune_at_layer}...")
    for layer in base_model.layers[:fine_tune_at_layer]:
        layer.trainable = False
    # ---------------------------------
        
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    # Adicionamos Dropout para evitar overfitting durante o fine-tuning
    x = layers.Dropout(0.3)(x) 
    x = layers.Dense(256, activation='relu')(x)
    return Model(inputs=base_model.input, outputs=x)

def euclidean_distance(vectors):
    (featsA, featsB) = vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def contrastive_loss(y_true, y_pred, margin=1.0):
    y_true = tf.cast(y_true, 'float32')
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    loss = K.mean(y_true * margin_square + (1 - y_true) * square_pred)
    return loss

def build_siamese_model(input_shape):
    img_a_input = layers.Input(shape=input_shape, name="input_ref")
    img_b_input = layers.Input(shape=input_shape, name="input_insp")
    base_network = build_base_network(input_shape)
    feat_vecs_a = base_network(img_a_input)
    feat_vecs_b = base_network(img_b_input)
    distance = layers.Lambda(euclidean_distance, name="distance")([feat_vecs_a, feat_vecs_b])
    model = Model(inputs=[img_a_input, img_b_input], outputs=distance)
    return model

# --- FIM: CÓDIGO DO SCRIPT 2 ---


# --- INÍCIO: CÓDIGO DO SCRIPT 3 (train_siamese.py) ---

def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    return img

def make_pairs_generator(df):
    def generator():
        df_shuffled = df.sample(frac=1)
        for _, row in df_shuffled.iterrows():
            img_ref = load_image(str(IMG_DIR / row["img_ref"]))
            img_insp = load_image(str(IMG_DIR / row["img_insp"]))
            label = np.array(row["label"], dtype='float32')
            yield ({"input_ref": img_ref, "input_insp": img_insp}, label)
    return generator

def main_train():
    if not os.path.exists(CSV_PATH):
        print(f"ERRO: {CSV_PATH} não encontrado.")
        print("Por favor, execute a CÉLULA 3 primeiro para gerar os dados.")
        return
        
    print("Carregando labels...")
    df = pd.read_csv(CSV_PATH)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print("Criando pipelines de dados (tf.data)...")
    
    output_signature = (
        {"input_ref": tf.TensorSpec(shape=IMG_SHAPE, dtype=tf.float32),
         "input_insp": tf.TensorSpec(shape=IMG_SHAPE, dtype=tf.float32)},
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )

    train_ds = tf.data.Dataset.from_generator(make_pairs_generator(train_df), output_signature=output_signature)
    train_ds = train_ds.repeat().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(make_pairs_generator(val_df), output_signature=output_signature)
    val_ds = val_ds.repeat().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print("Verificando se há um modelo salvo...")

    if os.path.exists(BEST_MODEL_SAVE_PATH):
        # --- CAMINHO 1: CARREGAR MODELO EXISTENTE ---
        print(f"Encontrado! Carregando modelo de: '{BEST_MODEL_SAVE_PATH}'")
        
        # NOTA CRÍTICA: Precisamos informar ao load_model sobre sua loss customizada
        custom_objects = {
            "contrastive_loss": contrastive_loss,
            "euclidean_distance": euclidean_distance  # <-- ADICIONE ESTA LINHA
        } 
        model = load_model(
            BEST_MODEL_SAVE_PATH,
            custom_objects=custom_objects
        )
        print("Modelo carregado com sucesso. Continuando o treino.")
    
    else:
        # --- CAMINHO 2: CONSTRUIR NOVO MODELO (código original) ---
        print("Nenhum modelo salvo encontrado. Construindo um novo...")
        print("Construindo o modelo siamês (com fine-tuning)...")
        model = build_siamese_model(IMG_SHAPE)
        
        # --- OTIMIZAÇÃO: LR MAIS BAIXO PARA FINE-TUNING ---
        print("Configurando otimizador inicial (Adam)...")
        optimizer = Adam(learning_rate=0.00001) # Era 0.0001
        # ----------------------------------------------------
        
        print("Compilando o novo modelo...")
        model.compile(optimizer=optimizer, loss=contrastive_loss)

    
    print("\n--- Sumário do Modelo ---")
    model.summary() # Irá mostrar o modelo (novo ou carregado) 
    # --- DEFINIÇÃO DOS CALLBACKS ---
    
    # 1. Checkpoint (igual ao anterior, salva o melhor)
    checkpoint_callback = ModelCheckpoint(
        filepath=BEST_MODEL_SAVE_PATH,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )
    
    # 2. OTIMIZAÇÃO: LEARNING RATE SCHEDULER
    # Se o 'val_loss' não melhorar por 3 épocas ('patience=3'),
    # reduza a taxa de aprendizagem por um fator de 0.2 (ex: 1e-5 -> 2e-6)
    lr_scheduler_callback = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7, # Não reduzir abaixo disto
        verbose=1
    )
    # -----------------------------------

    print("\nIniciando o treino (com Fine-Tuning)...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=max(1, len(train_df) // BATCH_SIZE),
        validation_steps=max(1, len(val_df) // BATCH_SIZE),
        
        # Passa AMBOS os callbacks
        callbacks=[checkpoint_callback, lr_scheduler_callback]
    )
    
    print(f"\nTreino concluído. O melhor modelo está salvo em '{BEST_MODEL_SAVE_PATH}'.")

# Executa o Script 3
main_train()
# --- FIM: CÓDIGO DO SCRIPT 3 ---
