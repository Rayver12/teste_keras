# Projeto de Deteção de Anomalias Visuais com Redes Siamesas

Este projeto implementa um sistema de inspeção visual automatizada para detetar defeitos em objetos, comparando-os com uma imagem de "referência dourada".

O sistema utiliza uma Rede Neural Siamese (baseada na MobileNetV2) para aprender uma representação de características (um "vetor") para pequenas secções (patches) de uma imagem. O modelo é treinado para diferenciar entre pares de patches "iguais" (com pequenas variações) e pares "diferentes" (onde um tem um defeito sintético).

Na inferência, o sistema divide a imagem de referência e a imagem de inspeção numa grelha de patches, calcula os vetores de características para cada um e compara-os. Se a distância (Euclidiana) entre o vetor de um patch de referência e o seu correspondente de inspeção for superior a um limiar, esse patch é marcado como defeituoso.

## Exemplo de Inspeção

O sistema compara uma imagem de referência (esquerda) com uma imagem de teste (direita) e identifica as regiões que diferem significativamente.

<table style="width:100%; border: none;">
  <tr style="border: none;">
    <td style="width: 50%; text-align: center; border: none;"><b>Imagem de Referência</b></td>
    <td style="width: 50%; text-align: center; border: none;"><b>Imagem de Teste (com defeito)</b></td>
  </tr>
  <tr style="border: none;">
    <td style="width: 50%; text-align: center; border: none;"><img src="referencia.jpeg" alt="Imagem de referência" width="300"></td>
    <td style="width: 50%; text-align: center; border: none;"><img src="teste.jpeg" alt="Imagem de teste com defeito" width="300"></td>
  </tr>
</table>

## Estrutura do Projeto

Aqui está uma descrição dos ficheiros Python e dos modelos gerados:

### Scripts Principais

* `create_seamese_dataset.py`:
    * **Função:** Gera o conjunto de dados de treino.
    * **Entrada:** Imagens de referência da pasta `real_references/`.
    * **Saída:** Cria a pasta `data_siamese/` contendo:
        * `images/`: Milhares de pares de imagens (referência e inspeção), onde ~50% dos pares de inspeção têm defeitos sintéticos adicionados.
        * `labels.csv`: Um CSV que mapeia os pares de imagens (`img_ref`, `img_insp`) e a sua etiqueta (0 = match, 1 = defeito).

* `train_seamese.py`:
    * **Função:** Treina o modelo siamês completo.
    * **Entrada:** O conjunto de dados de `data_siamese/`.
    * **Saída:** O modelo siamês treinado e guardado como `best_siamese_model.keras`. Este modelo aceita dois patches de imagem como entrada e calcula a distância entre eles.

* `converter.py`:
    * **Função:** Extrai a "rede base" (o extrator de características) do modelo siamês e converte-a para formatos de inferência.
    * **Entrada:** `best_siamese_model.keras`.
    * **Saída:**
        * `base_network.keras`: O modelo extrator de características (MobileNetV2 modificada) em formato Keras.
        * `base_network_int8.tflite`: O mesmo extrator de características, mas quantizado para INT8 (formato TFLite) para utilização em dispositivos embarcados (como o OpenMV).

* `gerar_referencia.py`:
    * **Função:** Um script utilitário para pré-calcular os vetores de características da sua "imagem de referência dourada" (`referencia.jpeg`) e guardá-los.
    * **Entrada:** `base_network.keras` e `referencia.jpeg`.
    * **Saída:** `ref_vectors_pc.npy`, um ficheiro NumPy que armazena os vetores de características para cada patch da imagem de referência.

* `inspecao_pc.py`:
    * **Função:** O script final de inspeção para PC.
    * **Entrada:**
        * `base_network.keras` (o modelo extrator).
        * `ref_vectors_pc.npy` (os vetores da referência).
        * `teste.jpeg` (a nova imagem a inspecionar).
    * **Saída:** Imprime no terminal quais os patches que falharam (com distância > `THRESHOLD`) e mostra um gráfico matplotlib com o pior patch de referência e de inspeção.

## Fluxo de Trabalho (Como Usar)

Siga estes passos para treinar o modelo e executar a inspeção.

### Passo 1: Configuração do Ambiente

Crie um ambiente virtual e instale as dependências:

```sh
pip install tensorflow opencv-python pandas numpy matplotlib scikit-learn
```
### Passo 2: Preparar Dados de Referência

    Crie uma pasta chamada real_references/.

    Adicione as suas imagens de referência (quanto mais, melhor, ex: 1000+) a esta pasta. O script create_seamese_dataset.py usa estas imagens para gerar os pares de treino.

### Passo 3: Gerar o Conjunto de Dados de Treino
```bash
#Execute o script para criar o dataset. Isto pode demorar algum tempo.


python create_seamese_dataset.py

Isto criará a pasta data_siamese/ com as imagens e o ficheiro labels.csv.
```
### Passo 4: Treinar o Modelo Siamês

```Bash
#Execute o script de treino. Recomenda-se a utilização de uma GPU.

python train_seamese.py

Isto irá treinar o modelo e guardar o melhor checkpoint como best_siamese_model.keras.
```
### Passo 5: Converter o Modelo para Inferência

    O processo de conversão para TFLite INT8 (quantização) requer um pequeno conjunto de dados "representativo".

    Crie uma pasta data_samples/.

    Copie cerca de 100-200 imagens da pasta data_siamese/images/ para dentro dela.

```Bash

mkdir data_samples

# Exemplo (Linux/macOS) para copiar as primeiras 100 imagens:
ls data_siamese/images/ | head -n 100 | xargs -I {} cp data_siamese/images/{} data_samples/
```
Execute o script de conversão:

```Bash
python converter.py
```
Isto gerará os ficheiros base_network.keras (para o PC) e base_network_int8.tflite (para dispositivos embarcados).

### Passo 6: Gerar Vetores da Imagem de Referência

Antes de poder inspecionar, precisa de "memorizar" a sua imagem de referência principal (ex: referencia.jpeg).

Execute o script gerar_referencia.py para pré-calcular e salvar os vetores da imagem de referência.
```Bash
python gerar_referencia.py
```
O script irá carregar o modelo base_network.keras e a imagem referencia.jpeg. Isto criará o ficheiro ref_vectors_pc.npy, que é necessário para a inspeção.

### Passo 7: Executar a Inspeção no PC

Finalmente, execute o script de inspeção principal, apontando para a sua imagem de teste (ex: teste.jpeg).
```Bash
python inspecao_pc.py
```
O script irá carregar o modelo, os vetores de referência (.npy) e a imagem de teste. Ele irá comparar os patches e imprimir um relatório de defeitos no terminal, seguido de um gráfico que mostra o patch com a maior diferença.