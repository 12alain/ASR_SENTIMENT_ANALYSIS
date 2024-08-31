#  Projet de  Reconnaissance Automatique de la Parole (ASR) et l'Analyse de Sentiment

## Partie 1 : Construire un modèle de Reconnaissance Automatique de la Parole (ASR)

### 1. Choix et Configuration du Modèle ASR

1. **Choisir le modèle ASR**
   - Nous utiliserons le modèle `facebook/wav2vec2-large-xlsr-53-french` pour la reconnaissance automatique de la parole en français. Vous pouvez trouver ce modèle sur Hugging Face [ici](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-french).

2. **Installation des bibliothèques nécessaires**
   - Installez les bibliothèques requises avec les commandes suivantes :
     ```python
     !pip install huggingsound 
     !pip install --upgrade torch torchvision
     !pip install ipython
     !pip install ipywebrtc
     
     ```

3. **Code pour la Reconnaissance Automatique de la Parole**
   - Utilisez le code suivant pour transcrire des enregistrements audio en texte :
   **fonction pour enregistrer un audio directement** 
    ```python
     from ipywebrtc import AudioRecorder, CameraStream
     from google.colab import output

     def create_audio_recorder():
         # Activer le gestionnaire de widgets pour afficher le widget dans Google Colab
         output.enable_custom_widget_manager()

         # Initialiser la caméra avec uniquement l'audio
         camera = CameraStream(constraints={'audio': True, 'video': False})

         # Créer un enregistreur audio à partir du flux de la caméra
         recorder = AudioRecorder(stream=camera)
         
         return recorder

     # Utiliser la fonction pour créer un enregistreur audio
     recorder = create_audio_recorder()
     recorder 

     # Convertir l'audio en wav et l'enregistrer localement 
    with open('recording.webm', 'wb') as f:
    f.write(recorder.audio.value)
    !ffmpeg -i recording.webm -ac 1 -f wav file.wav -y   -hide_banner -loglevel panic
    from IPython.display import Audio

    # Chemin vers le fichier audio enregistré 
    audio_paths="/content/file.wav"

    # Lecture de l'audio
    Audio(filename=audio_paths)
     
     ```
   - **Chargement du modele ASR** 
    ```python
    from huggingsound import SpeechRecognitionModel

    model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-french")

    transcriptions = model.transcribe([audio_paths])

    # Récupérer la transcription
    transcription = transcriptions [0]['transcription']
    
    
     ```
    ```python
     transcription: tu es le chat falle tuer tuer  tues tuez
     
     ```




## Partie 2 : Analyse de Sentiment

Dans cette partie, nous allons construire un modèle de classification de sentiment à partir de Hugging Face. Plus précisément, nous allons configurer notre dataset de manière similaire au dataset [IMDb](https://huggingface.co/datasets/imdb). Ensuite, nous utiliserons le code de [classification de texte de Hugging Face](https://huggingface.co/docs/transformers/v4.17.0/en/tasks/sequence_classification) pour entraîner notre modèle.


### 1. Préparation des Données pour l'Analyse de Sentiment

1. **Obtenir et préparer le dataset**
   - Nous Téléchargons le dataset depuis [Kaggle](https://www.kaggle.com/datasets/djilax/allocine-french-movie-reviews).
   - Nous utiliserons kaggle pour le training de notre modele

2. **Installation des bibliothèques nécessaires**
   - Installez `transformers` pour utiliser les modèles NLP :
     ```bash
     !pip install transformers
     !pip install datasets
     !pip install transformers datasets evaluate accelerate
    ```
3. **Modification des colonnes de notre dataset pour l'adapter a celle de IMDb**
   
     ```python
     from datasets import load_dataset
     # chargement des donnees 
    data=load_dataset("/kaggle/input/allocine-french-movie-reviews")
     ```
    ```
    # afficher du data
    data : 

            DatasetDict({
            train: Dataset({
                features: ['Unnamed: 0', 'film-url', 'review', 'polarity'],
                num_rows: 160000
            })
            validation: Dataset({
                features: ['Unnamed: 0', 'film-url', 'review', 'polarity'],
                num_rows: 20000
            })
            test: Dataset({
                features: ['Unnamed: 0', 'film-url', 'review', 'polarity'],
                num_rows: 20000
            })
    ```
     ```python 
    # Renommer les colonnes et supprimer les colonnes inutiles
    data = data.map(lambda x: {"text": x["review"],    "label": x["polarity"]})
    data = data.remove_columns(['Unnamed: 0', 'film-url', 'review', 'polarity'])
     ```
     ```
     data:

     DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 160000
    })
    validation: Dataset({
        features: ['text', 'label'],
        num_rows: 20000
    })
    test: Dataset({
        features: ['text', 'label'],
        num_rows: 20000
    })
    ```


3. **Code pour le training du modèle**
   - Utilisez le code suivant pour classifier les transcriptions :
     ```python
    # Importer les bibliothèques nécessaires
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
        import numpy as np
        import evaluate
        import torch

        # Chargement du tokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Prétraiter des données
        def preprocess_function(examples):
            return tokenizer(examples["text"], truncation=True)

        # pretraitement des données 
        tokenized_imdb = data.map(preprocess_function, batched=True)

        #  Configurarons du DataCollator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Définitions des labels
        id2label = {0: "negative", 1: "positive"}
        label2id = {"negative": 0, "positive": 1}

        #  Chargement la métrique d'exactitude
        accuracy = evaluate.load("accuracy")

        # Fonction de calcul des métriques
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return accuracy.compute(predictions=predictions, references=labels)

        # Chargement du modèle
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
        )

        # Configuration des arguments d'entraînement
        training_args = TrainingArguments(
            output_dir="my_awesome_model",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=2,
            weight_decay=0.01,
            eval_strategy="steps",
            save_strategy="steps",
            save_steps=1000,  # Sauvegarde toutes les 1000 étapes
            eval_steps=1000,  # Évaluation toutes les 1000 étapes
            load_best_model_at_end=True,
            save_total_limit=3,  # Conserver uniquement les 3 derniers checkpoints
        )

        # Initialisation du  Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_imdb["train"],
            eval_dataset=tokenized_imdb["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # Entraînement du  modèle
        trainer.train()

    ```

     [-----------------------------------------------------------------------] 100% [20000/20000 5:08:26, Epoch 2/2]

    | Step   | Training Loss | Validation Loss | Accuracy |
    |--------|---------------|-----------------|----------|
    | 4000   | 0.212600      | 0.205355        | 0.933500 |
    | 5000   | 0.195400      | 0.181482        | 0.939600 |
    | 6000   | 0.184100      | 0.211350        | 0.936650 |
    | 7000   | 0.177900      | 0.168735        | 0.945500 |
    | 8000   | 0.174500      | 0.213482        | 0.944050 |
    | 9000   | 0.166800      | 0.161974        | 0.947200 |
    | 10000  | 0.177200      | 0.161752        | 0.948500 |
    | 11000  | 0.134500      | 0.174505        | 0.951350 |
    | 12000  | 0.140700      | 0.176011        | 0.948000 |
    | 13000  | 0.125400      | 0.176126        | 0.952400 |
    | 14000  | 0.138100      | 0.170712        | 0.951550 |
    | 15000  | 0.128000      | 0.181787        | 0.951950 |
    | 16000  | 0.127800      | 0.171496        | 0.952700 |
    | 17000  | 0.122600      | 0.168253        | 0.952650 |
    | 18000  | 0.122300      | 0.174152        | 0.952600 |
    | 19000  | 0.114400      | 0.167135        | 0.954600 |
    | 20000  | 0.126300      | 0.167289        | 0.955150 |

    TrainOutput(global_step=20000, training_loss=0.1291647247314453, metrics={'train_runtime': 18506.9733, 'train_samples_per_second': 17.291, 'train_steps_per_second': 1.081, 'total_flos': 7.550016133545792e+16, 'train_loss': 0.1291647247314453, 'epoch': 2.0})
```python
# Préparation des données de test
test_dataset = tokenized_imdb["test"]

# Évaluation du modèle sur les données de test
test_results = trainer.evaluate(eval_dataset=test_dataset)
```
**output**:
```
[---------------------------------------------------] 100% [1250/1250 04:57]

{'eval_loss': 0.15533015131950378, 'eval_accuracy': 0.94955, 'eval_runtime': 297.7234, 'eval_samples_per_second': 67.176, 'eval_steps_per_second': 4.199, 'epoch': 2.0}

```
```python
trainer.save_model("my_awesome_model/final_model")
```
    # Charge le modèle et le tokenizer depuis le répertoire sauvegardé
    model = AutoModelForSequenceClassification.from_pretrained("my_awesome_model/final_model")
    tokenizer = AutoTokenizer.from_pretrained("my_awesome_model/final_model")

    # 16. Tokenizer le texte pour la prédiction
    transcription = "Votre texte ici"  # Remplacez par le texte à prédire
    inputs = tokenizer(transcription, return_tensors="pt")

        # 17. Faire une prédiction
        with torch.no_grad():  # Désactiver le calcul des gradients pour l'inférence
            outputs = model(**inputs)

        # 18. Obtenir les logits et convertir en prédictions
        logits = outputs.logits
        predictions = logits.argmax(dim=-1)

        # 19. Afficher la prédiction
        if predictions.item() == 0:
            print("Sentiment: Negative")
        else:
            print("Sentiment: Positive")



### 2. Synthèse et Résultats

1. **Transcrire l'audio**
   - Utilisez le modèle ASR `wav2vec2-large-xlsr-53-french` pour transcrire l'enregistrement audio fourni.

2. **Analyser le sentiment**
   - Passez la transcription obtenue à travers le modèle d'analyse de sentiment pour obtenir la classification positive ou négative.

3. **Rapport**
   - Présentez les résultats obtenus dans un rapport en expliquant les modèles utilisés, les résultats de la transcription, et les conclusions de l'analyse de sentiment.

---

Assurez-vous de tester soigneusement chaque étape et de valider vos résultats pour garantir la précision des transcriptions et de l'analyse de sentiment. Bonne chance avec votre projet !
```
