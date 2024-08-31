# Guide pour la Reconnaissance Automatique de la Parole (ASR) et l'Analyse de Sentiment

## Partie 1 : Construire un modèle de Reconnaissance Automatique de la Parole (ASR)

### 1. Choix et Configuration du Modèle ASR

1. **Choisir un modèle ASR sur Hugging Face**
   - Rendez-vous sur le [Hub de Hugging Face](https://huggingface.co/models).
   - Recherchez un modèle de reconnaissance automatique de la parole pour le français. Par exemple, utilisez `facebook/wav2vec2-large-xlsr-53-french`.

2. **Installation des bibliothèques nécessaires**
   - Installez les bibliothèques nécessaires avec les commandes suivantes :
     ```bash
     pip install transformers datasets librosa
     ```

3. **Code pour la Reconnaissance Automatique de la Parole**
   - Utilisez le code suivant pour transcrire des enregistrements audio en texte :
     ```python
     from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
     import torch
     import librosa

     # Charger le modèle et le tokenizer
     model_name = "facebook/wav2vec2-large-xlsr-53-french"
     tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
     model = Wav2Vec2ForCTC.from_pretrained(model_name)

     # Fonction de transcription
     def transcribe_audio(file_path):
         # Charger l'audio
         audio_input, _ = librosa.load(file_path, sr=16000)
         inputs = tokenizer(audio_input, return_tensors="pt", padding="longest")
         
         # Transcrire
         with torch.no_grad():
             logits = model(input_values=inputs.input_values).logits
         
         # Convertir les logits en texte
         predicted_ids = torch.argmax(logits, dim=-1)
         transcription = tokenizer.batch_decode(predicted_ids)
         return transcription[0]

     # Exemple d'utilisation
     file_path = "chemin/vers/votre/audio.wav"
     print(transcribe_audio(file_path))
      ```

## Partie 2 : Analyse de Sentiment

### 1. Préparation des Données pour l'Analyse de Sentiment

1. **Obtenir et préparer le dataset**
   - Téléchargez le dataset depuis [Kaggle](https://www.kaggle.com/datasets/djilax/allocine-french-movie-reviews).
   - Décompressez le fichier et chargez les données dans votre environnement de travail.

2. **Installation des bibliothèques nécessaires**
   - Installez `transformers` pour utiliser les modèles NLP :
     ```bash
     pip install transformers
     ```

3. **Code pour l'Analyse de Sentiment**
   - Utilisez le code suivant pour classifier les transcriptions :
     ```python

    from transformers import BertTokenizer, BertForSequenceClassification
    from torch.nn.functional import softmax
    import torch

     # Charger le modèle et le tokenizer
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)

     # Fonction d'analyse de sentiment
    def analyze_sentiment(text):
         inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
         with torch.no_grad():
             logits = model(**inputs).logits
         probs = softmax(logits, dim=-1)
         sentiment = torch.argmax(probs, dim=-1).item()
         return sentiment

     # Exemple d'utilisation
     transcription = "Votre transcription ici"
     sentiment = analyze_sentiment(transcription)
     print("Sentiment (0: négatif, 1: positif):", sentiment)
      ```

### 2. Synthèse et Résultats

1. **Transcrire l'audio**
   - Utilisez le modèle ASR pour transcrire l'enregistrement audio fourni.

2. **Analyser le sentiment**
   - Passez la transcription obtenue à travers le modèle d'analyse de sentiment pour obtenir la classification positive ou négative.

3. **Rapport**
   - Présentez les résultats obtenus dans un rapport en expliquant les modèles utilisés, les résultats de la transcription, et les conclusions de l'analyse de sentiment.

---

Assurez-vous de tester soigneusement chaque étape et de valider vos résultats pour garantir la précision des transcriptions et de l'analyse de sentiment. Bonne chance avec votre projet !
