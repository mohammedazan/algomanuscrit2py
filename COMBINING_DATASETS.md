# 📚 Guide Complet: Entraîner le Modèle avec les Deux Datasets

## 🎯 Objectif

Améliorer la précision de votre modèle OCR en combinant:
- **Dataset 1**: ~160 images (JSON format) 
- **Dataset 2**: ~82,000+ images (CSV format)
- **Total**: 82,000+ images pour un meilleur entraînement

## ⚡ Pourquoi Deux Datasets?

| Aspect | Impact |
|--------|--------|
| **Plus de données** | Meilleure généralisation, moins d'overfitting |
| **Variance** | Modèle plus robuste face à différentes écritures |
| **Précision** | Extraction de texte plus fiable |
| **Convergence** | Modèle apprend plus rapidement avec plus d'exemples |

## 🚀 Quick Start

### 1. Utiliser le Nouveau Chargeur Unifié

```python
from src.data.unified_dataset_loader import UnifiedDatasetLoader

# Créer le loader
loader = UnifiedDatasetLoader()

# Charger les deux datasets
df = loader.load_all_datasets()

# Obtenir les chemins et labels
image_paths = loader.get_image_paths()
labels = loader.get_labels()

# Diviser en train/validation
train_df, val_df = loader.get_split(train_ratio=0.85)

# Vérifier les fichiers
loader.verify_images(sample_size=100)
```

### 2. Entraîner le Modèle (Nouvelle Méthode - Recommandé)

```bash
cd algomanuscrit2py
python src/ocr/train_combined.py
```

**Cette commande:**
- ✅ Combine les 2 datasets automatiquement
- ✅ Entraîne sur ~82,000 images
- ✅ Utilise meilleure augmentation de données
- ✅ Sauvegarde meilleur modèle
- ✅ Génère logs détaillés

### 3. Résultats et Checkpoints

Les fichiers suivants seront créés dans `checkpoints/`:

```
checkpoints/
├── best_model.weights.h5          ← Meilleur modèle
├── final_model.weights.h5         ← Modèle final
├── training_log.csv               ← Résultats d'entraînement
└── training_info.json             ← Métadonnées
```

## 📊 Améliorations Apportées

### Configuration du Modèle
| Paramètre | Ancienne Valeur | Nouvelle Valeur | Raison |
|-----------|-----------------|-----------------|---------|
| Batch Size | 8 | 16 | Meilleure utilisation GPU |
| Epochs | 100 | 50 | Moins nécessaire avec plus de données |
| Learning Rate | Fixe 1e-3 | Adaptative (1e-3) | Meilleure convergence |
| Augmentation | Simple | Avancée (rotation, contrast) | Plus de robustesse |

### Données d'Entraînement
- **Avant**: ~160 images
- **Après**: ~69,700 images (85% de 82k)
- **Amélioration**: 435x plus de données!

## 🔍 Structure des Fichiers

```
algomanuscrit2py/
│
├── Dataset/                          # Dataset 1 (JSON)
│   ├── dataset.json                 # Métadonnées
│   └── images/                      # ~160 images
│
├── dataset (1)/dataset/             # Dataset 2 (CSV)
│   ├── labels.csv                  # 82k+ labels
│   └── images/                     # 82k+ images
│
├── src/
│   ├── data/
│   │   ├── unified_dataset_loader.py  # ✨ NOUVEAU
│   │   ├── dataset_loader.py          # Ancien (JSON only)
│   │   └── text_normalizer.py
│   │
│   └── ocr/
│       ├── train_combined.py           # ✨ NOUVEAU
│       ├── train.py                    # Ancien (JSON only)
│       ├── model.py
│       └── predict.py
│
└── checkpoints/
    ├── best_model.weights.h5
    └── training_log.csv
```

## 💡 Utilisation Avancée

### A. Charger Uniquement le Grand Dataset

```python
configs = [
    {
        'type': 'csv',
        'data_path': 'dataset (1)/dataset/labels.csv',
        'images_dir': 'dataset (1)/dataset/images',
        'name': 'Large_Dataset'
    }
]

loader = UnifiedDatasetLoader(dataset_configs=configs)
df = loader.load_all_datasets()
```

### B. Ajouter d'Autres Datasets

```python
configs = [
    # Dataset 1
    {'type': 'json', 'data_path': 'Dataset/dataset.json', ...},
    # Dataset 2
    {'type': 'csv', 'data_path': 'dataset (1)/dataset/labels.csv', ...},
    # Dataset 3 (nouveau)
    {'type': 'csv', 'data_path': 'mon_dataset/labels.csv', ...},
]

loader = UnifiedDatasetLoader(dataset_configs=configs)
```

### C. Analyser les Données

```python
import matplotlib.pyplot as plt
from collections import Counter

loader = UnifiedDatasetLoader()
df = loader.load_all_datasets()

# Longueur des textes
text_lengths = df['text'].str.len()
print(f"Longueur moyenne: {text_lengths.mean():.0f} caractères")
print(f"Max: {text_lengths.max()}")
print(f"Min: {text_lengths.min()}")

# Distribution par dataset
print(df['dataset'].value_counts())

# Distribution des datasets
df['dataset'].value_counts().plot(kind='bar', title='Samples per Dataset')
plt.show()
```

## ⚙️ Configuration d'Entraînement

Éditer `src/ocr/train_combined.py` pour personnaliser:

```python
class EnhancedTrainingConfig:
    # Modèle
    self.batch_size = 16              # Taille des batches
    self.epochs = 50                  # Nombre d'epochs
    self.initial_lr = 1e-3            # Learning rate initial
    
    # Augmentation
    self.brightness_delta = 0.2       # Variation de luminosité
    self.contrast_range = (0.8, 1.2)  # Variation de contraste
    self.rotation_degrees = 3.0       # Rotation max
    
    # Callbacks
    self.early_stopping_patience = 10 # Arrêt anticipé
    self.reduce_lr_patience = 3       # Réduction LR patience
```

## 📈 Monitoring l'Entraînement

### Avec TensorBoard

```bash
tensorboard --logdir=logs
# Puis visitez http://localhost:6006
```

### Avec CSV Logger

```python
import pandas as pd

log_df = pd.read_csv('checkpoints/training_log.csv')

# Plot loss
import matplotlib.pyplot as plt
plt.plot(log_df['loss'], label='Train Loss')
plt.plot(log_df['val_loss'], label='Val Loss')
plt.legend()
plt.show()
```

## 🔧 Dépannage

### Problème: "File not found"
```
⚠ Vérifier que les chemins sont corrects:
- Dataset/dataset.json existe
- dataset (1)/dataset/labels.csv existe
- Les images sont présentes dans les dossiers
```

**Solution:**
```python
loader = UnifiedDatasetLoader()
loader.load_all_datasets()
loader.verify_images(sample_size=500)  # Vérifier tous
```

### Problème: Mémoire insuffisante
```python
# Réduire batch size
config.batch_size = 8  # au lieu de 16

# Réduire résolution
config.input_shape = (96, 384, 1)  # au lieu de (128, 512, 1)
```

### Problème: Entraînement très lent
```python
# Augmenter workers
dataset = dataset.map(..., num_parallel_calls=tf.data.AUTOTUNE)

# Augmenter batch size (si GPU permet)
config.batch_size = 32
```

## 📊 Métriques de Performance

Après entraînement, vérifier dans `checkpoints/training_log.csv`:

```
epoch | loss | val_loss | accuracy
    1 | 50.2 |   48.5   | 0.45
    2 | 42.1 |   40.3   | 0.52
    ...
   50 | 2.3  |   2.8    | 0.89    ✓ Bon!
```

**Bonnes métriques:**
- Loss < 5.0
- Val Loss proche du Loss (pas d'overfitting)
- Accuracy > 0.80

## 🎯 Prochaines Étapes

1. **Testez le modèle:**
   ```bash
   python src/ocr/predict.py --image test_image.jpg
   ```

2. **Comparez avec ancien modèle:**
   ```python
   # Charger nouveau modèle
   new_model = load_model('checkpoints/best_model.weights.h5')
   # Charger ancien modèle (si existe)
   old_model = load_model('checkpoints/final_inference_model.weights.h5')
   # Comparer sur mêmes images
   ```

3. **Ajustez les hyperparamètres** selon les résultats

4. **Continuez l'entraînement** si pas converge:
   ```python
   model.load_weights('checkpoints/best_model.weights.h5')
   # Entraîner 20 epochs de plus...
   ```

## 📚 Ressources

- **Documentation TensorFlow**: https://www.tensorflow.org/api_docs
- **CRNN Architecture**: https://arxiv.org/abs/1507.05717
- **CTC Loss**: https://en.wikipedia.org/wiki/Connectionist_temporal_classification

## ❓ Questions?

Vérifications avant de démarrer:
- [ ] Les 2 dossiers Dataset existent?
- [ ] GPU disponible pour accélérer?
- [ ] Espace disque suffisant (~5-10GB)?
- [ ] TensorFlow 2.10+ installé?

---

**Bon entraînement! 🚀**
