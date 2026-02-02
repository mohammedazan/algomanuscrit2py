# 📘 Présentation du Projet – PFE Deep Learning
## Reconnaissance d’algorithmes manuscrits et traduction en Python

---

## 👥 Équipe du projet
- **Mohammed** – Coordination générale, Data, Parsing, Intégration
- **Imad** – Modèle Deep Learning (OCR)
- **Houda** – Interface utilisateur (Web App)

---

## 🎯 1. تعريف المشروع (Objectif du projet)

هاد المشروع هو **Projet de Fin de Module – Deep Learning**  
الهدف ديالو هو:

> إنشاء تطبيق كياخذ صورة فيها خوارزمية مكتوبة بخط اليد  
> وكيحوّلها تلقائياً إلى **كود Python قابل للتنفيذ**.

بعبارة بسيطة:
```

📷 Image (Algorithme manuscrit)
↓
🧠 Intelligence Artificielle
↓
🐍 Code Python

```

المشروع كيجمع بين:
- Deep Learning
- Computer Vision
- Algorithmique
- Software Engineering

---

## 🧠 2. الفكرة العامة (Idée globale)

التطبيق كيخدم بهاد السلسلة (Pipeline):

```

Image → Preprocessing → OCR → Texte → Parsing → Python Code

```

بالتفصيل:
1. المستخدم كيدخل صورة
2. الصورة كتتصلّح (إضاءة، ألوان…)
3. موديل Deep Learning كيقرا النص المكتوب باليد
4. النص كيتحلّل (Lire, Afficher, Boucle…)
5. كنخرجو كود Python صحيح

---

## 🏗️ 3. بنية المشروع (Architecture)

```

handwritten_algo_to_python/
│
├── data/
│   ├── images/                # صور الخوارزميات المكتوبة باليد
│   └── annotations/
│       ├── dataset.csv
│       └── dataset.json
│
├── src/
│   ├── preprocessing/
│   │   └── image_preprocess.py
│   │
│   ├── ocr/
│   │   ├── model.py
│   │   ├── train.py
│   │   └── predict.py
│   │
│   ├── parser/
│   │   └── algo_to_python.py
│   │
│   └── app/
│       └── app.py
│
├── notebooks/
│   └── exploration.ipynb
│
├── requirements.txt
└── README.md

```

📌 مهم: خاص نحترمو هاد البنية باش ما يتخلطش المشروع.

---

## 🧪 4. Dataset (المعطيات)

- قرابة **100+ صورة**
- كل صورة عندها:
  - النص ديال الخوارزمية (pseudo-code)
  - كود Python الموافق لها
- الصيغة:
  - CSV (للتجارب)
  - JSON (أكثر أمان للنصوص المتعددة الأسطر)

أنواع الخوارزميات:
- Lire / Afficher
- Boucles For
- Calcul (Somme, Moyenne, Max…)
- (قابل للتوسيع)

---

## 🖼️ 5. Preprocessing (معالجة الصور)

قبل ما ندخلو الصورة للموديل، كنقومو بـ:
- تحويلها لـ Grayscale
- Gaussian Blur (نقص noise)
- Adaptive Threshold
- Resize إلى (128 × 512)

هاد الخطوة مهمة بزاف باش:
- نزيدو دقة OCR
- نخليو الموديل robust ضد الإضاءة الضعيفة

---

## 🤖 6. Deep Learning – OCR

غادي نستعملو:
- **CRNN (CNN + BiLSTM + CTC)**

الدور ديالو:
- ياخذ الصورة
- ويرجع النص المكتوب باليد كسلسلة حروف

التقنيات:
- TensorFlow / Keras
- CTC Loss
- Sequence modeling

📌 الهدف ماشي 100% accuracy، ولكن:
- دقة عالية
- سلوك مستقر
- قابل للشرح أكاديمياً

---

## 🧩 7. Parsing & Génération du code

من بعد OCR:
- كنحوّلو النص لقواعد

مثال:
| Algorithme | Python |
|-----------|--------|
| Lire(a) | a = int(input()) |
| Afficher(a) | print(a) |
| Pour i de 1 à n | for i in range(1, n+1): |

هاد الجزء Rule-based (ماشي DL).

---

## 🌐 8. Application Web

- مبنية بـ **Streamlit**
- فيها:
  - Upload image
  - عرض preprocessing
  - عرض النص المستخرج
  - عرض كود Python النهائي

واجهة بسيطة ولكن واضحة.

---

## 🧑‍💻 9. تقسيم المهام المقترح (Task Distribution)

### 🔹 Mohammed (Chef de projet)
- تنظيم المشروع والبنية
- Dataset + validation
- Parsing (Algorithm → Python)
- دمج جميع المكونات
- التحضير للعرض (présentation)

### 🔹 Imad (Deep Learning)
- OCR Model (CRNN)
- Training و tuning
- Tests de reconnaissance
- Explication du modèle

### 🔹 Houda (Interface & UX)
- Web App (Streamlit)
- Upload image
- Affichage des résultats
- تحسين تجربة المستخدم

📌 كل واحد خدام على جزء، ولكن التواصل ضروري.

