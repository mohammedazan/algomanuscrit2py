# 📘 Présentation du Projet – PFE Deep Learning
## Reconnaissance d’algorithmes manuscrits et traduction en Python

---

## 👥 Équipe du projet
Projet réalisé par une équipe de 3 étudiants en Master.  
Tous les membres ont le même niveau académique et participent de manière équitable au projet.

---

## 🎯 1. تعريف المشروع (Objectif du projet)

هاد المشروع هو **Projet de Fin de Module – Deep Learning**.  
الهدف الرئيسي ديالو هو:

> تطوير تطبيق ذكي كياخذ صورة فيها خوارزمية مكتوبة بخط اليد  
> وكيحوّلها تلقائياً إلى **كود Python قابل للتنفيذ**.

بشكل مبسّط:
```

📷 Image (Algorithme manuscrit)
↓
🧠 Intelligence Artificielle
↓
🐍 Code Python

```

المشروع كيدمج بين:
- Deep Learning
- Computer Vision
- Algorithmique
- Génie Logiciel

---

## 🧠 2. الفكرة العامة (Idée globale)

طريقة العمل العامة ديال التطبيق كتتبع هاد السلسلة:

```

Image → Preprocessing → OCR → Texte → Parsing → Python Code

```

الشرح:
1. المستخدم كيدخل صورة لخوارزمية مكتوبة باليد
2. الصورة كتتصلّح (تحسين الإضاءة، التباين…)
3. موديل Deep Learning كيتعرف على النص
4. النص كيتحلّل منطقياً
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

📌 احترام هاد البنية ضروري باش المشروع يبقى منظم وقابل للتوسيع.

---

## 🧪 4. Dataset (المعطيات)

- أكثر من **100 صورة** لخوارزميات مكتوبة بخط اليد
- كل صورة مرتبطة بـ:
  - النص ديال الخوارزمية (Pseudo-code)
  - كود Python الموافق لها (للتقييم)

الصيغ المستعملة:
- CSV (للتجارب والتحقق)
- JSON (أكثر أمان للنصوص متعددة الأسطر)

أنواع الخوارزميات:
- Lire / Afficher
- Boucles (For)
- Calculات (Somme, Moyenne, Max…)
- قابلة للتوسيع لاحقاً

---

## 🖼️ 5. Preprocessing (معالجة الصور)

قبل إدخال الصورة للموديل كنقومو بـ:
- تحويلها إلى Grayscale
- تقليل الضجيج بـ Gaussian Blur
- Adaptive Thresholding
- Resize إلى حجم ثابت (128 × 512)

هاد الخطوة كتساعد على:
- رفع دقة التعرف
- التعامل مع إضاءة ضعيفة أو خطوط مختلفة

---

## 🤖 6. Deep Learning – OCR

التعرف على النص المكتوب باليد كيعتمد على:

- **CRNN (CNN + BiLSTM + CTC)**

الدور ديال الموديل:
- ياخذ الصورة
- ويرجع النص كسلسلة حروف مرتّبة

التقنيات المستعملة:
- TensorFlow / Keras
- CTC Loss
- Sequence Modeling

الهدف:
- دقة عالية
- سلوك مستقر
- قابل للشرح أكاديمياً

---

## 🧩 7. Parsing & Génération du code

النص الناتج من OCR كيتحوّل إلى كود Python عبر قواعد محددة.

مثال:
| Algorithme | Python |
|-----------|--------|
| Lire(a) | a = int(input()) |
| Afficher(a) | print(a) |
| Pour i de 1 à n | for i in range(1, n+1): |

هاد المرحلة Rule-based وما فيهاش Deep Learning.

---

## 🌐 8. Application Web

التطبيق النهائي مبني بـ **Streamlit**.

الوظائف:
- Upload صورة
- عرض الصورة قبل وبعد preprocessing
- عرض النص المستخرج
- عرض كود Python النهائي

واجهة بسيطة وواضحة.

---

## 🧩 9. خطة تقسيم المهام (Proposition)

باش الخدمة تمشي بسلاسة، يمكن تقسيم العمل إلى محاور تقنية:

### 🔹 المحور 1: Data & Preprocessing
- تنظيم dataset
- Validation
- Image preprocessing

### 🔹 المحور 2: Deep Learning (OCR)
- بناء الموديل
- التدريب والتحسين
- اختبار الدقة

### 🔹 المحور 3: Parsing & Application
- تحويل النص إلى Python
- بناء الواجهة
- دمج المكونات

📌 المحاور مستقلة نسبياً ولكن خاص تنسيق مستمر بينها.

---

# 🧭 Répartition du travail (3 Axes) — Guide détaillé pour chaque محور  
> 📌 الهدف من هاد القسم هو كل واحد اللي غادي يشد محور يفهم:  
شنو يدير بالضبط ✅، فين يخدم ✅، وكيفاش يخرج نتيجة قوية ✅.  
(الشرح بالدارجة المغربية + المصطلحات بالفرنسية/الانجليزية)

---

## 🔹 المحور 1: Data & Preprocessing  
### (Organisation du dataset + Validation + Prétraitement d’images)

### 🎯 الهدف ديال المحور  
نخليو الـ dataset **منظم، نظيف، ومفهوم** + نوجدّو preprocessing قوي باش يزيد دقة OCR.

---

### ✅ المهام الرئيسية (بالترتيب)

#### 1) تنظيم الـ Dataset (Organisation)
- جمع جميع الصور فـ **folder واحد موحّد**:  
  `data/images/`
- جمع الـ annotations فـ:  
  `data/annotations/`

📌 الهدف: نخليو كلشي consistent وما كايناش paths عشوائية.

✅ الشكل النهائي المقترح:
```

data/
├── images/
└── annotations/
├── dataset.csv
└── dataset.json

```

#### 2) توحيد الـ paths (Normalisation des chemins)
- أي `image_path` داخل CSV/JSON خاصو يكون relative وموحّد:
  - مثال: `images/alg_001.jpg`

📌 نصيحة: JSON أسهل وأكثر robustness من CSV حيث فيه multiline text والكود.

#### 3) Validation & Quality Checks
- تطوير/تحديث loader باش:
  - يتحقق من وجود الصور
  - يتحقق من أن `text` ماشي فارغ
  - يخرج statistics: عدد العينات، distribution ديال categories
  - يخرج list ديال entries اللي فيها مشاكل

✅ Output مهم للتقرير:
- Total samples
- Invalid samples
- Missing images
- Empty labels

#### 4) Preprocessing ديال الصور (OpenCV)
الهدف: تحسين image باش تكون مناسبة للـ OCR:

المراحل الأساسية:
- Grayscale
- Gaussian Blur (noise reduction)
- Adaptive Thresholding
- Resize إلى (128×512)

✅ إضافة تحسينات اختيارية (لكن قوية):
- Morphological operations (Opening/Closing) لتنقية noise
- Deskew (تصحيح الميلان) إذا كان كاين
- Crop/ROI (تقليص المساحة لغير النص)

📌 نصيحة Master:  
دير preprocessing configurable (بارامترات قابلة للتعديل).

---

### 📌 مخرجات المحور 1 (Deliverables)
- ✅ Dataset structure موحد داخل `data/`
- ✅ Loader/validator قوي
- ✅ Preprocessing module يعطي صور واضحة
- ✅ تقرير صغير (حتى داخل README) فيه stats قبل/بعد

---

### ⭐ نصائح مهمة (Conseils)
- ما تزيدش preprocessing معقد بزاف (Keep it simple)
- دير visualization دائماً: original vs processed
- ركّز على robustness: صور بإضاءة ضعيفة وخط صعيب
- خدم دائماً بعينات مختلفة ماشي نفس الصورة

---

## 🔹 المحور 2: Deep Learning (OCR)  
### (Construction du modèle + Entraînement + Évaluation)

### 🎯 الهدف ديال المحور  
نبنيو OCR model (CRNN) يقدر يحوّل الصورة المعالجة إلى **نص pseudo-code** بدقة عالية وبـ robustness.

---

### ✅ المهام الرئيسية (بالترتيب)

#### 1) إعداد vocabulary (Alphabet / Charset)
- خاصنا لائحة الحروف/الرموز اللي كاينة فـ dataset:
  - lettres (a-z, A-Z)
  - chiffres (0-9)
  - symbols: `()`, `:`, `<-`, `+`, `-`, `*`, `/`, `"`, `\n`, space …

📌 هادي مهمة جداً حيت output layer ديال model مبني عليها.

✅ نصيحة:
- بدا بـ charset بسيط ثم زيد تدريجياً.

#### 2) بناء الموديل (Architecture CRNN)
- CNN لاستخراج features
- BiLSTM لفهم sequence
- CTC output layer

📌 الملفات:
- `src/ocr/model.py` : تعريف الموديل
- `src/ocr/train.py` : training
- `src/ocr/predict.py` : inference

#### 3) Training pipeline (Entraînement)
- Split dataset:
  - Train / Validation (مثلاً 80/20)
- Use augmentation:
  - rotation خفيفة
  - blur خفيف
  - contrast/brightness variation
  - noise خفيف

📌 الهدف: model يتعلم robust ضد الصور الرديئة.

✅ نصيحة:
- Start small: train على subset باش تتأكد كلشي خدام، ثم train على dataset كامل.

#### 4) Evaluation (Mesure de performance)
مقاييس مهمة:
- Character Error Rate (CER)
- Word Error Rate (WER) (اختياري)

📌 Output مهم للتقرير:
- accuracy curves
- sample predictions قبل/بعد training
- confusion points (فين كيخطأ أكثر)

---

### 📌 مخرجات المحور 2 (Deliverables)
- ✅ `model.py` معماري واضح ومشروح
- ✅ training script خدام
- ✅ weights محفوظين
- ✅ نتائج evaluation (CER/WER) + أمثلة predictions

---

### ⭐ نصائح مهمة (Conseils)
- ما تحاولش تجيب 100% accuracy: ركّز على “robust & usable”
- حافظ على reproducibility (seed, config)
- سجّل التجارب (hyperparameters) فـ notebook أو ملف log
- إذا وقع overfitting: زيد augmentation أو نقص model complexity

---

## 🔹 المحور 3: Parsing & Application  
### (Algorithm → Python + UI Streamlit + Integration)

### 🎯 الهدف ديال المحور  
نحوّلو النص اللي خرج من OCR إلى Python code صحيح، ونبني واجهة Streamlit تجمع كلشي وتعرض النتائج.

---

### ✅ المهام الرئيسية (بالترتيب)

#### 1) Parsing: تحويل pseudo-code إلى Python
📁 الملف الأساسي:
- `src/parser/algo_to_python.py`

الفكرة:
- Rules + Mapping + Regex

✅ Mapping أساسي:
- `Lire(x)` → `x = int(input())`
- `Afficher(x)` → `print(x)`
- `x <- expr` → `x = expr`
- `Pour i de 1 à n` → `for i in range(1, n+1):`
- `Fin Pour` → نهاية bloc (indentation)

📌 تحدي كبير: indentation
- خاص parser يبني blocks ويحسب indentation level.

✅ نصيحة:
- بدا بــ support ديال categories اللي عندنا دابا:
  - Lecture & Écriture
  - Boucles For بسيطة
- ثم زيد conditions لاحقاً.

#### 2) Application UI (Streamlit)
📁 الملف:
- `src/app/app.py`

الواجهة خاصها:
- Upload image
- عرض original image
- عرض preprocessed image
- زر “Run OCR”
- عرض النص المستخرج
- زر “Generate Python”
- عرض الكود النهائي (code block)
- Optional: زر “Copy” (Streamlit component) أو download .py

📌 مهم: واجهة بسيطة ولكن منظمة.

#### 3) Integration: دمج pipeline كامل
داخل `app.py`:
- call preprocessing
- call OCR predict
- call parser
- show results

✅ نصيحة:
- دير “error handling” واضح:
  - إذا OCR خرج فارغ
  - إذا parsing فشل
  - إذا الصورة ما تقراتش

---

### 📌 مخرجات المحور 3 (Deliverables)
- ✅ Parser rules خدامة لعدة أمثلة
- ✅ Streamlit UI خدامة
- ✅ Integration end-to-end (Image → Python code)
- ✅ Demo سيناريوهات (3-5 صور) جاهزين للعرض

---

### ⭐ نصائح مهمة (Conseils)
- ما تبقاش تبني rules شاملة بزاف مرة وحدة: زيد تدريجياً
- دير unit tests صغار (حتى غير scripts) باش تأكد mapping
- ركّز على user experience:
  - outputs واضحة
  - خطوات مفهومة
- حضّر 3–5 صور “demo” بجودة مختلفة (مزيانة/ضعيفة) باش تورّي robustness

---

✅ **ملاحظة ختامية**
هاد 3 محاور كيتلاقاو فـ integration، لذلك أي محور يكمّل خاصو:
- يكتب code clean
- يحافظ على structure
- ويخلّي functions قابلة للاستدعاء من app بسهولة

---
```
