# Twitter US Airline Sentiment Analysis

هذا المستودع يحتوي على تنفيذ فعلي لتحليل مشاعر تغريدات شركات الطيران، وليس ملفات إعداد فقط. الـ pipeline يقرأ `Tweets.csv`، ينظف النص، يزيل الروابط والـ mentions، يحذف stopwords مع الحفاظ على `not` و`no`، يحوّل النص إلى TF-IDF، يقسم البيانات إلى train/test، ويدرّب `LogisticRegression` ثم يطبع accuracy وclassification report ويحفظ الرسومات.

## الملفات المهمة

| الملف | الوظيفة |
|---|---|
| `analysis.py` | سكربت Python كامل قابل للتشغيل من الطرفية |
| `airline_sentiment_analysis.ipynb` | نفس الـ pipeline مقسّم إلى خلايا Notebook مع نتائج ورسومات محفوظة تحت الخلايا |
| `Tweets.csv` | نسخة الداتا المستخدمة في التشغيل، وتحتوي على 14,640 صفًا |
| `analysis_outputs/` | `sentiment_distribution.png` و`confusion_matrix.png` الناتجان من التشغيل |
| `requirements.txt` | مكتبات Python المطلوبة |

## مصدر الداتا

المصدر الأساسي هو Kaggle:

https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment

إذا لم يكن `Tweets.csv` موجودًا، سجّل الدخول إلى Kaggle، نزّل الملف، وضعه في نفس مجلد `analysis.py`. النسخة الموجودة في هذا المشروع مطابقة لبنية Kaggle وتحتوي على الأعمدة `text` و`airline_sentiment` المطلوبة للتدريب.

## التشغيل

```bash
pip install -r requirements.txt
python analysis.py
```

النتيجة التي تم التحقق منها على الملف الموجود في المشروع:

```text
Loaded 14,640 rows from Tweets.csv
TF-IDF matrix shape: (14640, 5000)
Accuracy: 0.7889
Sample prediction: positive
```

لتشغيل النسخة التفاعلية:

```bash
jupyter notebook
```

ثم افتح `airline_sentiment_analysis.ipynb` وشغّل الخلايا بالترتيب. النسخة الحالية محفوظة بعد التنفيذ، لذلك تحتوي على المخرجات والرسومات أسفل الخلايا.
