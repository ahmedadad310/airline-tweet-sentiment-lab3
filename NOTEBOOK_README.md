# تشغيل التحليل الحقيقي

الواجهة التعليمية موجودة في الموقع، والتنفيذ الفعلي موجود في `airline_sentiment_analysis.ipynb`.

## تشغيل محليًا

1. ضع ملف `Tweets.csv` في نفس مجلد الـ notebook. مصدر Kaggle:
   https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment
2. ثبّت المتطلبات:

```bash
pip install -r requirements.txt
```

3. افتح Jupyter:

```bash
jupyter notebook
```

4. افتح `airline_sentiment_analysis.ipynb` وشغّل الخلايا بالترتيب.

الـ notebook ينفذ فعليًا: قراءة الداتا، فحص التوزيع، تنظيف الروابط والـ mentions، حذف stopwords مع الحفاظ على `not` و`no`، بناء TF-IDF، تقسيم train/test، تدريب Logistic Regression، accuracy، classification report، وconfusion matrix.
