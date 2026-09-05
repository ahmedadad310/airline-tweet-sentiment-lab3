# Airline Tweet Sentiment Analysis

لوحة تفاعلية عربية ومشروع NLP عملي لتحليل مشاعر تغريدات شركات الطيران الأمريكية باستخدام **Twitter US Airline Sentiment**.

## الهدف

تحويل تغريدة قصيرة وغير مرتبة إلى تصنيف قابل للقياس: `negative` أو `neutral` أو `positive`، مع توضيح ما الذي يتعلمه النموذج وأين يخطئ.

## الداتاست

استخدمنا داتاست [Twitter US Airline Sentiment على Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)، وهي تحتوي على 14,640 تغريدة من فبراير 2015 عن ست شركات طيران أمريكية. الترخيص CC BY-NC-SA 4.0. يتم تنزيل `Tweets.csv` يدويًا من Kaggle أو من المصدر العام المذكور في الـ notebook؛ لا يتم تضمين البيانات الخام في هذا المستودع.

## المنهجية

1. تنظيف النص: lowercase، إزالة الروابط والـ mentions والهاشتاجات والرموز، ثم حذف stopwords مع الإبقاء على كلمات النفي مثل `not` و`no`.
2. تحويل النص إلى أرقام باستخدام `TfidfVectorizer` مع unigrams وbigrams.
3. تقسيم stratified بنسبة 80/20 إلى train وtest.
4. تدريب baseline باستخدام `LogisticRegression(max_iter=1500, C=2.0)`.
5. تقييم accuracy وclassification report وconfusion matrix.
6. استخراج أعلى الكلمات المؤثرة من coefficients النموذج.

## النتيجة المرجعية

تم تشغيل التجربة على `random_state=42`:

| المقياس | النتيجة |
| --- | ---: |
| الدقة Accuracy | **78.79%** |
| حجم التدريب | 11,712 |
| حجم الاختبار | 2,928 |
| حجم المفردات | 13,728 |
| F1 — negative | 0.87 |
| F1 — neutral | 0.58 |
| F1 — positive | 0.68 |

الفئة المحايدة هي الأصعب: `recall = 0.52`، ويميل النموذج إلى نقل جزء من التغريدات المحايدة إلى السلبية، وهو ما يظهر بوضوح في مصفوفة الالتباس داخل الموقع.

## التشغيل محليًا

```bash
pip install pandas scikit-learn nltk matplotlib seaborn
python -c "import nltk; nltk.download('stopwords')"
python analysis/train_model.py --data /path/to/Tweets.csv
```

الـ notebook القابل للتشغيل موجود في [`notebooks/airline_sentiment_analysis.ipynb`](notebooks/airline_sentiment_analysis.ipynb)، والواجهة في `client/` مبنية باستخدام React + Vite + Tailwind.

## تشغيل الموقع

```bash
pnpm install
pnpm dev
```

## روابط المصدر

- [Kaggle dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- [مصدر CSV العام المستخدم للتحقق](https://github.com/ruchitgandhi/Twitter-Airline-Sentiment-Analysis)
