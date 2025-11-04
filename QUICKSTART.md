# دليل البدء السريع

## التثبيت السريع

```bash
# 1. إنشاء البيئة الافتراضية
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows

# 2. تثبيت المتطلبات
pip install -r requirements.txt
```

## الاستخدام السريع

### 1. تشغيل المثال

```bash
python examples/example_usage.py
```

### 2. تشغيل واجهة الويب

```bash
python main.py web
# أو
python examples/run_web_app.py
```

ثم افتح المتصفح على: http://localhost:5000

### 3. استخدام البرنامج في الكود

```python
from src.genetic_algorithm import GeneticAlgorithm
from sklearn.datasets import make_classification

# بيانات تجريبية
X, y = make_classification(n_samples=500, n_features=50, random_state=42)

# تشغيل الخوارزمية
ga = GeneticAlgorithm(X, y, population_size=30, n_generations=20)
result = ga.run()

print(f"الميزات المختارة: {len(result['selected_features'])}")
```

### 4. مقارنة جميع الطرق

```python
from src.compare_methods import MethodComparer

comparer = MethodComparer(X, y)
results = comparer.compare_all()
summary = comparer.get_summary()
print(summary)
```

## الاختبارات

```bash
# جميع الاختبارات
pytest

# اختبار محدد
pytest tests/test_genetic_algorithm.py -v
```

## البنية الأساسية

```
src/
├── genetic_algorithm/      # الخوارزمية الجينية
├── traditional_methods/   # الطرق التقليدية
├── data_preprocessing/    # معالجة البيانات
├── web_app/              # تطبيق الويب
└── compare_methods.py     # المقارنة

tests/                     # الاختبارات
examples/                  # الأمثلة
```

## نصائح سريعة

1. **للبيانات الكبيرة**: قلل `population_size` و `n_generations` لتسريع العملية
2. **للدقة العالية**: زد `population_size` و `n_generations`
3. **للتوازن**: استخدم القيم الافتراضية أولاً

## حل المشاكل

**مشكلة**: خطأ في تثبيت الحزم
**الحل**: استخدم `pip install --upgrade pip` ثم أعد التثبيت

**مشكلة**: خطأ في تحميل ملف Excel
**الحل**: تأكد من تثبيت `openpyxl` و `xlrd`

**مشكلة**: الويب لا يعمل
**الحل**: تأكد من أن المنفذ 5000 غير مستخدم

