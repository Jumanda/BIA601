# مشروع اختيار الميزات باستخدام الخوارزميات الجينية

## نظرة عامة

يهدف هذا المشروع إلى تطوير نظام شامل لاختيار الميزات (Feature Selection) باستخدام الخوارزميات الجينية، مع إمكانية المقارنة مع الطرق الإحصائية التقليدية.

**الوحدة:** BIA601 - الخوارزميات المتقدمة  
**المؤلف:** د. عصام سلمان  
**تاريخ التسليم:** 03.09.2025  
**تاريخ الاستلام:** 10.11.2025

## المميزات

- ✅ تطبيق الخوارزمية الجينية لاختيار الميزات
- ✅ تمثيل الكروموسوم وعمليات التطور (Selection, Crossover, Mutation)
- ✅ مقارنة مع الطرق التقليدية (Correlation, Mutual Information, Univariate Tests, RFE)
- ✅ واجهة ويب متكاملة لرفع البيانات وعرض النتائج
- ✅ معالجة البيانات التلقائية
- ✅ تقارير تفصيلية ومقارنات مرئية

## هيكل المشروع

```
.
├── src/
│   ├── genetic_algorithm/      # تنفيذ الخوارزمية الجينية
│   │   ├── chromosome.py       # تمثيل الكروموسوم
│   │   ├── operators.py         # عمليات التطور
│   │   └── genetic_algorithm.py # الخوارزمية الرئيسية
│   ├── traditional_methods/    # الطرق التقليدية
│   │   ├── correlation.py
│   │   ├── mutual_info.py
│   │   ├── univariate.py
│   │   └── recursive_elimination.py
│   ├── data_preprocessing/      # معالجة البيانات
│   │   └── preprocessor.py
│   ├── web_app/                 # تطبيق الويب
│   │   ├── app.py
│   │   ├── templates/
│   │   └── static/
│   └── compare_methods.py      # مقارنة الطرق
├── tests/                       # الاختبارات
├── examples/                    # أمثلة الاستخدام
├── data/                        # البيانات (اختياري)
├── uploads/                     # ملفات مرفوعة
├── results/                     # النتائج المحفوظة
└── requirements.txt

```

## التثبيت

### المتطلبات

- Python 3.8 أو أحدث
- pip

### خطوات التثبيت

1. استنساخ المشروع أو تنزيله

2. إنشاء بيئة افتراضية (موصى به):

```bash
python3 -m venv venv
source venv/bin/activate  # على macOS/Linux
# أو
venv\Scripts\activate  # على Windows
```

3. تثبيت المتطلبات:

```bash
pip install -r requirements.txt
```

## الاستخدام

### 1. استخدام البرنامج من سطر الأوامر

#### مثال بسيط:

```python
from src.genetic_algorithm import GeneticAlgorithm
from sklearn.datasets import make_classification

# إنشاء بيانات تجريبية
X, y = make_classification(n_samples=500, n_features=50, n_informative=15, random_state=42)

# تشغيل الخوارزمية الجينية
ga = GeneticAlgorithm(X, y, population_size=30, n_generations=20)
result = ga.run(verbose=True)

print(f"الميزات المختارة: {result['selected_features']}")
print(f"النتيجة: {result['best_fitness']:.4f}")
```

#### مقارنة جميع الطرق:

```python
from src.compare_methods import MethodComparer

comparer = MethodComparer(X, y)
results = comparer.compare_all()
summary = comparer.get_summary()
print(summary)
```

#### تشغيل المثال الكامل:

```bash
python examples/example_usage.py
```

### 2. استخدام واجهة الويب

1. تشغيل الخادم:

```bash
python examples/run_web_app.py
```

2. فتح المتصفح والانتقال إلى:

```
http://localhost:5000
```

3. رفع ملف البيانات (CSV أو Excel)
4. اختيار عمود الهدف
5. ضبط معاملات الخوارزمية الجينية (اختياري)
6. تشغيل التحليل
7. عرض النتائج والمقارنات

## العرض التجريبي (Demo)

<video src="Screen%20Recording%202025-11-01%20at%2012.04.36%E2%80%AFPM.mov" controls width="800" muted>
  متصفحك لا يدعم تشغيل الفيديو. يمكنك تنزيله من الرابط أدناه.
</video>

رابط مباشر للفيديو: [مشاهدة/تنزيل](Screen%20Recording%202025-11-01%20at%2012.04.36%E2%80%AFPM.mov)

### معايير البيانات المدعومة

- **CSV**: ملفات CSV بفاصلة أو أي فاصل آخر
- **Excel**: ملفات .xlsx و .xls
- **الحد الأقصى لحجم الملف**: 16 MB

### تنسيق البيانات المطلوب

- يجب أن تكون البيانات في شكل جدول (صفوف = عينات، أعمدة = ميزات)
- يجب تحديد عمود الهدف (Target) - يمكن اختياره تلقائياً (آخر عمود)
- يجب تنظيف البيانات مسبقاً قدر الإمكان

## الوثائق

### الخوارزمية الجينية

#### معاملات الخوارزمية:

- `population_size`: حجم المجتمع (الافتراضي: 50)
- `n_generations`: عدد الأجيال (الافتراضي: 50)
- `crossover_rate`: احتمالية التزاوج (الافتراضي: 0.8)
- `mutation_rate`: احتمالية الطفرة (الافتراضي: 0.01)
- `selection_method`: طريقة الاختيار ('tournament' أو 'roulette')
- `crossover_method`: طريقة التزاوج ('single_point', 'two_point', 'uniform')
- `elite_size`: عدد الأفراد المميزين (الافتراضي: 2)
- `fitness_alpha`: وزن أداء النموذج في اللياقة (الافتراضي: 0.7)
- `fitness_beta`: وزن تقليل الميزات في اللياقة (الافتراضي: 0.3)

#### مثال متقدم:

```python
ga = GeneticAlgorithm(
    X, y,
    population_size=50,
    n_generations=50,
    crossover_rate=0.8,
    mutation_rate=0.01,
    selection_method='tournament',
    crossover_method='uniform',
    elite_size=2,
    fitness_alpha=0.7,
    fitness_beta=0.3,
    cv_folds=5,
    random_state=42
)
```

### الطرق التقليدية

#### 1. Correlation-based Selection

```python
from src.traditional_methods import CorrelationSelector

selector = CorrelationSelector(k=20)  # اختيار أفضل 20 ميزة
selector.fit(X, y)
selected_features = selector.get_selected_features()
```

#### 2. Mutual Information

```python
from src.traditional_methods import MutualInfoSelector

selector = MutualInfoSelector(k=20)
selector.fit(X, y)
selected_features = selector.get_selected_features()
```

#### 3. Univariate Statistical Tests

```python
from src.traditional_methods import UnivariateSelector

selector = UnivariateSelector(k=20, score_func='f_classif')
selector.fit(X, y)
selected_features = selector.get_selected_features()
```

#### 4. Recursive Feature Elimination

```python
from src.traditional_methods import RecursiveEliminationSelector

selector = RecursiveEliminationSelector(n_features_to_select=20)
selector.fit(X, y)
selected_features = selector.get_selected_features()
```

## الاختبارات

تشغيل جميع الاختبارات:

```bash
pytest tests/ -v
```

تشغيل اختبارات محددة:

```bash
pytest tests/test_genetic_algorithm.py -v
pytest tests/test_traditional_methods.py -v
pytest tests/test_preprocessing.py -v
```

مع تغطية الكود:

```bash
pytest tests/ --cov=src --cov-report=html
```

## سلم التصحيح

- **الكود (30 نقطة)**: جودة التطبيق والهيكلة
- **تطبيق الخوارزمية الجينية (30 نقطة)**: صحة التنفيذ
- **استخدام Github (10 نقاط)**: التوثيق والتاريخ
- **واجهات الويب (10 نقاط)**: التصميم والوظائف
- **التقرير (10 نقاط)**: الوضوح والشمولية
- **الاستضافة على الويب (10 نقاط)**: التوفر والعمل

## المساهمة

هذا مشروع أكاديمي. للأسئلة أو التحسينات، يرجى فتح issue أو pull request.

## الترخيص

هذا المشروع مخصص للاستخدام الأكاديمي فقط.

## المؤلفون

- فريق المشروع (6-8 طلاب)
- المشرف: د. عصام سلمان

## شكر وتقدير

شكراً لجميع المساهمين في تطوير هذا المشروع.
