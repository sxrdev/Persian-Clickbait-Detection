# Persian-Clickbait-Detection
Persian Clickbait Detection | تشخیص کلیک بیت فارسی | Logistic Regression + TF-IDF


[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)

پروژه تشخیص خبر کلیک بیت(طعمه) فارسی و بررسی عملکرد مدل لاجستیک ریگرسیون در تشخیص و طبقه بندی خبر ورودی با آموزش بیش از 8 هزار عنوان خبر طعمه و عادی به مدل.
## خروجی کلی تا به این لحظه

- Accuracy: **95.96**
- Precision: **98.08** | Recall **93.70**
- Cross-Validation: **94.5 +- 0.5**

## در صورت تمایل به نصب مدل مراحل زیر را دنبال کنید

git clone https://github.com/sxrdev/Persian-Clickbait-Detection.git

cd Persian-Clickbait-Detection

python -m venv .venv

.venv\Scripts\activate  # 

# نصب افزونه ها و ماژول های مورد نیاز
pip install -r requirements.txt


## دیتاست استفاده شده

- **تعداد:** 8100 عنوان خبر فارسی
- **کلاس‌ها:** 50.9 درصد خبر طعمه | 49.1 درصد خبر عادی
- **منابع:** 17 خبرگزاری و پایگاه خبری فارسی


 این دیتاست با صرف زمان طولانی از 17 خبرگزاری فارسی جمع‌آوری و لیبل زنی دستی شده است. به دلیل حجم بالا و ارزش پژوهشی ترجیح داده شده به صورت عمومی منتشر نشود.
