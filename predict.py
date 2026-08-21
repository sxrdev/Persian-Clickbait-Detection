import pickle
import re
from feature_database import DATABASE_HAND_FEATURES

def normlz(text):
# سعی میکنیم متن را نرمالایز کنیم و به نحوی پیش پردازش انجام بدیم
    engtoPer_numbers = {
        '0': '۰', '1': '۱', '2': '۲', '3': '۳', '4': '۴',
        '5': '۵', '6': '۶', '7': '۷', '8': '۸', '9': '۹'
    }
    for eng, per in engtoPer_numbers.items():
        text = text.replace(eng, per)
    
    text = text.replace('%', '٪')

    text = re.sub(r'(.)\1+', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


news_list = "نرخ جدید دلار در بازار امروز اعلام شد" #اینجا خبری که میخوام را وارد میکنیم

with open('train_model/model_trained.pkl', 'rb') as pklfile:
    model = pickle.load(pklfile)

with open('train_model/comb_hand_and_model_features.pkl', 'rb') as pklfile:
    model_comb_hand_model_features = pickle.load(pklfile)


feature_db = DATABASE_HAND_FEATURES()
HIGH_THRESHOLD = 75
LOW_THRESHOLD = 45

def predict_lable(input):
    if input >= HIGH_THRESHOLD:
        return "خبر به عنوان طعمه کلیک شناسایی شد"
    elif input >= LOW_THRESHOLD:
        return "خبر کمی مشکوک است"
    else:
        return "خبر عادی است و مشکوکیتی وجود ندارد"


original_news = news_list
normalized_news = normlz(news_list)
        
print(f"ورودی  \"{original_news}\"")
print(f"نرمالایز شده   \"{normalized_news}\"")
    
features = model_comb_hand_model_features.transform([normalized_news])
prob_check = model.predict_proba(features)[0]

# print(features[0])
# -> مدل پس از ترتیب بندی کردن فیچر ها تمام فیچر هایی که کاربر وارد کرده لیست میکند و ما با لیبل 0 اولین خروجی را میگیریم


# print(model.predict_proba(features)[0])
#   #["عادی", "طعمه کلیک"]   مورد اول احتمال لیبل 0 و یا همان عادی خبر است و مورد دوم احتمال لیبل 1 و یا همان طعمه کلیک است
# [0.13326913 0.86673087] -> 0.133 احتمال عادی خبر 
# [0.13326913 0.86673087] -> 0.866 احتمال طعمه کلیک بودن خبر 
 

# print(features)
# خروجی به صورت زیر خواهد بود. که تعداد 300 ویژگی دیفالت مدل به علاوه 2 ویژگی دستی که خودمان اضافه کردیم است
#<Compressed Sparse Row sparse matrix of dtype 'float64'
#        with 7 stored elements and shape (1, 302)>
#  Coords        Values
#  (0, 37)       0.3245689677226644
#  (0, 115)      0.1825254144657761
#  (0, 117)      0.5077009537985473
#  (0, 125)      0.41772769683974187
#  (0, 157)      0.348261575185341
#  (0, 161)      0.2606487305182607
#  (0, 192)      0.48975390309745886


bad_news_prob = prob_check[1] * 100    # احتمال طعمه کلیک بودن خبر
good_news_prob = prob_check[0] * 100   # احتمال عادی بودن خبر


label = predict_lable(bad_news_prob)

print(f"\n{label}")
print(f"احتمال طعمه کلیک بودن خبر :  {bad_news_prob:.1f}%")
print(f"احتمال عادی بودن این خبر: {good_news_prob:.1f}%")