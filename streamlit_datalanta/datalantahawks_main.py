import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import missingno as msno
from scipy.stats import skew

pd.set_option('display.max_columns', None)

# Adjusting display settings for pandas
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 500)
# df = pd.read_csv('/Users/Furkan/Desktop/airbnb-listings.csv', sep=';', low_memory=False)
df = pd.read_csv("DatalantaProject/datasets/la_dataframe_v4.csv", low_memory=False)


def check_df(dataframe, head=5, tail=5, quan=False):
    print("##################### İnfo #####################")
    print(dataframe.info())
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(tail))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    if quan:
        print("##################### Quantiles #####################")
        print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


df.columns.tolist()
df['Property Type'].unique()
# Genel resmi incelerken dikkat etmemiz gerekenler veri setindeki değişkenlerimizin tipleri doğru atanmış mı? Boş gözlemler var mı? Sayısal değişkenlerimin veri setindeki dağılımları nasıl?"""
check_df(df, quan=True)

############################################
df.isnull().sum()
# Eksik verileri görselleştirmek için Missingno
# msno.bar(df)
# plt.figure(figsize=(20, 10))
# plt.show()

# Her sütun için eksik veri yüzdesini hesaplayalım
missing_percentage = df.isnull().sum() * 100 / len(df)
# print
"""Out[86]: 
ID                                  0.000000
Listing Url                         0.000000
Scrape ID                           0.000000
Last Scraped                        0.000000
Name                                0.025739
Summary                             2.584166
Space                              29.213425
Description                         0.041182
Experiences Offered                 0.000000
Neighborhood Overview              39.328735
Notes                              55.945640
Transit                            40.440647
Access                             38.170493
Interaction                        40.533306
House Rules                        29.007516
Thumbnail Url                      16.277154
Medium Url                         16.277154
Picture Url                         0.066921
XL Picture Url                     16.277154
Host ID                             0.000000
Host URL                            0.000000
Host Name                           0.036034
Host Since                          0.036034
Host Location                       0.468444
Host About                         35.195099
Host Response Time                 19.051786
Host Response Rate                 19.051786
Host Acceptance Rate              100.000000
Host Thumbnail Url                  0.036034
Host Picture Url                    0.036034
Host Neighbourhood                 13.806239
Host Listings Count                 0.036034
Host Total Listings Count           0.036034
Host Verifications                  0.036034
Street                              0.000000
Neighbourhood                      19.731288
Neighbourhood Cleansed              0.000000
Neighbourhood Group Cleansed      100.000000
City                                0.000000
State                               0.000000
Zipcode                             1.111912
Market                              0.267682
Smart Location                      0.000000
Country Code                        0.000000
Country                             0.000000
Latitude                            0.000000
Longitude                           0.000000
Property Type                       0.000000
Room Type                           0.000000
Accommodates                        0.000000
Bathrooms                           0.334603
Bedrooms                            0.097807
Beds                                0.144137
Bed Type                            0.000000
Amenities                           0.833934
Square Feet                        98.764542
Price                               1.333265
Weekly Price                       82.142489
Monthly Price                      78.817049
Security Deposit                   52.156903
Cleaning Fee                       21.038814
Guests Included                     0.000000
Extra People                        0.000000
Minimum Nights                      0.000000
Maximum Nights                      0.000000
Calendar Updated                    0.000000
Has Availability                  100.000000
Availability 30                     0.000000
Availability 60                     0.000000
Availability 90                     0.000000
Availability 365                    0.000000
Calendar last Scraped               0.000000
Number of Reviews                   0.000000
First Review                       22.655204
Last Review                        22.598579
Review Scores Rating               23.607536
Review Scores Accuracy             23.710491
Review Scores Cleanliness          23.720787
Review Scores Checkin              23.818594
Review Scores Communication        23.720787
Review Scores Location             23.818594
Review Scores Value                23.864923
License                           100.000000
Jurisdiction Names                  3.032019
Cancellation Policy                 0.000000
Calculated host listings count      0.000000
Reviews per Month                  22.655204
Geolocation                         0.000000
Features                            0.056625"""

# %80'den fazla eksik veri içeren sütunları çıkar
threshold = 80
df = df.loc[:, missing_percentage < threshold]

# %78 atıyoruz
df = df.drop('Monthly Price', axis=1)

df.columns = df.columns.str.lower()


def grab_col_names(dataframe, cat_th=5, car_th=20):
    """
    grab_col_names for given dataframe

    :param dataframe:
    :param cat_th:
    :param car_th:
    :return:
    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    # cat_cols + num_cols + cat_but_car = değişken sayısı.
    # num_but_cat cat_cols'un içerisinde zaten.
    # dolayısıyla tüm şu 3 liste ile tüm değişkenler seçilmiş olacaktır: cat_cols + num_cols + cat_but_car
    # num_but_cat sadece raporlama için verilmiştir.

    return cat_cols, cat_but_car, num_cols


cat_cols, cat_but_car, num_cols = grab_col_names(df)

df["bed type"].value_counts()

missing_data = pd.DataFrame({
    'Missing_Count': df.isnull().sum(),
    'Missing_Percent': df.isnull().mean() * 100
})

# Sıfırdan büyük eksik veri içeren sütunları seçelim
missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values(ascending=False, by='Missing_Percent')

id_cols = ['id', 'host id', 'scrape id']
num_cols = [col for col in num_cols if col not in id_cols]

for column in num_cols:
    if column in missing_data.index:
        df[column].fillna(df[column].median(), inplace=True)

# Eksik veri analizini tekrar yaparak sayısal sütunlardaki eksik verilerin doldurulup doldurulmadığını kontrol edelim

############################

# medians_by_neighbourhood = df.groupby('neighbourhood')[num_cols].median()
#
# # Eksik değerleri doldur
# for col in num_cols:
#     if col != 'Neighbourhood':
#         df[col].replace(0, np.nan, inplace=True)
#         df[col].fillna(df.groupby(cat_cols)[col].transform('median'), inplace=True)
#
# modes_by_cat_cols = df[cat_cols].mode().iloc[0]
#
# # Kategorik değişkenlerdeki eksik değerleri ilgili mod değerleriyle doldur
# df[cat_cols] = df[cat_cols].apply(lambda col: col.fillna(modes_by_cat_cols[col.name]), axis=0)
#
# for column in num_cols:
#     if column in missing_data.index:
#         df[column].fillna(df.groupby('neighbourhood')[column].transform('median'), inplace=True)
#
# df.isnull().sum().sort_values(ascending=False)
#
# df[cat_cols] = df[cat_cols].apply(lambda col: col.fillna(col.mode(dropna=True).iloc[0]), axis=0, inplace=True)


#################################


# bu degıskenımız onemlı oldugu için doldurmuyoruz eksıklıklerı atıyoruz
df["neighbourhood"].dropna(inplace=True)
df.head()

for col in df:
    df[col].fillna('unknown', inplace=True)

df.isnull().sum()

# tarih türü değişkenlerin formatını düzenliyoruz

df["first review"] = pd.to_datetime(df["first review"], errors='coerce')
df["last review"] = pd.to_datetime(df["last review"], errors='coerce')
df["last scraped"] = pd.to_datetime(df["last scraped"], errors='coerce')
df["host since"] = pd.to_datetime(df["host since"], errors='coerce')


######################################
# Aykırı Değer Analizi
######################################

# Aykırı değerlerin baskılanması
def outlier_thresholds(dataframe, variable, low_quantile=0, up_quantile=0.95):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit


# Aykırı değer kontrolü
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    if col in df.columns:
        print(col, check_outlier(df, col))


# Aykırı değerlerin baskılanması
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    replace_with_thresholds(df, col)
df.columns.tolist()

# Her bir sayısal sütun için bir histogram çizelim
# for col in num_cols:
#     plt.figure(figsize=(10, 4))
#     sns.histplot(df[col], kde=True)
#     plt.title(f'Histogram of {col}')
#     plt.xlabel(col)
#     plt.ylabel('Frequency')
#     plt.show()

# Sağa çarpıklıkları belırleyelım
num_cols = [col for col in num_cols if not np.issubdtype(df[col].dtype, np.datetime64)]

# Her sütunda çarpıklığı hesaplayın ve kontrol edin
for col in num_cols:
    skewness = skew(df[col].dropna())
    print(f"Çarpıklık değeri ({col}): {skewness}")
    if skewness > 1:
        print(f"  {col} sütunu sağa çarpıktır.")

# Sağa çarpık olduğu tespit edilen sütunlar
right_skewed_columns = [
    "host listings count", "host total listings count", "accommodates",
    "bathrooms", "bedrooms", "beds", "price", "security deposit",
    "cleaning fee", "guests included", "extra people", "minimum nights",
    "number of reviews", "calculated host listings count", "reviews per month"
]

# Log1p dönüşümü uygula ve yeni sütunlar oluştur
for col in right_skewed_columns:
    df[f"log_{col}"] = np.log1p(df[col])

# df.to_csv("/Users/mrpurtas/Desktop/ensondff.csv", index=True)
# df = pd.read_csv('datasets/ensondff.csv', sep=',', low_memory=False)


df.columns

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

amenities_list = df['amenities'].str.split(',')

all_amenities = [amenity for sublist in amenities_list.dropna() for amenity in sublist]

amenities_counts = Counter(all_amenities)

amenities_df = pd.Series(amenities_counts).sort_values(ascending=False)

total_rows = len(df)
amenities_normalized = {amenity: count / total_rows for amenity, count in amenities_counts.items()}

print("Normalleştirilmiş Frekanslar (Sıralı):")
sorted_amenities_normalized = dict(sorted(amenities_normalized.items(), key=lambda x: x[1], reverse=True))
print(sorted_amenities_normalized)

sorted_amenities_list = list(sorted_amenities_normalized.items())

top_amenities = amenities_df.head(40)

print(top_amenities)

top_amenities_normalized = dict(list(sorted_amenities_normalized.items())[:40])

plt.figure(figsize=(16, 8))
plt.barh(list(top_amenities_normalized.keys()), list(top_amenities_normalized.values()), color='skyblue')
plt.title('Top 40 Amenities with Normalized Frequencies')
plt.xlabel('Normalized Frequency')
plt.ylabel('Amenity')
plt.show()

for amenity in top_amenities.index:
    df[f'Has_{amenity}'] = df['amenities'].str.contains(amenity).astype(int)

##### Amenities değişken droplama
# amenities_to_drop = top_amenities.index[:y]
# df = df.drop([f'Has_{amenity}' for amenity in amenities_to_drop], axis=1)

###########
# PCA Testi - Amenities Değişkenleri için
###########
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
#
# # Seçtiğiniz değişkenleri bir alt küme DataFrame'e alın
# selected_columns = [f'Has_{amenity}' for amenity in top_amenities.index]
# subset_df = df[selected_columns]
#
# # Standartlaştırma
# scaler = StandardScaler()
# subset_df_standardized = scaler.fit_transform(subset_df)
#
# pca = PCA()
# pca_fit = pca.fit_transform(subset_df_standardized)
#
# pca.explained_variance_ratio_
# np.cumsum(pca.explained_variance_ratio_)
#
# pca = PCA().fit(subset_df_standardized)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel("Bileşen Sayısı")
# plt.ylabel("Kümülatif Varyans Oranı")
# plt.show()

# print(df[[f'Has_{amenity}' for amenity in top_20_amenities]].head())

# Price_per_person
df['Price_Per_Person'] = df['price'] / df['beds']

# min-max night farkı
df['Night_diff'] = df['maximum nights'] - df['minimum nights']

# Özel Gün Değişkeni
# df['last review'] = pd.to_datetime(df['last review'] + ' 2023', format='%d %b %Y')
df['IsSpecialDay'] = 0
df.head()
special_date_ranges = [
    ('2013-10-25', '2013-11-02'),
    ('2013-11-21', '2013-11-25'),
    ('2013-12-20', '2013-12-31'),
    ('2013-01-01', '2013-01-02'),
    ('2013-01-13', '2013-01-14'),
    ('2014-10-25', '2014-11-02'),
    ('2014-11-21', '2014-11-25'),
    ('2014-12-20', '2014-12-31'),
    ('2014-01-01', '2014-01-02'),
    ('2014-01-13', '2014-01-14'),
    ('2015-10-25', '2015-11-02'),
    ('2015-11-21', '2015-11-25'),
    ('2015-12-20', '2015-12-31'),
    ('2015-01-01', '2015-01-02'),
    ('2015-01-13', '2015-01-14'),
    ('2016-10-25', '2016-11-02'),
    ('2016-11-21', '2016-11-25'),
    ('2016-12-20', '2016-12-31'),
    ('2016-01-01', '2016-01-02'),
    ('2016-01-13', '2016-01-14'),
    ('2017-10-25', '2017-11-02'),
    ('2017-11-21', '2017-11-25'),
    ('2017-12-20', '2017-12-31'),
    ('2017-01-01', '2017-01-02'),
    ('2017-01-13', '2017-01-14'),
    ('2013-02-08', '2013-02-11'),
    ('2014-01-29', '2014-02-01'),
    ('2015-02-17', '2015-02-20'),
    ('2016-02-06', '2016-02-09'),
    ('2017-01-26', '2017-01-29'),
]

for start, end in special_date_ranges:
    mask = (df['last review'] >= start) & (df['last review'] <= end)
    df.loc[mask, 'IsSpecialDay'] = 1

df['IsSpecialDay'].value_counts()

###########  House Rules   ##############3

df['house rules'] = df['house rules'].apply(lambda x: str(x) if not pd.isnull(x) else '')
# df['Has_NoPet'] = df['house_rules'].apply(lambda x: 1 if 'no pet' in x.lower() else 0)
# df['Has_NoSmoking'] = df['house_rules'].apply(lambda x: 1 if 'no smoking' in x.lower() else 0)
df['Has_ShoesOff'] = df['house rules'].apply(
    lambda x: 1 if 'shoes off' in x.lower() or 'remove shoes' in x.lower() else 0)
df['Has_NoParties'] = df['house rules'].apply(lambda x: 1 if 'no parties' in x.lower() else 0)
df['cleaning_expectations'] = df['house rules'].str.contains('clean up|cleaning|leave the apartment clean',
                                                             case=False).astype(int)

# df['Has_NoPet'].value_counts()
# df['Has_NoSmoking'].value_counts()
df['Has_ShoesOff'].value_counts()
df['Has_NoParties'].value_counts()
df['cleaning_expectations'].value_counts()

# 'number_of_reviews' ve 'reviews_per_month' sütunlarının toplamını hesaplama:
df['_total_reviews'] = df['log_number of reviews'] + (df['log_reviews per month'] * 12)
df.head()
"""'minimum_nights' ve 'availability_365' sütunlarını kullanarak rezervasyon esnekliğini hesaplama:"""
df['booking_flexibility'] = df['availability 365'] / df['log_minimum nights']

df.columns = df.columns.str.replace(' ', '_')

############
df.dropna(subset=["last_review"], inplace=True)

df['day_of_week'] = df['last_review'].dt.dayofweek
df['month'] = df['last_review'].dt.month

from datetime import datetime


def get_season(month):
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:
        return 'Winter'


def is_weekend(day_of_week):
    return 'Weekend' if day_of_week > 4 else 'Weekday'


# Mevsimleri ve hafta içi/sonu bilgisini ekle
df['_season'] = df['month'].apply(get_season)
df['_weekend_or_weekday'] = df['day_of_week'].apply(is_weekend)

# Her bir review skoru için ağırlıklar (örnek olarak)

weights = {
    'review_scores_value': 2,  # Fiyat/değer oranının önemi yüksek olabilir.
    'review_scores_checkin': 1,  # Check-in sürecinin pürüzsüz olması önemli ancak diğerlerinden daha az.
    'review_scores_location': 3,  # Konum genellikle fiyat için çok önemli bir faktördür.
    'review_scores_cleanliness': 4,  # Temizlik, özellikle konaklama sektöründe, çok önemlidir.
    'review_scores_communication': 1,  # İletişim önemli ancak fiyat üzerinde dolaylı etkisi olabilir.
    'review_scores_accuracy': 2,  # İlanın doğruluğu, müşteri memnuniyeti için önemlidir.
    'review_scores_rating': 5  # Genel puanlama, konukların genel memnuniyetini yansıtır.
}


# Ağırlıklı toplamı hesaplamak için bir fonksiyon
def weighted_review_score(row, weight_dict):
    total_score = 0
    total_weight = sum(weight_dict.values())
    for col, weight in weight_dict.items():
        # Eğer skor NaN ise, bu skoru hesaba katmamak için sıfır puan verilebilir.
        total_score += row[col] * weight if not pd.isna(row[col]) else 0
    return total_score / total_weight


# Ağırlıklı review toplam skoru hesaplama
df['_weighted_total_review_score'] = df.apply(lambda row: weighted_review_score(row, weights), axis=1)

####
df['activity_duration'] = (pd.to_datetime(df['last_review']) - pd.to_datetime(df['first_review'])).dt.days
df.drop(['activity_duration'], axis=1, inplace=True)

df['_days_since_last_review'] = (pd.to_datetime('today') - pd.to_datetime(df['last_review'])).dt.days

# 'host_since' tarihinden bugüne kadar geçen süreyi hesaplayarak ev sahibinin platformda ne kadar süredir aktif olduğunu gösteren bir özellik oluşturun
df['host_duration'] = (pd.to_datetime('today') - pd.to_datetime(df['host_since'])).dt.days
df.head()
df.columns.tolist
df = df.drop(['unnamed:_0'], axis=1)
df = df.drop(['unnamed:_0.1'], axis=1)
df = df.drop(['unnamed:_0.2'], axis=1)

df.isnull().sum()

"""df['_host_response_time'] = df['host_response_time'].fillna(0)

# 'host_response_time' sütununu kategorik değişkene dönüştürün
df['_host_response_time'] = df['host_response_time'].map({
    'within an hour': 1,
    'within a few hours': 2,
    'within a day': 3,
    'a few days or more': 4,
    np.nan: 0  # NaN değerleri için doğru atama
})
df['_host_response_time'].fillna(0, inplace=True)

df.head()
df.host_response_time.isna().sum()
import numpy as np"""

from sklearn.preprocessing import LabelEncoder

# 'host_listings_count' kullanarak ev sahibinin toplam ilan sayısını direkt olarak kullanın veya eşik değere göre kategorize edin
df['is_multi_host'] = df['log_host_listings_count'] > 1  # 1'den fazla ilanı olan host'lar için True

label_encoder = LabelEncoder()
df['is_multi_host'] = label_encoder.fit_transform(df['is_multi_host'])

# 'host_about' metin sütununu analiz ederek, metin uzunluğunu bir özellik olarak kullanın
df['_host_about_length'] = df['host_about'].apply(lambda x: len(str(x)))

df['_host_verification_count'] = df['host_verifications'].apply(lambda x: len(x.split(',')))

weights = {
    'host_duration': 0.2,  # Platformdaki süre
    'host_response_rate': 0.2,  # Yanıt oranı
    "is_multi_host": 0.15,  # Birden fazla ilanı olma durumu
    '_host_about_length': 0.1,  # Ev sahibi hakkında bilgi uzunluğu
    '_host_verification_count': 0.1  # Doğrulama yöntemleri sayısı
}


# Ağırlıklı toplamı hesaplamak için bir fonksiyon
def calculate_host_activity_score(row, weights):
    activity_score = 0
    for key, weight in weights.items():
        activity_score += row[key] * weight
    return activity_score


df['_host_activity_score'] = df.apply(calculate_host_activity_score, axis=1, args=(weights,))

########## TARGET ENCODING #################
smooth_property_type = df.shape[0] * 0.3  # Örnek sayısının %30'u

# room_type için smooth değerini hesaplama
# Benzersiz değer sayısı daha az ve dağılım daha dengeli olduğu için daha düşük bir değer kullanacağız.
smooth_room_type = df.shape[0] * 0.1  # Örnek sayısının %10'u

# Kategorik değişkenlere göre smooth değerlerini ayarlıyoruz
smooth_values = {'property_type': smooth_property_type, 'room_type': smooth_room_type}

categorical_cols = ['property_type', 'room_type']
overall_mean = df['log_price'].mean()


# Target encoding fonksiyonu, bu sefer smooth değerini parametre olarak alacak şekilde güncellendi
def target_encode(column, df, target, smooth):
    # Her kategori için hedef değişkenin ortalaması
    category_means = df.groupby(column)[target].mean()

    # Her kategori için gözlem sayısı
    category_counts = df[column].value_counts()

    # Düzeltilmiş ortalama hesaplama
    smooth_mean = (category_counts * category_means + smooth * overall_mean) / (category_counts + smooth)

    # Orijinal değerleri düzeltilmiş ortalama ile değiştirme
    return df[column].map(smooth_mean)


# Kategorik sütunlara özel smooth değerleri ile target encoding uygulama
for col in categorical_cols:
    df[col + '_encoded'] = target_encode(col, df, 'log_price', smooth_values[col])
df.head()
df.head()

# Uzun süreli uygunluk skoru: Daha uzun süre uygun olan ilanlar daha yüksek skor alır
# Bu skoru hesaplamak için, 365 günlük uygunluk süresini baz alıp diğer uygunluk sürelerini orantılayacağız
df['_long_term_availability_score'] = (
                                              df['availability_30'] / 30 +
                                              df['availability_60'] / 60 +
                                              df['availability_90'] / 90 +
                                              df['availability_365'] / 365
                                      ) / 4  # 4 farklı süre için ortalama alıyoruz

df.head()

current_date = pd.to_datetime('today')

# İlanın Piyasada Kalma Süresi
df['_listing_duration_days'] = (current_date - df['first_review']).dt.days


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, ["bed_type"], drop_first=True)

# En sık rastlanan 10 mahalleyi seçme
top_neighbourhoods = df['neighbourhood_cleansed'].value_counts().nlargest(10).index

# 'Diğer' olarak adlandırılacak kategorileri belirleme
df['neighbourhood_reduced'] = df['neighbourhood_cleansed'].apply(lambda x: x if x in top_neighbourhoods else 'Other')

neighbourhood_dummies = pd.get_dummies(df['neighbourhood_reduced'], drop_first=True)

# One-hot encoded sütunları orijinal DataFrame'e ekleme
df = pd.concat([df, neighbourhood_dummies], axis=1)

cat_cols, cat_but_car, num_cols = grab_col_names(df)

######################################################################################################################
# Sağa çarpıklıkları belırleyelım
num_cols = [col for col in num_cols if not np.issubdtype(df[col].dtype, np.datetime64)]

# Her sütunda çarpıklığı hesaplayın ve kontrol edin
for col in num_cols:
    skewness = skew(df[col].dropna())
    print(f"Çarpıklık değeri ({col}): {skewness}")
    if skewness > 1:
        print(f"  {col} sütunu sağa çarpıktır.")

# Her bir sayısal sütun için bir histogram çizelim
# for col in num_cols:
#     plt.figure(figsize=(10, 4))
#     sns.histplot(df[col], kde=True)
#     plt.title(f'Histogram of {col}')
#     plt.xlabel(col)
#     plt.ylabel('Frequency')
#     plt.show()


mildly_right_skewed_columns = [
    "log_guests_included",
    "log_minimum_nights",
    "log_calculated_host_listings_count",
    "month",
    "log_beds",
    "_host_verification_count",
    "_listing_duration_days",
    # "activity_duration"
]

# Karekök dönüşümü uygula ve yeni sütunlar oluştur
for col in mildly_right_skewed_columns:
    df[f"sqrt_{col}"] = np.sqrt(df[col])

heavily_right_skewed_columns = [
    "security_deposit",
    "cleaning_fee",
    "guests_included",
    "extra_people",
    "minimum_nights",
    "number_of_reviews",
    "calculated_host_listings_count",
    "log_host_listings_count",
    "log_host_total_listings_count",
    "log_bathrooms",
    "_days_since_last_review",
    "_host_about_length"
]

# Log1p dönüşümü uygula ve yeni sütunlar oluştur
for col in heavily_right_skewed_columns:
    df[f"log_{col}"] = np.log1p(df[col])

df.dtypes

df['_weekend_or_weekday'] = df['_weekend_or_weekday'].map({'Weekday': 0, 'Weekend': 1})


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, ["_season"], drop_first=True)

df.calendar_last_scraped = pd.to_datetime(df.calendar_last_scraped)

cat_cols, cat_but_car, num_cols = grab_col_names(df)

num_cols.extend([
    'room_type_encoded',
    'Echo Park',
    'Hollywood',
    'Hollywood Hills',
    'Koreatown',
    'Mid-Wilshire',
    'Other',
    'Sawtelle',
    'Silver Lake',
    'Venice',
    'Westlake',
    '_weekend_or_weekday',
    '_is_multi_host',
    '_season_Spring',
    '_season_Summer',
    '_season_Winter'
])


# Aykırı değerlerin baskılanması
def outlier_thresholds(dataframe, variable, low_quantile=0, up_quantile=0.95):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit


# Aykırı değer kontrolü
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    if col in df.columns:
        print(col, check_outlier(df, col))

remove_cols = [
    "Echo Park", "Hollywood Hills", "Koreatown", "Mid-Wilshire",
    "Sawtelle", "Silver Lake", "Westlake", "room_type_encoded", "Hollywood", "Venice", "Other", "_season_Summer",
    "_season_Winter", "_is_multi_host", "_weekend_or_weekday", "_season_Spring", "id", "host_id",
]

# Çıkarılacak değişkenleri num_cols listesinden çıkar
num_cols = [col for col in num_cols if col not in remove_cols]


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    replace_with_thresholds(df, col)

num_cols = [col for col in num_cols if col not in ["first_review", "last_review", "host_since"]]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[num_cols])

target_corr = df.corrwith(df['log_price'])
target_corr = target_corr.sort_values(ascending=False)

df["_host_activity_score"].dropna(inplace=True)
df['sqrt__listing_duration_days'].fillna(df['sqrt__listing_duration_days'].mean(), inplace=True)

df.to_csv('/Users/Furkan/Desktop/DATALANTA.csv', index=True)
df = pd.read_csv("/Users/Furkan/Desktop/DXlantaa.csv")

Y = df['price']
df.columns.tolist()
feature_columns = [
    'price',
    'latitude',
    'longitude',
    'log_accommodates',
    'log_bedrooms',
    'log_security_deposit',
    'log_cleaning_fee',
    'log_extra_people',
    'log_number_of_reviews',
    'log_reviews_per_month',
    # 'property_type_encoded',
    '_total_reviews',
    'booking_flexibility',
    '_weighted_total_review_score',
    'host_duration',
    '_host_activity_score',
    '_long_term_availability_score',
    'log_log_host_total_listings_count',
    'log_log_bathrooms',
    'log__days_since_last_review',
    'log__host_about_length',
    'sqrt_log_guests_included',
    'sqrt_log_minimum_nights',
    'sqrt_log_calculated_host_listings_count',
    'sqrt_month',
    'sqrt_log_beds',
    'sqrt__host_verification_count',
    'sqrt__listing_duration_days',
    # 'sqrt__activity_duration',
    # 'room_type_encoded',
    'Echo Park',
    'Hollywood',
    'Hollywood Hills',
    'Koreatown',
    'Mid-Wilshire',
    'Other',
    'Silver Lake',
    'Venice',
    'Westlake',
    '_weekend_or_weekday',
    'is_multi_host',
    '_season_Spring',
    '_season_Summer',
    '_season_Winter',
    'Has_Family/kid_friendly',
    'Has_TV',
    'Has_Free_parking_on_premises',
    'Has_Cable_TV',
    'Has_Indoor_fireplace',
    'Has_Dryer',
    'Has_Washer',
    'Has_24-hour_check-in',
    'Has_Pets_allowed',
    'Has_NoParties',
    'Has_Air_conditioning',
    'Has_Pool',
    'Has_Gym',
    'Has_Shampoo',
    'Has_Self_Check-In',
    'Has_Carbon_monoxide_detector',
    'Has_Private_entrance',
    'Has_Wheelchair_accessible',
    'Has_Heating',
    'Has_Kitchen',
    'Has_Hot_tub',
    'Has_Iron',
    'landmark_score',
    'property_type_Bed & Breakfast',
    'property_type_Boat',
    'property_type_Boutique hotel',
    'property_type_Bungalow',
    'property_type_Cabin',
    'property_type_Camper/RV',
    'property_type_Castle',
    'property_type_Cave',
    'property_type_Chalet',
    'property_type_Condominium',
    'property_type_Dorm',
    'property_type_Earth House',
    'property_type_Guest suite',
    'property_type_Guesthouse',
    'property_type_Hostel',
    'property_type_House',
    'property_type_Hut',
    'property_type_In-law',
    'property_type_Loft',
    'property_type_Other',
    'property_type_Plane',
    'property_type_Serviced apartment',
    'property_type_Tent',
    'property_type_Townhouse',
    'property_type_Treehouse',
    'property_type_Villa',
    'property_type_Yurt',
    'room_type_Private room',
    'room_type_Shared room'
]

# Create the feature DataFrame
X = df[feature_columns]
X.isnull().sum()

new_df = df[feature_columns]
new_df.to_csv('/Users/Furkan/Desktop/DATALANTA_SON.csv', index=True)

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.naive_bayes import GaussianNB  # Not: Naive Bayes genellikle regresyon için kullanılmaz

# İleri düzey ansamble modelleri
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


def base_models_regression_with_cv(X, y, cv=5):
    print("Base Models with Cross-Validation for Regression....")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    regressors = [
        # ('Linear Regression', LinearRegression()),
        # ('Ridge', Ridge(alpha=1.0)),
        # ('Random Forest Regressor', RandomForestRegressor(n_estimators=100)),
        # ('Gradient Boosting Regressor', GradientBoostingRegressor()),
        ('XGBoost Regressor',
         XGBRegressor(colsample_bytree=1, learning_rate=0.1, max_depth=7, min_child_weight=3, n_estimators=157,
                      subsample=1)),
        # ('LightGBM Regressor', LGBMRegressor())
    ]
    # Best Parameters: {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 7, 'min_child_weight': 3, 'n_estimators': 157, 'subsample': 1.0}
    for name, regressor in regressors:
        scores = cross_val_score(regressor, X, y, cv=cv, scoring='r2')
        mse_scores = cross_val_score(regressor, X, y, cv=cv, scoring='neg_mean_squared_error')
        mae_scores = cross_val_score(regressor, X, y, cv=cv, scoring='neg_mean_absolute_error')

        print(f"{name} Regressor:")
        print(f"Mean R^2 Score: {np.mean(scores)} (std: {np.std(scores)})")
        print(f"Mean Negative Mean Squared Error: {np.mean(mse_scores)} (std: {np.std(mse_scores)})")
        print(f"Mean Negative Mean Absolute Error: {np.mean(mae_scores)} (std: {np.std(mae_scores)})\n")


# Fonksiyonu kullanma örneği
base_models_cv = base_models_regression_with_cv(X, Y, cv=5)

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


################ Random Forest Optimize
#  Parameters: {'max_depth': 30, 'min_samples_leaf': 2, 'min_samples_split': 7, 'n_estimators': 179}
# Best R^2 Score: 0.787240164867604

def optimize_random_forest_random_search(X, y, param_dist, cv=5, n_iter=10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Random Forest Regressor
    rf_regressor = RandomForestRegressor()

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=rf_regressor,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    # Fit the model to the data
    random_search.fit(X_train, y_train)

    # Print the best parameters and corresponding score
    print("Best Parameters:", random_search.best_params_)
    print("Best R^2 Score:", random_search.best_score_)

    # Get the best model
    best_rf_model = random_search.best_estimator_

    return best_rf_model


# Define the parameter distribution for Random Forest
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 4)
}

# Optimize Random Forest using Random Search
best_rf_model_random_search = optimize_random_forest_random_search(X, Y, param_dist, cv=5, n_iter=10)

################## Gradient Boost Optimize
# Best Parameters: {'learning_rate': 0.2, 'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 139, 'subsample': 0.8}
# Best R^2 Score: 0.7972860378567486

from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import randint


def optimize_gradient_boosting_random_search(X, y, param_dist, cv=5, n_iter=10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Gradient Boosting Regressor
    gb_regressor = GradientBoostingRegressor()

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=gb_regressor,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    # Fit the model to the data
    random_search.fit(X_train, y_train)

    # Print the best parameters and corresponding score
    print("Best Parameters:", random_search.best_params_)
    print("Best R^2 Score:", random_search.best_score_)

    # Get the best model
    best_gb_model = random_search.best_estimator_

    return best_gb_model


# Define the parameter distribution for Gradient Boosting
param_dist_gb = {
    'n_estimators': randint(50, 200),
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
    'max_depth': [3, 5, 7, 9],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 4),
    'subsample': [0.8, 0.9, 1.0]
}

# Optimize Gradient Boosting using Random Search
best_gb_model_random_search = optimize_gradient_boosting_random_search(X, Y, param_dist_gb, cv=5, n_iter=10)

##################### XGBoost Regressor Optimize
# Best Parameters: {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 7, 'min_child_weight': 3, 'n_estimators': 157, 'subsample': 1.0}
# Best R^2 Score: 0.8080793302166764

from xgboost import XGBRegressor
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV


def optimize_xgboost_random_search(X, y, param_dist, cv=5, n_iter=10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an XGBoost Regressor
    xgb_regressor = XGBRegressor()

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=xgb_regressor,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    # Fit the model to the data
    random_search.fit(X_train, y_train)

    # Print the best parameters and corresponding score
    print("Best Parameters:", random_search.best_params_)
    print("Best R^2 Score:", random_search.best_score_)

    # Get the best model
    best_xgb_model = random_search.best_estimator_

    return best_xgb_model


# Define the parameter distribution for XGBoost
param_dist_xgb = {
    'n_estimators': randint(50, 200),
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
    'max_depth': [3, 5, 7, 9],
    'min_child_weight': randint(1, 10),
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
}

# Optimize XGBoost using Random Search
best_xgb_model_random_search = optimize_xgboost_random_search(X, Y, param_dist_xgb, cv=5, n_iter=10)

################ Light GBM Optimize
# Best Parameters: {'colsample_bytree': 0.9, 'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 104, 'num_leaves': 83, 'subsample': 0.8}
# Best R^2 Score: 0.8038264923010965

from lightgbm import LGBMRegressor
from scipy.stats import randint


def optimize_lightgbm_random_search(X, Y, param_dist, cv=5, n_iter=10):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Create a LightGBM Regressor
    lgbm_regressor = LGBMRegressor()

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=lgbm_regressor,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    # Fit the model to the data
    random_search.fit(X_train, y_train)

    # Print the best parameters and corresponding score
    print("Best Parameters:", random_search.best_params_)
    print("Best R^2 Score:", random_search.best_score_)

    # Get the best model
    best_lgbm_model = random_search.best_estimator_

    return best_lgbm_model


# Define the parameter distribution for LightGBM
param_dist_lgbm = {
    'n_estimators': randint(50, 200),
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
    'max_depth': [3, 5, 7, 9],
    'num_leaves': randint(20, 100),
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
}

# Optimize LightGBM using Random Search
best_lgbm_model_random_search = optimize_lightgbm_random_search(X, Y, param_dist_lgbm, cv=5, n_iter=10)

##############

from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile
import graphviz

Y = df['log_price']

# Bağımsız değişkenler
X = df[feature_columns]

# Grid Search için parametre aralıkları
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Modelinizin tipine göre uygun bir regresör seçin
model = RandomForestRegressor()

# Grid Search
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
grid_search.fit(X, Y)

# En iyi parametre setini ve modeli bulma
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Hyperparameters:", best_params)
print("Best Model:", best_model)

import os
import pickle

# Örneğin, best_xgb_model_random_search adlı bir modeliniz varsa
best_xgb_model_random_search.fit(X, Y)  # Modeli tekrar eğitin (isteğe bağlı)
with open('xgb_model.pkl', 'wb') as file:
    pickle.dump(best_xgb_model_random_search, file)

model_file_path = os.path.join('/Users/Furkan/Desktop', 'XGB_model.pkl')

import pickle
from xgboost import XGBRegressor

# XGBoost modelini oluşturun (istenilen parametreleri ayarlayın)
xgb_model = XGBRegressor(colsample_bytree=1, learning_rate=0.1, max_depth=7, min_child_weight=3, n_estimators=157,
                         subsample=1)

# Modeli eğitin (X ve Y verilerinizi kullanın)
xgb_model.fit(X, Y)

# Modeli pickle formatında kaydedin
with open('xgboost_model.pkl', 'wb') as file:
    pickle.dump(xgb_model, file)

with open('xgboost_model.pkl', 'rb') as file:
    loaded_xgb_model = pickle.load(file)

# Tahminler yapın (örneğin, X_test verileri üzerinde)

model_file_path = os.path.join('/Users/Furkan/Desktop', 'XGB_model.pkl')

import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# XGBoost modelini oluştur
model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=1, learning_rate=0.1, max_depth=7,
                         min_child_weight=3, n_estimators=157, subsample=1)
model.fit(X_train, y_train)

# Modelin performansını değerlendir
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

predictions = loaded_xgb_model.predict(X_test)

print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')

joblib.dump(model, 'xgboost_model.joblib')
