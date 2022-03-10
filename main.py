# PROJECT: DIABETES FEATURE ENGINEERING
# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin parçasıdır.
# ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan Pima Indian kadınları
# üzerinde yapılan diyabet araştırması için kullanılan verilerdir. 768 gözlem ve 8 sayısal bağımsız değişkenden oluşmaktadır
# Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

# Değişkenler
#
# Pregnancies: Hamilelik sayısı
# Glucose: Glikoz
# BloodPressure: Kan basıncı (Diastolic(Küçük Tansiyon))
# SkinThickness: Cilt Kalınlığı
# Insulin: İnsülin.
# BMI: Beden kitle indeksi.
# DiabetesPedigreeFunction: Soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon.
# Age: Yaş (yıl)
# Outcome: Kişinin diyabet olup olmadığı bilgisi. Hastalığa sahip (1) ya da değil (0)

#Outcome bağımlı değişken; kişinin diyabet olup olmaması
#Bir model oluşturacağım, kişinin verilen özelliklere göre diyabet olup olmama konusunda bir tahminde bulunması amaç
#Bu veriyi hazır hale getirip bir modele sokup modelden bir yapı yakalayıp bir sınıflandırma modeli oluşturmak


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier#sınıflandırma modeli
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action="ignore")#warningleri görmemek için

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df =pd.read_csv("diabetes.csv")
###########################
# EDA
###########################
# 1. Genel Resim
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
# 5. Korelasyon Analizi (Analysis of Correlation) (Opsiyonel)

# 1. Genel Resim
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)  #(768,9) -->768 tane gözlem birimi 9 tane değişken
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T) #17 hamilelik ? glikozun 0 olması? kan basıncının 0 olması?

check_df(df)


print("###############################################################")

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    #numerik değişkenin unique değerlerinin sayısı 10dan küçükse ve sütunun veri tipi objeden farklı ise o zaman bunu kategorik alalım(num_but_cat)
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    #kategorik görünümlü kardinaller name değişkeni gibi(cat_but_car)
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car =  grab_col_names(df)

print("###########################################################")
# 2. Kategorik Değişken Analizi

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)})) #oranı gözlemlemek önemlidir. dengesiz veri setlerini yorumlamak açısından; oran yüzde 20'nin altına düştüğünde üzerinde durulmalıdır
    print("#################################################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df,col)

print("#################################################################")
# 3. Sayısal Değişken Analizi

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col) #insulün için büyük bir kısım 0'dan oluşuyor

print("#################################################################")

# 4. Hedef Değişken Analizi (Analysis of Target Variable)
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df,"Outcome",col)
    #Çıktıdaki hamilelik ve glikoz değerleri yorumlanmak istenirse; diyatbet hastası olanlar yani outcome değeri 1 olanların pregnancy ve glikoz değerleri daha yüksek(ortalama açısından)

#Boş değerler 0 olarak yer aldığı için target değişkenine göre yapılan analizlerin yorumları çok doğru olmayabilir
#ama bize genel bir bilgi veriyor daha iyi yorum için eksik değer olarak işleme alınıp bu aşamalara tekrar bakmak iyi olacaktır
print("#################################################################")
# 5. Korelasyon Analizi (Analysis of Correlation)

df.corr()


# Korelasyon Matrisi
#açık renkler yüksek korelasyonu ifade eder<3
#hangi değişkenlerin targeta etkisi olabilir bunu gözlemledik
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

#############################################
# Veri Önişleme / Feature Engineering
#############################################

#Eğer önce eksik değerlere dokunulacaksa (aykırı değerlere henüz dokunulmamışsa ve aykırılık olduğu düşünülüyorsa) eksik değerleri medyan ile doldurmak daha iyidir.
#

# 1. Outliers (Aykırı Değerler)
# 2. Missing Values (Eksik Değerler)
# 3. Feature Extraction (Özellik Çıkarımı)
# 4. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
# 5. Feature Scaling (Özellik Ölçeklendirme)

#############################################
# Missing Values (Eksik Değerler)
#############################################
# Bir insanda Pregnancies ve Outcome dışındaki değişken değerleri 0 olamayacağı bilinmektedir.
# Bundan dolayı bu değerlerle ilgili aksiyon kararı alınmalıdır. 0 olan değerlere NaN atanabilir .
zero_columns =  [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]
zero_columns #NAN olarak ele alacağım değişkenlerin isimlerini tutar --> Glikoz, Kan basıncı, cilt kalınlığı, insülin, bmi

df.isnull().sum() #boş değer yok

# Gözlem birimlerinde 0 olan degiskenlerin her birisine gidip 0 iceren gozlem degerlerini NaN ile değiştirdim.
for col in zero_columns:
    df[col] = np.where(df[col] == 0, np.nan, df[col]) #değeri 0 olanı bul ve nan ata; 3.argüman bu işlemi hangi sütuna yapacağını belirtiyor, ilk argüman koşulu belirtiyor

df.isnull().sum()#boş değer var çünkü 0 değerini nan ile değiştirdim




def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


na_columns = missing_values_table(df, na_name=True) #sütunlardaki verilerin yüzde kaçı boş bilgisine erişiyoruz
#insülin değerinin yüzde 48'inin boş olması; bunu ortalama/medyan ile doldurmak doğru olmaz. diğer değişkenlerin de durumunu göz önünde bulundurmak gerekmekte çünkü boş oran çok fazla; imputer yapıları kullanılabilir
print(na_columns)
missing_values_table(df)

for col in zero_columns:
    df.loc[df[col].isnull(), col] = df[col].median()
#aykırılıklara bu aşamada henüz dokunulmadı, bu sebeple ortalama yerine medyan olması önemli bir nokta

df.isnull().sum() #eksik değerler ile başa çıkıldığı görülür

#target değişkeninden feature engineering kapsamında herhnagi bir şey türetmemek gerekmekte
#çünkü elimizde target yok canlıya alındığı durumda örneğin.

#############################################
# Outlier (Aykırı Değer)
#############################################

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):#eşik değerlerin belirlenmesi
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Aykırı Değer Analizi ve Baskılama İşlemi
print("Aykırı değer yok ise False var ise True")
for col in df.columns:
    print(col, check_outlier(df, col)) #skinthickness, insulin #aykırılık olan sütunları getirdi
    if check_outlier(df, col):#aykırı değer var ise
        replace_with_thresholds(df,col)


#veriyi hazır hale getirdik, ham veri elimizde
#Model kurulması

# Base Model -->feature engineering yapılmadan önce bir baz oluşturulması; bu modelin başarısını aldık, yeni değişkenler türettiğimde mesela base modelden
#ileriye gidebiliyor muyum, yaptığım değişkenler modelimi ileriye götürmüş mü, bunların gözlemlenmesini sağlar
#base modelin de yüksek olması gerekmekte, 0,50lik bir base model ile ilerlenmemeli

y = df["Outcome"]
X= df.drop("Outcome", axis=1)#target harici diğer değişkenleri ayrı ayrı bir yerde tutmak amacıyla yapılır

y.head()#target değişkeni ifade eder
X.head()

rf_model = RandomForestClassifier(random_state=12345).fit(X,y)#RandomForestClassifier-->makine öğrenmesi algoritması

cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])#çeşitli metrikler ile başarının ölçülmesi

#Accuracy
cv_results['test_accuracy'].mean()#0.75lik bir base model var

# Recall
cv_results['test_recall'].mean()


# Precision
cv_results['test_precision'].mean()

## F1
cv_results['test_f1'].mean()

# AUC
cv_results['test_roc_auc'].mean()

#feature engineering'de yaptığım işlemler ile modelimin performasını artırabiliyor muyum

#############################################
# Feature Engineering  ---> yeni featurelar türetilmesi ve ettkisinin gözlemlenmesi
#############################################

#literatür taraması yapmak, korelasyon matrixine göre yorumda bulunup yeni değişken çıkarma aşamasında göz önünde bulundurulabilir

# Yaş değişkenini kategorilere ayırıp yeni yaş değişkeni oluşturulması
df.loc[(df["Age"] >= 21) & (df["Age"] < 50), "NEW_AGE_CAT"] = "mature"
df.loc[(df["Age"] >= 50), "NEW_AGE_CAT"] = "senior"

df.head()

# BMI 18,5 aşağısı underweight, 18.5 ile 24.9 arası normal, 24.9 ile 29.9 arası Overweight ve 30 üstü obez  --> araştırılmadan çıkan sonuç
df['NEW_BMI'] = pd.cut(x=df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100],
                       labels=["Underweight", "Healthy", "Overweight", "Obese"])

df.head()

# Glukoz degerini kategorik değişkene çevirme
df["NEW_GLUCOSE"] = pd.cut(x=df["Glucose"], bins=[0, 140, 200, 300], labels=["Normal", "Prediabetes", "Diabetes"]) # 0-140 arası, 140-200 arası, 200-300 arasınının labellanması


# Yaş ve beden kitle indeksini bir arada düşünerek kategorik değişken oluşturma; mature-senior ile underweight, healthy, overweight, obese in her bir kombinasyonu
df.loc[(df["BMI"] < 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "underweightmature"
df.loc[(df["BMI"] < 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "underweightsenior"

df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "healthymature"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "healthysenior"

df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "overweightmature"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "overweightsenior"

df.loc[(df["BMI"] > 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "obesemature"
df.loc[(df["BMI"] > 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "obesesenior"


df.head()


# İnsulin Değeri ile Kategorik değişken türetmek
def set_insulin(dataframe, col_name="Insulin"):
    if 16 <= dataframe[col_name] <= 166:
        return "Normal"
    else:
        return "Abnormal"

df["NEW_INSULIN_SCORE"] = df.apply(set_insulin, axis=1) #insülin değişkenindeki her bir gözleme gidip, o gözlem birimindeki değere göre yeni değişken olan NEW_INSULIN_SCORE'a Normal veya Abnormal yazdırıyoruz


df["NEW_GLUCOSE*INSULIN"] = df["Glucose"] * df["Insulin"]

df["NEW_GLUCOSE*PREGNANCIES"] =df["Glucose"] * df["Pregnancies"]

df.head()

# Kolonların isimlerinin büyültülmesi
df.columns = [col.upper() for col in df.columns]

df.head()
#modele veriyi sokacaksak tüm değişkenlerin sayısal değişken olması ve standartlaştırılması gerekmekte; scikit learn üzerinden çalıştırılan modeller ya da random forest üzerinden çalıştırılan modeller sayısal değer zorunluluğu var
#kategorik değişkenleri sayısal değişkene çevirmemiz gerekiyor bu yüzden; encoding işlemi yapılmalı bu amaçla

################################
# Encoding
################################

# Değişkenlerin tiplerine göre ayrılması işlemi
cat_cols, num_cols, cat_but_car = grab_col_names(df) #yeni değişkenlerimiz de olduğu için grab col yeniden çalıştırdık


# LABEL ENCODING --> değişkenlerde sadece 2 sınıf varsa; ordinallik durumu varsa da kullanılabiliyor
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col) #her bir sınıfı(number unique sınıf sayısı 2 olan ve tipi obje olan) 0 ve 1 olarak göstermiş oldu

df.head()


# One-Hot Encoding İşlemi --> değişkenlerde 2 den fazla sınıf varsa
# cat_cols listesinin güncelleme işlemi
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["OUTCOME"]]
cat_cols


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()#herhangi bir kategorik değişken kalmamış durumda

################################
# Standartlaştırma
################################


num_cols

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()


y = df["OUTCOME"]#bağımlı değişken
X = df.drop("OUTCOME", axis=1)

y.head()
X.head()

rf_model = RandomForestClassifier(random_state=12345).fit(X,y)
cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")
print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")

#Base Model
#Accuracy: 0.7591
#Recall: 0.6009
#Precision: 0.6833
#F1: 0.6336
#Auc: 0.8292


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X)

