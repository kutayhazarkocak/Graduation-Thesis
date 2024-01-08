import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import openai
import streamlit as st
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from PIL import Image
from pandas.api.types import CategoricalDtype
import nltk
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.set_page_config(page_title = "Twitter Dashboard",
                   page_icon=":bar_chart",
                   layout="wide")

df_raw = pd.read_excel(
    io='tweet_sentimented.xlsx',
    engine='openpyxl',
    sheet_name='Tweets',
    usecols='A:D',
    nrows=427)



## Veri temizleme
def veri_temizleme_kelimeler_isaretler(text):
    
    text = text.lower()
    text = re.sub('#[^, ]*', '', text)
    text = re.sub('@', '', text)
    text = re.sub('\[.*?\]','', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)

    return text

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001F1E6-\U0001F1FF"
                           u"\U0001F600-\U0001F64F"
                           u"\U0001f926-\U0001f937"
                           u"\U0001F1F2"
                           u"\U0001F1F4"
                           u"\U0001F620"
                           u"\u200d"
                           u"\u2640-\u2642"
                               

                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

for i in range(len(df_raw["Tweet"])):
    df_raw["Tweet"][i]=veri_temizleme_kelimeler_isaretler(df_raw["Tweet"][i])
    df_raw["Tweet"][i]=remove_emoji(df_raw["Tweet"][i])
    

df = df_raw


#Sidebar'a tarih filtresi ekleme
st.sidebar.header("Tweet Paylaşım Tarihleri::calendar:")
Ay = st.sidebar.multiselect("Lütfen tarih filtreleyiniz:",
    options=df["Ay"].unique(),
    default=df["Ay"].unique())

df_selection=df.query(
    "Ay == @Ay"
    )

st.title(":bar_chart: Tweet Analiz Raporu")
st.markdown("##")
#st.dataframe(df_selection)

# kelime aratma
#st.title(" Desteği")
#st.write("Sormak istediğiniz bilgiyi giriniz:")

#user_input = st.text_input("Kullanıcı girdisi")

#Türkçe stopwords
trstop=['a','acaba','altı','altmış','ama','ancak','arada','artık','asla','oha','aslında','aslında','ayrıca','az','bi','bana','bazen','bazı','bazıları','belki','ben','benden','beni','benim','beri','beş','bile','bilhassa','bin','bir','biraz','birçoğu','birçok','biri','birisi','birkaç','birşey','biz','bizden','bize','bizi','bizim','böyle','böylece','bu','buna','bunda','bundan','bunlar','bunları','bunların','bunu','bunun','burada','bütün','çoğu','çoğunu','çok','çünkü','da','daha','dahi','dan','de','defa','değil','diğer','diğeri','diğerleri','diye','doksan','dokuz','dolayı','dolayısıyla','dört','e','edecek','eden','ederek','edilecek','ediliyor','edilmesi','ediyor','eğer','elbette','elli','en','etmesi','etti','ettiği','ettiğini','fakat','falan','filan','gene','gereği','gerek','gibi','göre','hala','halde','halen','hangi','hangisi','hani','hatta','hem','henüz','hep','hepsi','her','herhangi','herkes','herkese','herkesi','herkesin','hiç','hiçbir','hiçbiri','i','ı','için','içinde','iki','ile','ilgili','ise','işte','itibaren','itibariyle','kaç','kadar','karşın','kendi','kendilerine','kendine','kendini','kendisi','kendisine','kendisini','kez','ki','kim','kime','kimi','kimin','kimisi','kimse','kırk','madem','mi','mı','milyar','milyon','mu','mü','nasıl','ne','neden','nedenle','nerde','nerede','nereye','neyse','niçin','nin','nın','niye','nun','nün','o','öbür','olan','olarak','oldu','olduğu','olduğunu','olduklarını','olmadı','olmadığı','olmak','olması','olmayan','olmaz','olsa','olsun','olup','olur','olur','olursa','oluyor','on','ön','ona','önce','ondan','onlar','onlara','onlardan','onları','onların','onu','onun','orada','öte','ötürü','otuz','öyle','oysa','pek','rağmen','sana','sanki','sanki','şayet','şekilde','sekiz','seksen','sen','senden','seni','senin','şey','şeyden','şeye','şeyi','şeyler','şimdi','siz','siz','sizden','sizden','size','sizi','sizi','sizin','sizin','sonra','şöyle','şu','şuna','şunları','şunu','ta','tabii','tam','tamam','tamamen','tarafından','trilyon','tüm','tümü','u','ü','üç','un','ün','üzere','var','vardı','ve','veya','ya','yani','yapacak','yapılan','yapılması','yapıyor','yapmak','yaptı','yaptığı','yaptığını','yaptıkları','ye','yedi','yerine','yetmiş','yi','yı','yine','yirmi','yoksa','yu','yüz','zaten','zira','zxtest']

#wordcloud ekleme
mask = np.array(Image.open("2021 Twitter logo - blue.png"))
mask_colors = ImageColorGenerator(mask)
wc = WordCloud(collocations=False,
               mask=mask, background_color="white",
               max_words=100, max_font_size=256,
               width=mask.shape[1],
               stopwords=trstop,
               height=mask.shape[0],color_func=mask_colors)

wc.generate(df_selection["Tweet"].to_string())
fig, ax = plt.subplots(figsize=(3,3))
ax.imshow(wc, interpolation="bilinear")
#plt.imshow(wc, interpolation="bilinear")
plt.axis('off')
#plt.show()
#st.pyplot(fig=fig, use_container_width=False)


# Raw tablo ile wordcloud'u yerleştirme
col1, col2 = st.columns([2,2])

col1.subheader("Tweet Verileri:")
col1.dataframe(df_selection)

col2.subheader("En sık kullanılan kelimeler:")
col2.pyplot(fig=fig, use_container_width=False)


st.markdown("---")



# Data without stopwords
df_selection_withoutstop = df_selection

df_selection_withoutstop['Tweet'] = df_selection['Tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (trstop)]))

# son 3 ayın en sık kullanılan kelimeleri

def en_cok_kullanilanlar(data, en_sık_adet):
    
    from collections import Counter
    
    pd.set_option('display.max_colwidth', 1000)

    data_set = data.dropna().to_string()

    split_it = data_set.split()

    Counter = Counter(split_it)

    most_occur = Counter.most_common(en_sık_adet)

    return most_occur

ocak_en_sık_10 = pd.DataFrame(en_cok_kullanilanlar(df_selection_withoutstop["Tweet"].where(df_selection_withoutstop["Ay"]=="Ocak"),10),columns=("Kelime","Kullanım Adedi"))
subat_en_sık_10 = pd.DataFrame(en_cok_kullanilanlar(df_selection_withoutstop["Tweet"].where(df_selection_withoutstop["Ay"]=="Şubat"),10),columns=("Kelime","Kullanım Adedi"))
mart_en_sık_10 = pd.DataFrame(en_cok_kullanilanlar(df_selection_withoutstop["Tweet"].where(df_selection_withoutstop["Ay"]=="Mart"),10),columns=("Kelime","Kullanım Adedi"))
nisan_en_sık_10 = pd.DataFrame(en_cok_kullanilanlar(df_selection_withoutstop["Tweet"].where(df_selection_withoutstop["Ay"]=="Nisan"),10),columns=("Kelime","Kullanım Adedi"))
mayis_en_sık_10 = pd.DataFrame(en_cok_kullanilanlar(df_selection_withoutstop["Tweet"].where(df_selection_withoutstop["Ay"]=="Mayıs"),10),columns=("Kelime","Kullanım Adedi"))


st.header("Aylara Göre En Sık Kullanılan Kelimeler")

col1 ,col2, col3, col4, col5 = st.columns(5)

with col1:
    st.subheader("Ocak")
    st.dataframe(ocak_en_sık_10)

with col2:
    st.subheader("Şubat:")
    st.dataframe(subat_en_sık_10)

with col3:
    st.subheader("Mart")
    st.dataframe(mart_en_sık_10)

with col4:
    st.subheader("Nisan")
    st.dataframe(nisan_en_sık_10)
    
with col5:
    st.subheader("Mayıs")
    st.dataframe(mayis_en_sık_10)
    


st.markdown("---")




# kelime bazlı hesaplamalar
st.header("Kelime Bazlı Metrikler:dart:")
kelime_input = st.text_input("Detaylı analiz için anahtar kelime giriniz::small_red_triangle_down:")
list_kelime = []
list_date = []

for i in range(len(df_selection)):
    tweet = df_selection.iloc[i]["Tweet"].lower()
    if tweet.count(kelime_input.lower()) == True:
        list_kelime.append(1)
        list_date.append(df_selection.iloc[i]["Ay"])
    else:
        list_kelime.append(0)
        list_date.append(df_selection.iloc[i]["Ay"])
        
data_table = {"Ay":list_date,
              "Kelime":list_kelime}
filtered_data = pd.DataFrame(data_table) 

toplam_tweet = int(df_selection["Tweet"].count())
toplam_secili_kelime = int(sum(list_kelime))
toplam_gun = int(df_selection["Ay"].nunique())


left_column,middle_column = st.columns(2)

with left_column:
    st.subheader("Toplam Tweet Sayısı:")
    st.subheader(f"{toplam_tweet:,}")

with middle_column:
    st.subheader(f"'{kelime_input}' kelimesi geçen Tweet Sayısı:")
    st.subheader(f"{toplam_secili_kelime:,}")



# aylık grafik eklenmesi


month_order = CategoricalDtype(['Ocak','Şubat','Mart','Nisan','Mayıs','Haziran',
                                'Temmuz','Ağustos','Eylül','Ekim','Kasım','Aralık'],ordered=True)
filtered_data["Ay"]=filtered_data["Ay"].astype(month_order)
aylik_tweet_adedi =(

filtered_data.groupby(by=["Ay"])[["Kelime"]].sum().sort_values(by="Ay")
)

fig_monthly_tweets = px.bar(
    aylik_tweet_adedi,
    x="Kelime",
    y=aylik_tweet_adedi.index,
    orientation="h",
    title=f"<b>Aylık '{kelime_input}' Kelimesi Geçen Tweet Sayısı</b>",
    color_discrete_sequence=["#0083B8"]*len(aylik_tweet_adedi),
    template="plotly_white")


fig_monthly_tweets.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=(dict(showgrid=False)))



list_kelime_haftalik = []
list_hafta = []

for i in range(len(df_selection)):
    tweet = df_selection.iloc[i]["Tweet"].lower()
    if tweet.count(kelime_input.lower()) == True:
        list_kelime_haftalik.append(1)
        list_hafta.append(df_selection.iloc[i]["Hafta"])
    else:
        list_kelime_haftalik.append(0)
        list_hafta.append(df_selection.iloc[i]["Hafta"])
        
data_table_haftalik = {"Hafta":list_hafta,
                       "Kelime":list_kelime_haftalik}
filtered_data_haftalik = pd.DataFrame(data_table_haftalik) 



haftalik_tweet_adedi = filtered_data_haftalik.groupby(by=["Hafta"])[["Kelime"]].sum().sort_values(by="Hafta")


fig_weekly_tweets = px.bar(
    haftalik_tweet_adedi,
    x="Kelime",
    y=haftalik_tweet_adedi.index,
    orientation="h",
    title=f"<b>Haftalık '{kelime_input}' Kelimesi Geçen Tweet Sayısı</b>",
    color_discrete_sequence=["#0083B8"]*len(haftalik_tweet_adedi),
    template="plotly_white")


fig_weekly_tweets.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=(dict(showgrid=False)))


#st.plotly_chart(fig_monthly_tweets)


first_column, second_column = st.columns(2)

with first_column:
    st.plotly_chart(fig_monthly_tweets)
    
with second_column:
    st.plotly_chart(fig_weekly_tweets)

st.markdown("---")

#chatgpt uzantısı

openai.api_key = 'insert_your_key_here'

# Title and description
st.sidebar.header("ChatGPT :robot_face:")
#User input
user_input = st.sidebar.text_input("Sorunuzu buradan aratabilirsiniz :small_red_triangle_down:")

list = []

response = openai.Completion.create(    
        engine='text-davinci-003',  # Choose the appropriate ChatGPT model engine
        #prompt = user_input,
        #prompt = "İstanbul Teknik Üniversitesi kaç yılında kurulmuştur?",
        prompt= f"'{df_selection['Tweet']}' {user_input} ",
        #prompt= f"'{result}' {user_input} ",
        max_tokens=300  # Adjust the desired length of the generated response
    )

# ChatGPT model
#Tablo formatında bu tweetlerin yüzde kaçının pozitif, kaçının negatif ve kaçının nötr olduğunu gösterir misin? Ayrıca 3 madde ile negatif  Tweet'lerin neden negatif olduğunu söyler misin?

# Display response
st.sidebar.write("Cevap:")
st.sidebar.write(response.choices[0].text.strip())

#Sentiment Analysis
st.header("Duygu Analizi:dart:")
train_input = st.number_input("Veri setinizin yüzde kaçı ile modeli eğitmek istiyorsunuz? (örnek:0.7) ")

vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=trstop)
processed_features = vectorizer.fit_transform(df_selection["Tweet"]).toarray()

# train test ayırma işlemi
X_train, X_test, y_train, y_test = train_test_split(processed_features, df_selection["Duygu"], test_size=(1-train_input), random_state=0)

#model implementation
text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(X_train, y_train)

predictions = text_classifier.predict(X_test)

result = classification_report(y_test,predictions)

pozitif_sayisi = sum(predictions)
negatif_sayisi = len(predictions)-sum(predictions)

labels = "Pozitif","Negatif"
sizes = [pozitif_sayisi,negatif_sayisi]
explode = (0.1,0.1)
fig2, ax2 = plt.subplots()
ax2.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax2.axis('equal')


col1, col2 = st.columns(2)

col2.subheader("Doğruluk Skoru:")
col2.write(accuracy_score(y_test, predictions))

col1.subheader("Pozitif ve Negatif Tweet Yüzdeleri:")
col1.pyplot(fig2)


# prediction probablity sıralayalım, en pozitif 5 ve en nagatif 5 i sıralayalım
