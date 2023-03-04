import pandas as pd
import re

#Veri tükleme
data = pd.read_csv('./restaurantReviews.csv')
print("Verinin uzunluğunu len(data) ile bulabiliriz")
print(len(data))


#Stop Word dediğimiz anlamlı kelimeleri çıkartmak için bazı kodlar (the,that,this..)
import nltk
#Stop word'leri indirelim
#nltk.download('stopwords') #Bu satırı bir kere çalıştırmanız indirmesi için yetecektir, sonra yorum satırına alabilirsiniz
from nltk.corpus import stopwords

#Kelimelerin eklerini ayırmak için PorterStammer kullanacağız
from nltk.stem.porter import PorterStemmer
porterStemmer = PorterStemmer()

# ------------- PREPROCESSING ------------- #
#Tüm satırları işleyip en sonunda bu arrayde tutacağız
processedReviews = []

#Her kelime için aynı işlemler yapılacağı için döngü kullanıyoruz
for i in range(len(data)):
    #Noktalama işaretlerini regex kullanarak sileceğiz
    #sub fonksiyonu ile harf olmayan işaretlerin yerine space koyuyoruz
    review = re.sub('[^a-zA-Z]', ' ', data['Review'][i])

    #tüm harfleri küçüğe çevirelim (Python büyük-küçük harf duyarlı bir dildir)
    review = review.lower().split()

    #Stop words'leri silelim
    #stem, kelimenin gövdesini bulan fonksiyondur (Son güncelleme ile stopwords'e turkish de eklendi!)
    #Her bir kelime stopwords kümesinde var mı diye kontrol edilip yoksa review'e tekrar koyuluyor
    review = [porterStemmer.stem(word) for word in review if not word in set(stopwords.words('english'))]

    #ileride her bir kelimenin kaç kere kullanıldığını bulmamız gerekecek, bu satır orası için bir hazırlıktır
    review = ' '.join(review)

    processedReviews.append(review)


# ------------- FEATURE EXTRACTION ------------- #
#Bag Of Words işlemini uygulayalım
#Module importing
from sklearn.feature_extraction.text import CountVectorizer
#CountVectorizer ile en çok kullanılan kelimeleri alalım
#max_features ile en çok kullanılan 1000 kelimeyi alacağız (Ne kadar artarsa RAM'iniz o kadar zorlanır :) )
countVectorizer = CountVectorizer(max_features=2000)
x = countVectorizer.fit_transform(processedReviews).toarray() #Sparce Matrix elde ettik
y = data.iloc[:,1:].values.ravel()
#x ve y değişkenlerini elde ettiğimize göre artık bildiğimiz ML algoritmalarını kullanarak
#tahmin işlemlerine başlayabiliriz



# ------------- MACHINE LEARNING ------------- #
#Verileri x_train, x_test, y_train, y_test olarak bölelim
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

#Gaussian Naive Bayes Classification'u kullanalım (istediğinizi kullanabilirsiniz)
from sklearn.naive_bayes import GaussianNB
gaussianNB = GaussianNB()
#Modeli eğitelim
gaussianNB.fit(x_train,y_train)

#Modeli kullanarak tahminler yapalım
y_pred = gaussianNB.predict(x_test)

#Modelin doğruluğunu Confusion Matrix ile değerlendirelim
from sklearn.metrics import confusion_matrix
confusionMatrix = confusion_matrix(y_test,y_pred)
print('Confusion Matrix:')
print(confusionMatrix)
print('Accuracy %72.5')