import numpy as np
import pandas as pd
import keras
import matplotlib as plt
import warnings
warnings.filterwarnings("ignore") #suppress warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
from matplotlib import pyplot

data_1=pd.read_csv(r"C:\Users\hatic\Desktop\International students Time management data.csv", encoding='utf-8')

data = pd.read_csv(r"C:\Users\hatic\Desktop\International students Time management data.csv", encoding='utf-8') #veri seti import edildi.
data.isnull().sum()  #veri setimdeki kolonlardaki boş veri sayısı
data=data.fillna(method = 'backfill') #boş veriler dolduruldu.Tekrardan data.isnull().sum() çalıştırıldı.
data = data.drop(['Number','13','15'], axis=1)  #axis=1 dediğimizde sütundan siliyoruz, 0 dersek satırdan siliyoruz.

#Kolon başlıkları listeye kaydedildi, for yapısı kullanarak her bir kolon numeric değere döndürüldü.
column_header=['Age','Gender','Nationality','Program','Course','English','Academic','Attendance','6','7','8','9','10','11','12','14','16','17']
for i in column_header:
    labelencoder_X = LabelEncoder()
    data[i]=labelencoder_X.fit_transform(data[i])
    
girdi=data.drop(columns=['7'])  #hedef çıktı drop ederek kalanları girdi olarak alıyorum.
hedef_cikti=data['7'] #veri setinden hedef çıktıyı seçtik.

Xtrain, Xtest, ytrain, ytest = train_test_split(girdi, hedef_cikti, test_size=0.2, random_state=25)

stdsc = StandardScaler()
Xtrain = stdsc.fit_transform(Xtrain)#eğitim setinin standartlıştırılması
Xtest = stdsc.transform(Xtest)#test setinin standartlaştırılması

print(f"Train setinin girdisi : {Xtrain.shape}")
print(f"Train setinin çıktısı : {ytrain.shape}")
print(f"Test setinin girdisi : {Xtest.shape}")
print(f"Test setinin çıktısı : {ytest.shape}")

# modelimize katmanlar ekliyoruz. 
model = Sequential()
model.add(Dense(9, input_dim=17, activation='sigmoid'))
model.add(Dense(14, activation='sigmoid'))
#model.add(Dense(15, activation='sigmoid'))

model.add(Dense(1, activation='sigmoid'))

opt = Adam(learning_rate=0.04)
model.compile(optimizer=opt, loss='mean_squared_logarithmic_error', metrics=['accuracy'])

a=model.fit(Xtrain, ytrain, epochs=100,validation_data=(Xtest, ytest), verbose=0, batch_size=17)
train_acc = model.evaluate(Xtrain, ytrain, verbose=0)[1]
test_acc = model.evaluate(Xtest, ytest, verbose=0)[1]


print("Yapay sinir ağının train oranı: {}".format(round((train_acc * 100), 5)))
print("Yapay sinir ağının test oranı: {}".format(round((test_acc * 100),5)))


#eğitilmiş olan modelin eğitim boyunca başarım oranını gösteren başarı grafiğini
plt.subplot(211)
plt.title('Loss')
plt.plot(a.history['loss'], label='train')
plt.plot(a.history['val_loss'], label='test')
plt.legend()

plt.subplot(212)
plt.title('Accuracy')
plt.plot(a.history['accuracy'], label='train')
plt.plot(a.history['val_accuracy'], label='test')
plt.legend()
plt.show()


