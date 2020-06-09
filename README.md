# Uygulama Adımları
Gerekli tüm kütüphaneler Python'e eklenir. Sonrasında veri seti kütüphaneler yardımıyla eklenir.
Veri setindeki boş veri sayılarının tespiti yapılır. İsteğe göre boş veriler doldurulur ya da silinir. Bu işlemler sonrasında veri setindeki string değerler numeric değerlere çevrilir.
Girdi değerleri ve hedef çıktı belirlenir. Verileri train ve test olarak bölmek için train_test_split fonksiyonu kullanılır. Hem eğitim seti hem test seti StandardScaler() sınıfı kullanılarak standartlaştırılır.
İyi bir sonuç elde edene kadar Dense() sınıfı ile katman eklenir. Bu çalışmada 2 ara katman eklenmiştir. Birinci ara katman 9 nörondan oluşmuştur. İkinci ara katman 14 nörondan oluşmuştur.
Modelin derlenmesi için Adam optimizasyon algoritması kullanılmıştır. Bu fonksiyon birçok parametre alır fakat bu çalışmada optimizer, loss ve metrics olmak üzere kullanılmıştır. Optimizer, öğrenme oranını kontrol eder. Metrics, eğitim sırasında modelin nasıl bir performans gösterdiğini görmek için kullanılmıştır. Loss, modelin mevcut durumu için hatanın tekrar tekrar tahmin edilmesi gerekir. Tekrar edilen tahminler arasında kaybı azaltmak için bu fonksiyon kullanılır. Kayıp fonksiyonlar arasında bu model için en iyi sonucu veren mean_squared_logarithmic_error fonksiyonudur. Mean squared logarithmic error, tahmin edilen ve gerçek değerler arasındaki kare farklılıkların ortalaması olarak hesaplanır.
Sonra fit() fonksiyonunu çağırarak model yüklü veriler üzerinde eğitilmiştir. Bu fonksiyonda kullanılan parametreler epochs ve batch_size’dır. Epochs, eğitim veri kümesindeki tüm satırlardan geçiş sayısıdır. Örneğin epochs değeri 100 verildiğinde 100 kere tekrarlanacaktır. Batch_size, aynı anda kaç verinin işleneceğinin sayısını verir.


