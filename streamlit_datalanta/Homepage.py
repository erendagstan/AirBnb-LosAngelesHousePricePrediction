import streamlit as st

# Homepage
st.set_page_config(
    page_title="AIR BNB: Ev Fiyat Tahmin Modeli",
    page_icon="👋",
    layout="wide"
)

st.write("# 🏠 :red[AIR BNB] - :blue[Los Angeles] : Ev Fiyat Tahmin Modeli 🏠")

st.sidebar.success("Yukarıdan bir seçenek seçin.")

column1, column2 = st.columns(2)
column1.markdown(
    "<h3 style='line-height: 1.6;'>Bu çalışma, Los Angeles'taki Airbnb evlerinin fiyatlarını tahmin etmek amacıyla geliştirilen bir ev fiyat tahmin modelini sunmaktadır. Modelin oluşturulmasında, Airbnb evlerinin çevresindeki çeşitli etkenlerin de dikkate alındığı bir yaklaşım benimsendi.</h3>",
    unsafe_allow_html=True)

column1.markdown(
    "<h3 style='line-height: 1.6;'>Firestation, police station, landmarks, schools, metro stations, arrests areas, coffees, hospitals gibi önemli konumların çevresindeki etkileşimler, ev fiyatlarını belirlemede kritik faktörler olarak ele alındı. Bu kapsamlı model, ev fiyatlarını tahmin "
    "etmek için çoklu değişkenlerin karmaşıklığını ve çeşitliliğini dikkate alarak, Los Angeles'taki Airbnb ev pazarındaki fiyat dalgalanmalarını daha iyi anlamamıza katkıda bulunmayı amaçlamaktadır.</h3>",
    unsafe_allow_html=True)

column1.markdown("<h3 style='line-height: 1.6;'>Analizde kullanılan veri seti, ev fiyatlarını etkileyen çeşitli "
                 "faktörleri içererek, kullanıcılara daha bilinçli kararlar vermeleri için değerli bir kaynak sunmaktadır.</h3>",
                 unsafe_allow_html=True)

column2.image("streamlit_datalanta/mike-von-iFcuaH0fkKU-unsplash.jpg", use_column_width=True)

st.header(":red[Los Angeles Maps]")
st.markdown(
    "<h3 style='line-height: 1.6;'>Los Angeles Maps sekmesi üzerinden, çalışmamızda değerlendirdiğimiz önemli lokasyonların, yani itfaiye istasyonları, polis karakolları, anıtlar, okullar, metro istasyonları, tutuklama alanları, kafeler ve hastanelerin harita üzerindeki dağılımları ve konumları görsel olarak sunulmuştur. Bu harita, Airbnb evlerinin çevresindeki çeşitli etkenlerin analizinde kullandığımız veri setini temsil eder. Her bir konumun belirli bir etkileşimi veya özelliği temsil ettiği bu harita, ev fiyatlarını tahmin etmek için oluşturulan modelimizin geliştirilmesinde kritik öneme sahip olan çevresel faktörleri vurgular. Harita üzerindeki dağılımları inceleyerek, belirli konumların ev fiyatlarını nasıl etkileyebileceği konusunda önemli bilgiler elde edebilir ve bu doğrultuda daha bilinçli kararlar verebilirsiniz.</h3>",
    unsafe_allow_html=True)

st.image("DatalantaProject/streamlit_datalanta/jose-martin-ramirez-carrasco-yhNVwsKTSaI-unsplash.jpg",
         use_column_width=True)

st.header(":red[Meet Los Angeles Landmarks]")

st.markdown(
    "<h3 style='line-height: 1.6;'> Meet Los Angeles Landmarks: Closest Airbnb Houses sekmesi, kullanıcıların ziyaret etmek istedikleri Los Angeles anıtlarının çevresindeki en yakın Airbnb evlerini keşfetmelerine olanak tanır. Bu sekme üzerinden, kullanıcılar belirli bir landmark'ın etrafındaki evleri görüntüleyebilir, fiyatlarını inceleyebilir ve Airbnb listeleme sayfasına erişebilirler. Harita üzerinde evlerin konumlarını görme imkanı, kullanıcılara çevresel faktörleri değerlendirerek konaklama seçeneklerini karşılaştırma ve tercih etme olanağı sunar. Bu sayede, ziyaret edilecek landmark'a en yakın konaklama seçeneklerini görsel bir şekilde keşfetmek ve uygun seçenekleri değerlendirmek mümkündür. ",
    unsafe_allow_html=True
)
st.image("DatalantaProject/streamlit_datalanta/de-andre-bush-eoohqHDVEP0-unsplash.jpg", use_column_width=True)

st.header(":red[House Price Prediction]")

st.markdown(
    "<h3 style='line-height: 1.6;'>House Price Prediction sekmesi, kullanıcılardan çeşitli girdiler alarak Airbnb ev fiyatlarını tahmin etme imkanı sunar. Kullanıcılar, tahminde bulunmak istedikleri evin özelliklerini belirlemek adına çeşitli girişleri yaparlar. Bu girişler arasında, evin türü (Property type), bulunduğu mahalle (Neighbourhood), Echo Park'a olan uzaklık, yatak odası sayısı, oda tipi (Room Type), banyo sayısı, olanaklar (amenities), konaklama kapasitesi (accommodates), yatak sayısı, güvenlik depozitosu (security deposit), dahil olan misafir sayısı gibi çeşitli faktörler yer alır. Bu bilgiler, modelimiz tarafından işlenerek, kullanıcının belirttiği ev özelliklerine göre tahmin edilen bir fiyat ortaya çıkarılır. House Price Prediction sekmesi, kullanıcılara belirli ev özellikleri üzerinden fiyat tahmini yapma ve konaklama seçeneklerini değerlendirme imkanı sunar.</h3>",
    unsafe_allow_html=True)

st.image("DatalantaProject/streamlit_datalanta/patrick-tomasso-14FtF6S_MpI-unsplash.jpg", use_column_width=True)
