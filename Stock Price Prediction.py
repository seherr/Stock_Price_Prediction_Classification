#!/usr/bin/env python
# coding: utf-8

# In[107]:


import yfinance as yf


# In[108]:


df = yf.Ticker("^GSPC")


# In[109]:


# max vererek baştan sonra tüm verileri almak için ""
df = df.history(period="max")


# In[110]:


df


# In[111]:


df.plot.line(y="Close", use_index=True)  # x is date and y is close price 


# In[112]:


# dividends = temettü 
# stock split = hisse bölünmeleri 
# bunlar genel için değil de bireysel hisseler için daha uygun bu sebeple bizim işimize yaramıyor.

del df["Dividends"]
del df["Stock Splits"]


# In[113]:


df


# In[114]:


# for making clustering we need a binary target. But we dont have so that ; 
#I will create a target about tomorrow the price increased is represented by 1, decreased is 0.


# In[115]:


df["Tomorrow"] = df["Close"].shift(-1)


# In[116]:


df
# 3 ocak tarihinin kapanışını aldık yarın olarak 4 ocak tarihinin kapanışını tomorrow sutununda tuttuk. 
# Artık tahminleyeceğim şey yarının fiyatının bugünün fiyatından büyük olmasıdır. 


# In[117]:


df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)


# In[118]:


df


# In[119]:


# borsada geçmiş veriler çok yanıltıcı olabilir, değişkendir çünkü. 

df = df.loc["1990-01-01":].copy()


# In[120]:


df


# In[121]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 100, min_samples_split = 100, random_state = 1 )

# zaman series dataset
train = df.iloc[:-100]   # son 100 satır dışındakileri al bu demektir ki geçmişi kullanarak geleceği tahmin et. Çünkü zaman sıralı şekilde verilmiş. 
test = df.iloc[-100:] 


# In[122]:


predictors = ["Close","Volume","Open","High","Low"]

model.fit(train[predictors], train["Target"])


# In[123]:


# doğruluğunu ölçelim. 

from sklearn.metrics import precision_score

preds = model.predict(test[predictors])


# In[124]:


import pandas as pd 

preds = pd.Series(preds, index=test.index)


# In[125]:


preds


# In[126]:


precision_score(test["Target"], preds)


# In[127]:


# tahmin değerleri ile grçek değerleri karışılaştıralım turuncu = pred, blue = real
# son zamanlarda değişim büyük olduğu için hata fazla görünüyor.
combined = pd.concat([test["Target"], preds], axis=1)

combined.plot()


# In[128]:


# backtest 
#Backtest, bir stratejinin geçmiş verilere dayanarak nasıl performans gösterdiğini anlamak için güçlü bir araçtır. 
# Ancak, dikkatlice kullanılmalıdır çünkü geçmiş başarı, gelecekteki başarıyı garantilemez ve stratejinin aşırı uyarlamaya (overfitting) veya geçmişteki koşullara özel olmasına neden olabilir.

def predict(train, test, predictors, model): 
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


# In[129]:


# her ticaret yılında yaklaşım 250 gün vardır. Yani 10 yıllık veriyle modeli eğit diyoruz. 
#ilk 10 yıllık veriyle 11. yılın verilerini tahminnedeceiz.Sonra 11 yılı da alıp 12. yılı tahminleyecek. 

def backtest(data, model, predictors, start=2500, step=250):
    
    all_predictions = []  # yılların tahminleri sonra bu yıllar içinde gezeceğiz 
    
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)


# In[130]:


predictions = backtest(df, model, predictors)


# In[131]:


predictions["Predictions"].value_counts()


# In[132]:


precision_score(predictions["Target"], predictions["Predictions"])


# In[133]:


predictions["Target"].value_counts() / predictions.shape[0]


# In[134]:


#.53 ü yükselirken 0.46sı yükselmiyor. 


# In[135]:


# son 2 ortalama kapanış fiyatını hesaplayacağız. Son işlem haftasının 5 günü, son 2 ay (60) , son bir yıl, son 4 yılın 60 işlemi

# yükseliş zamanı

horizons = [2,5,60,250,1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = df.rolling(horizon).mean()
    # kapanış fiyatı hareketli ortalama ile elde edilecek işte bugün ile son 5 yüzde kaç değişmiş. 
    ratio_column = f"Close_Ration_{horizon}"
    df[ratio_column] = df["Close"] / rolling_averages["Close"]
    
    trend_column = f"Trend_{horizon}"
    df[trend_column] = df.shift(1).rolling(horizon).sum()["Target"]
    
    new_predictors += [ratio_column, trend_column]


# In[136]:


df


# In[137]:


df = df.dropna()


# In[138]:


df


# In[139]:


model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)


# In[140]:


get_ipython().system('pip install --upgrade scikit-learn')


# In[141]:


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >=.6] = 1
    preds[preds <.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


# In[142]:


predictions = backtest(df, model, new_predictors)


# In[143]:


predictions["Predictions"].value_counts()


# In[144]:


# 78359 gün boyunca fiyat düşecek, 31679 gün boyunca artacak.


# In[145]:


precision_score(predictions["Target"], predictions["Predictions"])


# In[ ]:




