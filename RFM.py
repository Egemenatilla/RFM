# Importing related libraries
import numpy as np
import pandas as pd 
import time, warnings
import datetime as dt
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
import pickle
warnings.filterwarnings("ignore")

#%% Reading the data from the file and making a copy so that the raw data is not corrupted
filename= pd.read_csv('data.csv',encoding="ISO-8859-1",dtype={'CustomerID': str,'InvoiceID': str})
df=filename.copy()
print(df)
print(df.info())

#%% Restriction of data to UK. England was chosen because there was too much data from England.
data_uk= df[df['Country']=='United Kingdom']

print(data_uk.shape)
print(data_uk.tail())

#%% Dropping NaN data
print(data_uk.dropna(axis=0,inplace=True))

# If the amount received is 0 or less than 0, the purchase is not considered to be made. Therefore, rows with positive amount were taken.
data_uk = data_uk[data_uk['Quantity']>0]

print(data_uk.shape)

# Determining the date when data starts and ends recording.
print("Orders from {} to {}".format(df["InvoiceDate"].min(),df["InvoiceDate"].max())) #ocakla eylül arası

# Converting the CustomerID column to int
data_uk["CustomerID"]=data_uk["CustomerID"].astype("int64")

# Setting the day to a date for later actions
now = dt.date(2011,12,9)

# Converting InvoiceDate column to dateTime
data_uk['date'] = pd.DatetimeIndex(data_uk['InvoiceDate']).date

print(data_uk)

# Determining how many times each user makes a purchase
df_freq=data_uk[["CustomerID","InvoiceNo"]].groupby("CustomerID").count().sort_values(by="InvoiceNo",ascending=False)
df_freq.columns=["cust_invoice_freq"]
print(df_freq)
print(df_freq.head())

#%%
print(data_uk[data_uk.CustomerID==12346])
#%% A new column named Price has been created and the price information of the purchases made here is kept.

data_uk['price'] = data_uk['Quantity'] * data_uk['UnitPrice']

data_uk.head()

# Determining RFM -> Recency,Frequency,Monetary Value values.
# Recency-> How many days ago did the customer last shop?
# Frequency -> How often did the customer shop during this period?
# Monetary -> How much money did the customer spend during this period?

rfm=data_uk.groupby("CustomerID").agg({
    "date":lambda date:(now-date.max()).days,
    "InvoiceNo":lambda x:x.count(),
    "price": lambda price: price.sum()
})

rfm.columns= ["Recency","Frequency","Monetary"]
rfm=rfm[(rfm["Monetary"])>0 & (rfm["Frequency"]>0)]
print(rfm)

# RFM scoring
# R,F,M Scoring the values ​​from 1-5.

rfm["RecencyScore"]=pd.qcut(rfm["Recency"],5, labels=[5,4,3,2,1],duplicates="drop")
rfm["FrequencyScore"]=pd.qcut(rfm["Frequency"].rank(method="first"),5,labels=[5,4,3,2,1],duplicates="drop")
rfm["MonetaryScore"]=pd.qcut(rfm["Monetary"],5, labels=[5,4,3,2,1],duplicates="drop")

rfm["RFM_Score"]=(rfm["RecencyScore"].astype(str) + rfm["FrequencyScore"].astype(str) + rfm["MonetaryScore"].astype(str))
print(rfm.head(15))

#%%
# RFM analysis
# Making sense of RFM values - Segmentation
seg_map = {
    r'[1-2][1-2]': 'uykuda',
    r'[1-2][3-4]': 'riskli',
    r'[1-2]5': 'kaybedilemez',
    r'3[1-2]': 'pasif olmak üzere',
    r'33': 'ilgilenilmeli',
    r'[3-4][4-5]': 'sadık musteri',
    r'41': 'umut verici',
    r'51': 'yeni müsteri',
    r'[4-5][2-3]': 'potansiyel sadık musteri',
    r'5[4-5]': 'şampiyon'
}
#%%
rfm["Segment"]=rfm["RecencyScore"].astype(str)+rfm["FrequencyScore"].astype(str)
rfm["Segment"]=rfm["Segment"].replace(seg_map,regex=True)
df[["CustomerID"]].nunique()
rfm[["Segment","Recency","Frequency","Monetary"]].groupby("Segment").agg(["mean"],["count"])
#%%
print(rfm[rfm["Segment"]=="sadık musteri"].head())
print(rfm[rfm["Segment"]=="kaybedilemez"].head())
print(rfm[rfm["Segment"]=="uykuda"].head())
print(rfm[rfm["Segment"]=="riskli"].head())

#%%
print(rfm["Segment"].value_counts())
print(rfm[["Segment","Recency","Frequency","Monetary"]].groupby("Segment").agg(["mean","count"]))

#%% Visualization
segments_counts = rfm['Segment'].value_counts().sort_values(ascending=True)

fig, ax = plt.subplots()

bars = ax.barh(range(len(segments_counts)),
              segments_counts,
              color='silver')
ax.set_frame_on(False)
ax.tick_params(left=False,
               bottom=False,
               labelbottom=False)
ax.set_yticks(range(len(segments_counts)))
ax.set_yticklabels(segments_counts.index)

for i, bar in enumerate(bars):
        value = bar.get_width()
        if segments_counts.index[i] in ['riskli', 'şampiyon']:
            bar.set_color('firebrick')
        ax.text(value,
                bar.get_y() + bar.get_height()/2,
                '{:,} ({:}%)'.format(int(value),
                                   int(value*100/segments_counts.sum())),
                va='center',
                ha='left'
               )
plt.show()