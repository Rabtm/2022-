## 카페 소재지 면적 결측치 제거 
```
import pandas as pd

DF = pd.read_excel('/content/진짜 최종 0816.xlsx')

df = DF.copy()

df.head()

![1](https://user-images.githubusercontent.com/103248870/188040052-b9fb2bbd-c2d6-47ca-bc17-b11294dadbd2.png)

df_drop_row = df.dropna(axis=0)

df_drop_row1= df_drop_row.iloc[:, [1,9,10,12,13,14,15]]

df_drop_row1
```

![KakaoTalk_20220902_103552112](https://user-images.githubusercontent.com/103248870/188040387-2eccd4a5-adab-4c5b-9805-452cb1356349.png)

## 리뷰 크롤링 코드 
```
from selenium import webdriver

from selenium.webdriver.common.keys import Keys

import selenium

import time

from selenium.webdriver.common.by import By

import pandas as pd

df = pd.read_excel('D:\\2022년\\데이터청년캠퍼스\\프로젝트\\데이터\\finaldataset.xlsx')

df.head()

![3](https://user-images.githubusercontent.com/103248870/188040565-ba324fe2-8531-474a-9143-cbbe4edf513a.png)

df['cafe_name'][0]

name = []

for i in range(17726):
    name.append(df['cafe_name'][i])
 ```   
name

![5](https://user-images.githubusercontent.com/103248870/188040773-6bfef844-c593-414a-8e69-a05917756913.png)

### 여러개 같이 나왔을 때
```
driver = webdriver.Chrome('C:\\Users\\koko1\\Downloads\\chromedriver_win32\\chromedriver.exe')

driver.get("https://www.naver.com/")

elem = driver.find_element(By.NAME,"query")

elem.send_keys("스타벅스 성신여대점")

elem.send_keys(Keys.RETURN)

time.sleep(5)

fir = driver.find_elements(By.CSS_SELECTOR,'#place-main-section-root > section > div > ul > li:nth-child(1) > div._3hn9q > a > div > div > span.place_bluelink.OXiLu')

for title in fir:
    fir_1 = title.text
    
sec = driver.find_elements(By.CSS_SELECTOR,'#place-main-section-root > section > div > ul > li:nth-child(2) > div._3hn9q > a > div > div > span.place_bluelink.OXiLu')
```
    for t in sec:
    sec_2 = t.text

    if fir_1 == '스타벅스 성신여대점':
        
    #find_text = driver.find_element("xpath", '//*[@id="place-main-section-root"]/section/div/ul/li[1]/div[1]/div[3]/span[3]/text()[2]')
    #print(find_text)
    
    temp = driver.find_elements(By.CSS_SELECTOR,'#place-main-section-root > section > div > ul > li:nth-child(1) > div._3hn9q > div._17H46')
    
    for title in temp:
        
        print(title.text)

    elif sec_2 == '스타벅스 성신여대점':
    
    #find_text = driver.find_element("xpath", '//*[@id="place-main-section-root"]/section/div/ul/li[2]/div[1]/div[3]/span[3]/text()[2]')
    #print(find_text)
    
    temp = driver.find_elements(By.CSS_SELECTOR,'#place-main-section-root > section > div > ul > li:nth-child(2) > div._3hn9q > div._17H46')
    for title in temp:
        print(title.text)

    else:
        print('error!')

![11](https://user-images.githubusercontent.com/103248870/188041154-041f598d-06e1-4868-8a08-3959cd698c07.png)

### 합치는과정
```
driver = webdriver.Chrome('C:\\Users\\koko1\\Downloads\\chromedriver_win32\\chromedriver.exe')

driver.get("https://www.naver.com/")

rlst = []

elem = driver.find_element(By.ID,"query")
```
```
for i in name:
    elem.send_keys(i)
    elem.send_keys(Keys.RETURN)
    time.sleep(5)
    if driver.find_elements(By.CSS_SELECTOR,"#loc-main-section-root > section > div > div.api_title_area > h2") or driver.find_elements(By.CSS_SELECTOR,"#place-main-section-root > section > div > div.api_title_area > h2"):
        fir = driver.find_elements(By.CSS_SELECTOR,'#place-main-section-root > section > div > ul > li:nth-child(1) > div._3hn9q > a > div > div > span.place_bluelink.OXiLu')
        for title in fir:
            fir_1 = title.text
  ```          
            
        sec = driver.find_elements(By.CSS_SELECTOR,'#place-main-section-root > section > div > ul > li:nth-child(2) > div._3hn9q > a > div > div > span.place_bluelink.OXiLu')
        for t in sec:
            sec_2 = t.text
            

        if fir_1 == i:
            temp = driver.find_elements(By.CSS_SELECTOR,'#place-main-section-root > section > div > ul > li:nth-child(1) > div._3hn9q > div._17H46')
            for title in temp:
                rlst.append([i, title.text])
                
                
        elif sec_2 == i:
            temp = driver.find_elements(By.CSS_SELECTOR,'#place-main-section-root > section > div > ul > li:nth-child(2) > div._3hn9q > div._17H46')
            for title in temp:
                rlst.append([i, title.text])
                
                
        else:
            rlst.append([i, 'error!'])
        
    else:
        temp = driver.find_elements(By.CLASS_NAME,"place_bluelink")
        
        for title in temp[:2]:
            rlst.append([i, title.text])
        
    driver.find_element(By.ID,"nx_query").clear()
    
    elem = driver.find_element(By.ID, 'nx_query')
    driver.close()
rlst

![545](https://user-images.githubusercontent.com/103248870/188041312-dca8c6d8-55bd-4950-bbce-ebf15d6f0346.png)



good = pd.DataFrame(rlst) #rlst를 데이터프레임으로 만드는 과정 


# K_Means clustering
```
#필요한 모듈
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.datasets import make_blobs

from sklearn.decomposition import PCA

import sklearn

%matplotlib inline
```

### 데이터셋 불러오기 
```
df = pd.read_excel('C:\\Users\\osh27\\Desktop\\동대문구.xlsx')

df1 = df.iloc[:, [14,15]]
```
## Standard Scaler
```
df = pd.read_excel('C:\\Users\\osh27\\Desktop\\동대문구.xlsx')

df1 = df.iloc[:, [14,15]]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()   

scaled = scaler.fit_transform(df1)

df = pd.DataFrame(scaled)

df.columns=df1.columns
```
```
wcss = []

for i in range(1,11):
    model = KMeans(n_clusters = i, init = "k-means++")
    model.fit(df)
    wcss.append(model.inertia_)

plt.figure(figsize=(10,10))

plt.plot(range(1,11), wcss, marker='o')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
```
![22](https://user-images.githubusercontent.com/103248870/188054067-a824ce67-c659-429d-b064-d28904a3fe78.png)
```
#차원축소

pca = PCA(2)

data = pca.fit_transform(df)
```
```
centers = np.array(model.cluster_centers_)

model = KMeans(n_clusters =3 , init = "k-means++")

label = model.fit_predict(data)

plt.figure(figsize=(7,7))

uniq = np.unique(label)

for i in uniq:
    plt.scatter(data[label == i , 0] , data[label == i , 1] , label = i)

plt.scatter(centers[:,0], centers[:,1], marker="x", color='k')

plt.legend()

plt.show()
```
![23](https://user-images.githubusercontent.com/103248870/188054129-a1a1e700-971e-4118-98ac-302f55fe3971.png)

model.inertia_

#88.67799706001539

## Robust Scaler 
```
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

scaled = scaler.fit_transform(df1)
```
![robust](https://user-images.githubusercontent.com/103248870/188054223-cfd88fd0-455c-434c-bd75-64f31cdfa0b0.png)

![robust2](https://user-images.githubusercontent.com/103248870/188054266-48c8e954-2ad8-43eb-b37f-c3527774c227.png)


## MaxAbsScaler
```
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()

scaled = scaler.fit_transform(df1)
```
![maxAbs](https://user-images.githubusercontent.com/103248870/188054377-c88cbd4d-890e-4baf-b61c-a79313339b99.png)

![maxabs2](https://user-images.githubusercontent.com/103248870/188054386-b012811e-5829-4e9f-9ee1-ab84dd98fdaf.png)


## MinMax Scaler
```
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaled = scaler.fit_transform(df1)

df = pd.DataFrame(scaled)

df.columns=df1.columns
```
### 군집 개수 결정

## Elbow method
```
inertia_arr=[]

k_range = range(2,15)

for k in k_range:
    Kmeans=KMeans(n_clusters=k,random_state=200)
    Kmeans.fit(df)
    inertia = Kmeans.inertia_
    print('k: ',k,'inertia:',inertia)
    inertia_arr.append(inertia)

inertia_arr=np.array(inertia_arr)

plt.plot(k_range,inertia_arr)

plt.title('Elbow Method(inertia)')

plt.xlabel('Number of clusters')

plt.ylabel('Inertia')

plt.show()
```
![minmax1](https://user-images.githubusercontent.com/103248870/188054559-7b227ee2-e0b5-48ef-bfc7-45d03597444c.png)

## Elbow method2
```
wcss = []
for i in range(1,11):
    model = KMeans(n_clusters = i, init = "k-means++")
    model.fit(df)
    wcss.append(model.inertia_)

plt.figure(figsize=(10,10))

plt.plot(range(1,11), wcss, marker='o')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
```
![minmax6](https://user-images.githubusercontent.com/103248870/188054681-9aa51841-090d-4d44-a90e-b52c8a9547ce.png)

### Silhouette score
```
from sklearn.metrics import silhouette_score

k_range=range(2,15)

best_n=-1

best_silhouette_score = -1

silhouette_score_arr = []

for k in k_range:
    kmeans=KMeans(n_clusters=k, random_state=200)
    kmeans.fit(df)
    clusters=kmeans.predict(df)
    score=silhouette_score(df,clusters)
    print('k:',k,'score: ',score)
    silhouette_score_arr.append(score)

    if score> best_silhouette_score:
        best_n=k
        best_silhouette_score=score
        

print('best_n:',best_n,'best score:',best_silhouette_score)

silhouette_score_arr=np.array(silhouette_score_arr)

plt.plot(k_range,silhouette_score_arr)

plt.title('silhouette_score')

plt.xlabel('Number of clusters')

plt.ylabel('silhouette_score')

plt.show()
```
![KakaoTalk_20220902_124152588](https://user-images.githubusercontent.com/103248870/188054750-2818d0a9-7ce2-405c-bb32-3728c8736a2c.png)

### 군집화 결과
```
centers = np.array(model.cluster_centers_)

model = KMeans(n_clusters = 3, init = "k-means++")

label = model.fit_predict(data)

plt.figure(figsize=(10,10))

uniq = np.unique(label)

for i in uniq:
    plt.scatter(data[label == i , 0] , data[label == i , 1] , label = i)

plt.scatter(centers[:,0], centers[:,1], marker="x", color='k')

plt.legend()

plt.show()
```
![KakaoTalk_20220902_124227039](https://user-images.githubusercontent.com/103248870/188054800-8f46ab8b-3a9f-4e5e-a718-2a760a895abe.png)

model.inertia_

#4.655291222796844
```
cluster = pd.Series(label)

df1['cluster'] = cluster

df1['cluster'].value_counts()

df2 = pd.DataFrame(columns = df1.columns[:-1])

c1=[]

c2=[]

for i in range(4):    
    c1.append(df1[df1['cluster']==i]['x_coord'].mean())
    c2.append(df1[df1['cluster']==i]['y_coord'].mean())
  
df2['x_coord']=c1

df2['y_coord']=c2

data = pd.read_excel('C:\\Users\\osh27\\Desktop\\동대문구.xlsx')

df1['gid'] = data['gid']

### 클러스터에 해당하는 그리드 출력
gid0 = []

gid1 = []

gid2 = []

data = pd.read_excel('C:\\Users\\osh27\\Desktop\\동대문구.xlsx')

for i in range(len(df1)):
    if df1.iloc[i]['cluster']==0:
        gid0.append(df1.iloc[i]['gid'])
    elif df1.iloc[i]['cluster']==1:
        gid1.append(df1.iloc[i]['gid'])
    else:
        gid2.append(df1.iloc[i]['gid'])
 ```
 # K_Medoids clustering
```
!pip install scikit-learn-extra

import numpy as np

import matplotlib.pyplot as plt

from sklearn_extra.cluster import KMedoids

from sklearn.datasets import make_blobs

import pandas as pd

%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

import sklearn

#위 방식과 동일 
```
![1](https://user-images.githubusercontent.com/103248870/188055000-a15cf744-4909-4973-9f48-562cc65bf868.png)

![2](https://user-images.githubusercontent.com/103248870/188055005-fb8a3b66-1c7f-4253-8144-d14e15c84df6.png)

![3](https://user-images.githubusercontent.com/103248870/188055014-bad50716-20c8-48cf-bfda-aa06a572c0ca.png)


# Hierarchical_Agglomerative
```
#data

import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings("ignore") 

#visualization

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

#preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

#model

from sklearn.cluster import KMeans

from sklearn.cluster import DBSCAN

from scipy.cluster.hierarchy import dendrogram, ward

from sklearn.cluster import AgglomerativeClustering

from sklearn.cluster import AffinityPropagation

from sklearn.cluster import MeanShift, estimate_bandwidth

#grid search

from sklearn.model_selection import GridSearchCV

#evaluation

from sklearn.metrics.cluster import silhouette_score

from sklearn.model_selection import cross_val_score

from sklearn import metrics

from sklearn.metrics import *
```
```
plt.figure(figsize=(15,8))

linkage_array = ward(reduced_df)

dendrogram(linkage_array)

plt.xlabel("Sample Num")

plt.ylabel("Cluster Dist")

#클러스터를 구분하는 커트라인을 표시

ax = plt.gca()

bounds = ax.get_xbound()

ax.plot(bounds, [350, 350], '--', c='k')

ax.plot(bounds, [200, 200], '--', c='k')

ax.plot(bounds, [100, 100], '--', c='k')

ax.text(bounds[1], 700, ' 3 Clusters ', va='center', fontdict={'size': 10})

ax.text(bounds[1], 690, ' 4 Clusters ', va='center', fontdict={'size': 10})

ax.text(bounds[1], 680, ' 5 Clusters ', va='center', fontdict={'size': 10})

ax.text(bounds[1], 670, ' 6 Clusters ', va='center', fontdict={'size': 10})

ax.text(bounds[1], 660, ' 7 Clusters ', va='center', fontdict={'size': 10})

ax.text(bounds[1], 650, ' 8 Clusters ', va='center', fontdict={'size': 10})

ax.text(bounds[1], 640, ' 9 Clusters ', va='center', fontdict={'size': 10})

ax.text(bounds[1], 630, ' 10 Clusters ', va='center', fontdict={'size': 10})

ax.text(bounds[1], 620, ' 11 Clusters ', va='center', fontdict={'size': 10})

ax.text(bounds[1], 610, ' 12 Clusters ', va='center', fontdict={'size': 10})

ax.text(bounds[1], 600, ' 13 Clusters ', va='center', fontdict={'size': 10})

ax.text(bounds[1], 590, ' 14 Clusters ', va='center', fontdict={'size': 10})

ax.text(bounds[1], 580, ' 15 Clusters ', va='center', fontdict={'size': 10})

plt.show
```
![h4](https://user-images.githubusercontent.com/103248870/188058520-23bf0e3e-5c63-49ff-af42-186e381a936c.png)

```
n = [3,4,5,6,7,8,9,10,11,12,13,14,15]

for i in n:
    plt.figure(figsize=(15,8))
    agg = AgglomerativeClustering(n_clusters=i)
    cluster = agg.fit(reduced_df)
    # cluster = agg.fit(r_scaled_df)
    cluster_id = pd.DataFrame(cluster.labels_)
    
    d4 = pd.DataFrame()
    d4 = pd.concat([reduced_df,cluster_id],axis=1)
    d4.columns = [0, 1, "cluster"]
    
    sns.scatterplot(d4[0], d4[1], hue = d4['cluster'], legend="full")
    plt.title('Agglomerative with {} clusters'.format(i))
    plt.show()
    
    print('Silhouette Coefficient: {:.4f}'.format(metrics.silhouette_score(d4.iloc[:,:-1], d4['cluster'])))
    print('Davies Bouldin Index: {:.4f}'.format(metrics.davies_bouldin_score(d4.iloc[:,:-1], d4['cluster'])))
```
![h1](https://user-images.githubusercontent.com/103248870/188058153-d2b3e57b-71b9-4e36-a773-fd20096015b0.png)

```
#n = [2,3,5,7,9]

#for i in n:

plt.figure(figsize=(15,8))

agg = AgglomerativeClustering(n_clusters=7)

cluster = agg.fit(reduced_df)

cluster_id = pd.DataFrame(cluster.labels_)

d4 = pd.DataFrame()

d4 = pd.concat([reduced_df,cluster_id],axis=1)

d4.columns = [0, 1, "cluster"]

sns.scatterplot(d4[0], d4[1], hue = d4['cluster'], legend="full")

plt.title('Agglomerative with {} clusters'.format(5))

plt.show()

print('Silhouette Coefficient: {:.4f}'.format(metrics.silhouette_score(d4.iloc[:,:-1], d4['cluster'])))

print('Davies Bouldin Index: {:.4f}'.format(metrics.davies_bouldin_score(d4.iloc[:,:-1], d4['cluster'])))
```
![h3](https://user-images.githubusercontent.com/103248870/188058558-f669659a-f937-44e2-8dae-a854321d78e9.png)


# 커피박수거함 위치 뽑아내는 과정
```
import numpy as np

import math

import pandas as pd

import os

import time

from tqdm import tqdm

import math
```
```
#점과 점 사이의 거리 구하는 함수

def distance(x1, y1, x2, y2):
    result = math.sqrt( math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
    return result
    
PATH = os.getcwd()
```
all_cafe = pd.read_excel('C:\\Users\\osh27\\Desktop\\군집화 전 진짜 최종 0822.xlsx')

all_cafe.head()

![KakaoTalk_20220902_105205499](https://user-images.githubusercontent.com/103248870/188041966-ea7dbfc8-ec18-4d5c-b879-4916977848af.png)
```
#해당 구 뽑아오기

gu_cafe = all_cafe[all_cafe['layer'] == '마포구']

#구 군집화한 것들 들고오기

gu_grid = pd.read_excel('C:\\Users\\osh27\\Desktop\\마포구 그리드.xlsx')

gu_grid['마포구'].unique()#해당 구에 군집이 몇개 인지 확인

#해당 구 카페 그리드와 카페를 군집한 그리드 합치기

rlst= pd.merge(gu_cafe,gu_grid, how= 'left', on='gid')

#해당 구 클러스터 설정

gid =rlst[rlst['마포구'] == 9]

gid.reset_index(drop=True, inplace=True)

cafe_x = []

cafe_y = []

for i in range(len(gid)):
    cafe_x.append(gid.iloc[i][13])
    cafe_y.append(gid.iloc[i][14])
    
dis = []

for i in range(len(cafe_x)):
    for j in range(len(cafe_x)):
        dis.append(distance(cafe_x[i], cafe_y[i], cafe_x[j], cafe_y[j]))
        
num = round(max(dis) / 0.007882882) #해당 군집을 재군집 할 때 군집 개수

#재군집

new_gid = gid.iloc[:, [13, 14]]

#정규화

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaled = scaler.fit_transform(new_gid)

minmax_gid = pd.DataFrame(scaled)

minmax_gid.columns=new_gid.columns
```
```
#필요한 모듈

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.datasets import make_blobs

from sklearn.decomposition import PCA

import sklearn

%matplotlib inline
```

#Elbow method2
```
wcss = []

for i in range(1,11):
    model = KMeans(n_clusters = i, init = "k-means++")
    model.fit(minmax_gid)
    wcss.append(model.inertia_)

plt.figure(figsize=(10,10))

plt.plot(range(1,11), wcss, marker='o')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
```
![elbow method](https://user-images.githubusercontent.com/103248870/188045728-9f7adc14-3ad6-41a3-bf34-c5da49a95997.png)
```
from sklearn.decomposition import PCA

#차원축소

pca = PCA(2)

data = pca.fit_transform(minmax_gid)
```
```
#군집화 결과

centers = np.array(model.cluster_centers_)

model = KMeans(n_clusters = num, init = "k-means++")

label = model.fit_predict(data)

plt.figure(figsize=(10,10))

uniq = np.unique(label)

for i in uniq:
    plt.scatter(data[label == i , 0] , data[label == i , 1] , label = i)

plt.scatter(centers[:,0], centers[:,1], marker="x", color='k')#각 클러스터 중심점 찾기 위한 과정

plt.legend()

plt.show()
```
![캡처](https://user-images.githubusercontent.com/103248870/188077987-7085ee67-1c7d-44bc-a140-2b0bb7b605ec.PNG)


model.inertia_

#1.1163378343099866
```
cluster = pd.Series(label)

minmax_gid = minmax_gid.reset_index()

minmax_gid = minmax_gid.drop(['index'], axis=1)

new_cluster = pd.concat([minmax_gid, cluster], axis=1)

new_cluster.columns = ['x_coord', 'y_coord', 'cluster']

new_cluster['gid'] = gid['gid']

new_cluster['pop'] = gid['pop']

new_cluster['living_pop'] = gid['living_pop']

new_cluster['floating_pop'] = gid['floating_pop']

new_cluster['area_sum'] = gid['area_sum']

new_cluster['review_visitor'] = gid['review_visitor']

new_cluster['blog_visitor'] = gid['blog_visitor']

new_cluster['avg_sal_sum'] = gid['avg_sal_sum']

new_cluster['x_coord_1'] = gid['x_coord']

new_cluster['y_coord_1'] = gid['y_coord']

new_cluster['num_point'] = gid['num_point']

new_cluster.head()

#새로운 클러스터의 클러스터 불러오기

test = new_cluster[new_cluster['cluster'] == 1]

test.reset_index(drop=True, inplace=True)


#그리드 별 minmax 해서 가중치 계산하기

test_w = test.iloc[:, [4, 5, 6, 7, 8, 9, 10]]


#스케일링 할 때 필요한 모듈

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaled = scaler.fit_transform(test_w)

weight = pd.DataFrame(scaled)

weight.columns=test_w.columns

weight.head()

weight['all_sum'] = weight.sum(axis=1)

weight.head()

rlst = test.iloc[:, [3, 2,  11, 12, 13]]
```
```
#군집된 카페 그리드에서 가중치 합치기

rlst = pd.concat([rlst,weight['all_sum']],axis=1)

rlst.head()

rlst['weight'] = rlst[['num_point', 'all_sum']].sum(axis=1)

rlst.head()   # 최종 데이터 프레임
```
![가중치 최종 데이터프레임](https://user-images.githubusercontent.com/103248870/188046102-421f9fb4-9530-4d15-96f5-36bc8fe23436.png)
```
candi_grid = pd.read_excel('C:\\Users\osh27\\Desktop\\마포구 전체 포인트.xlsx')

candi_grid = candi_grid[candi_grid['NUMPOINTS'] == 0]

candi_grid = candi_grid.iloc[:, [0, 16, 17]]

candi_grid = candi_grid.reset_index()

candi_grid = candi_grid.drop(['index'], axis=1)

candi_grid.head()

candidate_x = []

candidate_y = []

candidate_i = []

for i in range(len(candi_grid)):
    candidate_x.append(candi_grid.loc[i][1])
    candidate_y.append(candi_grid.loc[i][2])
    candidate_i.append(candi_grid.loc[i][0])
    cafe_x = []

cafe_y = []

cafe_w = []

for i in range(len(rlst)):
    cafe_x.append(rlst.iloc[i][2])
    cafe_y.append(rlst.iloc[i][3])
    cafe_w.append(rlst.iloc[i][6])
    dtc = sum([distance(x, y, candidate_x[0], candidate_y[0])*w for x, y, w in zip(cafe_x, cafe_y, cafe_w)])

for i in range(len(candidate_x)):
    dist = sum([distance(x, y, candidate_x[i], candidate_y[i])*w for x, y, w in zip(cafe_x, cafe_y, cafe_w)])
    if dist < dtc:
        dtc = dist
        idx = i
```

dtc

![1](https://user-images.githubusercontent.com/103248870/188070128-8553c53b-d0bb-4514-a629-6cae0157dc88.png)

candidate_i[idx]

![2](https://user-images.githubusercontent.com/103248870/188070139-bc98ba98-d16a-4b82-8e2d-30c1f4b0a498.png)

# RandomForest Feature Importance

### 필요한 모듈
```
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
```
```
data = pd.read_excel('/content/군집화 전 진짜 최종 0822.xlsx')#데이터 불러오기
data = data.dropna(axis=0)
data = data.iloc[:, [1,7,8,9,10,11,12,13]]
```
```
def getDf(self,data):
  X_data = data.drop([data.columns[6],data.columns[7]],axis=1)
  y_data = data.drop([data.columns[0],data.columns[1],data.columns[2],data.columns[3],data.columns[4],data.columns[5],data.columns[7]],axis=1)
  #정규화
  ss = StandardScaler()
  X_scaled = ss.fit_transform(X_data)
  y_scaled = ss.fit_transform(y_data)
  x_train, x_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3,shuffle=True)
  #학습 진행
  forest = RandomForestRegressor(n_estimators=100)
  forest.fit(x_train, y_train)

  return list(forest.feature_importances_)
  getDf(data,data)
  ```
<img width="806" alt="변수중요도" src="https://user-images.githubusercontent.com/103248870/188073449-24d3baf5-a47e-4660-8d4d-6f8f6544516b.png">

# 상관분석
```
library(ggplot2)

library(reshape2)

setwd('C:/Users/lee/Documents')

fi <- read.csv("0831minmax.csv", fileEncoding="UTF-8-BOM")

str(fi)

cormat <- round(cor(fi),2)

cormat

melted_cormat <- melt(cormat)

melted_cormat

get_lower_tri<-function(cormat){
  
  cormat[upper.tri(cormat)] <- NA
 
 return(cormat)
}

get_upper_tri <- function(cormat){
  
  cormat[lower.tri(cormat)]<- NA
  
  return(cormat)
}

upper_tri <- get_upper_tri(cormat)

upper_tri

melted_cormat <- melt(upper_tri, na.rm = TRUE)


ggheatmap <- ggplot(melted_cormat, aes(Var2, Var1, fill = value))+ geom_tile(color = "white")+
  
scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       
                       name="Pearson\nCorrelation") +
  
  theme_minimal()+ # minimal theme
  
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                  
                                  size = 12, hjust = 1))+
  
  coord_fixed()
```
#Print the heatmap
```
print(ggheatmap)

ggheatmap + 
  
  geom_text(aes(Var2, Var1, label = value), color = "black", size = 4) +
  
  theme(
   
    axis.title.x = element_blank(),
    
    axis.title.y = element_blank(),
    
    panel.grid.major = element_blank(),
    
    panel.border = element_blank(),
    
    panel.background = element_blank(),
    
    axis.ticks = element_blank(),
    
    legend.justification = c(1, 0),
    
    legend.position = c(0.6, 0.7),
    
    legend.direction = "horizontal")+
  
    guides(fill = guide_colorbar(barwidth = 7, barheight = 1,
                               
                               title.position = "top", title.hjust = 0.5))

```

<img width="276" alt="KakaoTalk_20220902_154028913" src="https://user-images.githubusercontent.com/103248870/188075481-54df0de1-ed3e-4e82-8031-34d1450ec232.png">

# 분산분석
```
options(scipen=999)

options(digits=3)

setwd('C:/Users/lee/Documents')

fi <- read.csv("0831minmax.csv", fileEncoding="UTF-8-BOM")

str(fi)

pop <- aov(avg_sal_sum ~ pop, data =fi)

living_pop <- aov(avg_sal_sum ~ living_pop, data =fi)

floating_pop <- aov(avg_sal_sum ~ floating_pop, data =fi)

area_sum <- aov(avg_sal_sum ~ area_sum, data =fi)

review_visitor_sum <- aov(avg_sal_sum ~ review_visitor, data =fi)

review_blog_sum <- aov(avg_sal_sum ~ blog_visitor, data =fi)

summary(pop)

summary(living_pop)

summary(floating_pop)

summary(area_sum)

summary(review_visitor_sum)

summary(review_blog_sum)
```
![KakaoTalk_20220902_154139442](https://user-images.githubusercontent.com/103248870/188075703-58b3b9df-480a-4db5-8657-b1a6a03afff2.png)

