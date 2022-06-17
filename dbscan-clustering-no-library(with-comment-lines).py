#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import random as rnd
import matplotlib.pyplot as plt

# load the dataset
# --------------------------------------------------------------------------------------------------
# veri setini yükle
hcv_data = pd.read_csv("hcvdat0.csv")

# read the data
# --------------------------------------------------------------------------------------------------
# veri setini oku
hcv_data.head()


# In[32]:


# see number of rows of the data
# --------------------------------------------------------------------------------------------------
# verinin satır sayısına bak
number_of_rows = len(hcv_data)
print("***********")
print("number_of_rows")
print(number_of_rows)
print("***********")


# In[33]:


# prepare dataframe
# --------------------------------------------------------------------------------------------------
# veri setini hazırla, ilgili olmayan sütunları kes

# use the rows you want with test purposes
# --------------------------------------------------------------------------------------------------
# test amaçlı, verinin istediğin satırlarını kullan
DATA_FRAME = hcv_data.iloc[0:615,:]
DATA_FRAME.drop(['Unnamed: 0','Category','Age','Sex'], inplace=True, axis=1)

# read the data
# --------------------------------------------------------------------------------------------------
# veri setini oku
DATA_FRAME.head()


# In[34]:


# normalize data
# --------------------------------------------------------------------------------------------------
# veri setini normalize et
DATA_FRAME=(DATA_FRAME-DATA_FRAME.mean())/DATA_FRAME.std()


# In[35]:


# read the data
# --------------------------------------------------------------------------------------------------
# veri setini oku
DATA_FRAME.head()


# In[36]:


# check for null or missing values
# --------------------------------------------------------------------------------------------------
# satır bazlı eksik veri sayısına bak
DATA_FRAME.isna().sum()


# In[37]:


# to insert the mean value of each column into its missing rows:
# --------------------------------------------------------------------------------------------------
# eksik verileri ortalamaya göre doldur
# DATA_FRAME.fillna(DATA_FRAME.mean().round(1), inplace=True)
y = DATA_FRAME.fillna(DATA_FRAME.mean().round(1))


# In[38]:


# check for null or missing values
# --------------------------------------------------------------------------------------------------
# eksik veriler doldurulmuş mu, kontrol et
DATA_FRAME.isna().sum()


# In[39]:


y.isna().sum()


# In[40]:


# dbscan function
# dbscan algoritmasını uygulayan fonksiyon

def dbscan(D, eps, MinPts):
    '''
    Cluster the dataset `D` using the DBSCAN algorithm.
    
    dbscan takes a dataset `D`, a threshold distance
    `eps`, and a required number of points `MinPts`.
    
    It will return a list of cluster cluster_labels. The label -1 means noise, and then
    the clusters are numbered starting from 1.
    --------------------------------------------------------------------------------------------------
    DBSCAN algoritmasını kullanarak `D` veri setini kümeleyeceğiz.
    
    dbscan fonksiyonu `D` veri setini,  `eps` sınır değerini (dairenin yarıçapı gibi  düşünebiliriz)
    ve asgari nokta sayısı olan `MinPts`(dairenin içinde kalan nokta sayısı) değerini alır.
    
    cluster_labels adlı, hangi noktanın hangi kümeye ait olduğunu gösterir bir liste döndürür.
    -1 gürültü değerleri için ayrılmıştır, küme değerleri 1'den başlar.
    '''
 
    # This list will hold the final cluster assignment for each point in D.
    # There are two reserved values:
    #    -1 - Indicates a noise point
    #     0 - Means the point hasn't been considered yet.
    # Initially all class_labels are 0.
    # --------------------------------------------------------------------------------------------------
    # Bu liste, algoritmanın çalışması sonucunda elde edilecek kümeleri verir.
    # Rezerve edilen 2 değer mevcuttur:
    #    -1 - Gürültü (herhangi bir sınıfa dahil olmayan değer)
    #     0 - Henüz işleme alınmamış nokta/değer
    # Başlangıçta bütün noktalar 0'dır, yani işleme henüz alınmamışlardır ve herhangi bir kümeye dahil değildir.
    
    data_len = len(D)
    cluster_labels = [0]*data_len

    # C is the ID of the current cluster.  
    # --------------------------------------------------------------------------------------------------
    # C değeri dahil olunan küme değerini gösterir.
    C = 0
    
    # This outer loop is just responsible for picking new seed points--a point from which to grow a new cluster.
    # Once a valid seed point is found, a new cluster is created, and the 
    # cluster growth is all handled by the 'grow_cluster' routine.
    # --------------------------------------------------------------------------------------------------
    # Bu dış döngü, yeni merkez noktalarını belirlemede kullanılıyor.
    # Geçerli bir merkez noktası bulunduğunda, yeni bir küme yaratılıyor ve 
    # kümenin genişletilmesi 'grow_cluster' metodu ile sağlanıyor.
    
    # For each point P in the Dataset D...
    # ('P' is the index of the datapoint, rather than the datapoint itself.)
    # --------------------------------------------------------------------------------------------------
    # D veri setindeki her bir noktayı ele alıyoruz(P)...
    # ('P' veri setindeki verilerin kendisi değil, indeksidir)
    for P in range(0, data_len):
    
        # Only points that have not already been claimed can be picked as new seed points.    
        # If the point's label is not 0, continue to the next point.
        # --------------------------------------------------------------------------------------------------
        # Daha önceden hiç değerlendirilmeyen/ele alınmayan noktalar merkez noktası olarak seçilebilir.
        # Eğer noktanın etiketi 0 değilse(daha önceden ele alınmışsa), diğer noktaya devam et.
        if not (cluster_labels[P] == 0):
           continue
        
        # Find all of P's neighboring points.
        # --------------------------------------------------------------------------------------------------
        # P noktasına ait tüm komşu noktaları bulalım.
        NeighborPts = find_neighbors(D, P, eps, cluster_labels)
        print("***************")
        print(f"index: {P}")
        print("NeighborPts per index")
        print(NeighborPts)
        print("***************")
        # If the number is below MinPts, this point is noise. 
        # This is the only condition under which a point is labeled NOISE--when it's not a valid seed point. 
        # A NOISE point may later be picked up by another cluster as a boundary point (this is the only
        # condition under which a cluster label can change--from NOISE to something else).
        # --------------------------------------------------------------------------------------------------
        # Eğer NeighborPts değeri MinPts değerinden küçükse, o nokta gürültü olarak etiketlenir.
        # Merkez nokta olma koşullarını sağlamayan bir noktanın gürültü olarak etiketlendiği tek durum budur. 
        # Gürültü olarak etiketlenen bir nokta, daha sonradan bir merkez noktasına ait sınır noktası olabilir. 
        # Algoritmamızda gürültü olarak işaretlenen bir değerin, başka bir değere değişebileceği tek durum budur.
        if len(NeighborPts) < MinPts:
            cluster_labels[P] = -1
            
        # Otherwise, if there are at least MinPts nearby, use this point as the seed for a new cluster.
        # --------------------------------------------------------------------------------------------------
        # Diğer durumda, yani NeighborPts değeri MinPts değerine eşit veya ondan büyükse, bu nokta merkez noktası seçilir. 
        else: 
           C += 1
           print("**** C (Cluster) Label ******")
           print(C)
           print("***************")
           grow_cluster(D, cluster_labels, P, NeighborPts, C, eps, MinPts)
            
    
    # convert class_labels array to a dataframe
    # --------------------------------------------------------------------------------------------------
    # class_labels array'ini dataframe'e dönüştür
    clusters_data_frame = pd.DataFrame(cluster_labels, columns=['Number of cluster members'])
    number_of_clusters = clusters_data_frame.groupby('Number of cluster members').size()
    print("-----------------")
    print("Number of clusters")
    print(number_of_clusters)
    print("-----------------")
    
    # plot number of classes with histogram
    # --------------------------------------------------------------------------------------------------
    # sınıf değerlerini histogram ile gösterelim.
    plt.style.use('ggplot')
    plt.xlim(-2, len(number_of_clusters))
    labels, counts = np.unique(clusters_data_frame['Number of cluster members'], return_counts=True)
    plt.bar(labels, counts)
    plt.xticks(number_of_clusters.index)
    plt.show()
    
    # All data has been clustered.
    # --------------------------------------------------------------------------------------------------
    # Verinin tamamı kümelenmiş oldu.
    

    
    return cluster_labels


def grow_cluster(D, cluster_labels, P, NeighborPts, C, eps, MinPts):
    '''
    Grow a new cluster with label `C` from the seed point `P`.
    
    This function searches through the dataset to find all points that belong
    to this new cluster. When this function returns, cluster `C` is complete.
    
    Parameters:
      `D`      - The dataset (a list of vectors)
      `class_labels` - List storing the cluster labels for all dataset points
      `P`      - Index of the seed point for this new cluster
      `NeighborPts` - All of the neighbors of `P`
      `C`      - The label for this new cluster.  
      `eps`    - Threshold distance
      `MinPts` - Minimum required number of neighbors
      --------------------------------------------------------------------------------------------------
    `P`noktasını kullanarak `C` etiketli kümeyi genişlet .
    
    Bu fonksiyon tüm veri seti boyunca yeni oluşturulan kümeye ait diğer elemanları bulmaya çalışır. 
    Bu fonksiyon return ettiğinde,`C` kümesi tamamlanmış demektir.
    
    Parametreler:
      `D`      - Veri seti
      `cluster_labels` - Veri setindeki tüm noktalara ait küme etiketlerini tutan liste
      `P`      - Yeni kümeye ait merkez noktasının indeksi
      `NeighborPts` - `P` noktasının tüm komşuları
      `C`      - Kümeye ait etiket  
      `eps`    - Kümenin eşik değeri
      `MinPts` - Minimum gerekli komşu nokta sayısı
      
    '''

    # Assign the cluster label to the seed point.
    # --------------------------------------------------------------------------------------------------
    # Küme etiketini merkez noktasına ata.
    cluster_labels[P] = C
    
    # Look at each neighbor of P (neighbors are referred to as Pn). 
    # NeighborPts will be used as a FIFO queue of points to search--that is,
    # it will grow as we discover new branch points for the cluster. 
    # The FIFO behavior is accomplished by using a while-loop rather than a for-loop.
    # In NeighborPts, the points are represented by their index in the original dataset.
    # --------------------------------------------------------------------------------------------------
    # P'nin her bir komşusuna bak (komşu noktalar Pn olarak ele alındı). 
    # NeighborPts araştırılacak noktalar için bir FIFO sırası olarak kullanılacak--böylece,
    # kümeye ait yeni dallanma noktaları tespit edildiğinde küme büyüyecek. 
    # FIFO işlevi for-loop yerine while-loop ile yerine getirilmektedir.
    # NeighborPts'deki değerler veri setindeki noktaların index değerini içermektedir.    
    
    i = 0
    while i < len(NeighborPts):    
        
        # Get the next point from the queue.
        # --------------------------------------------------------------------------------------------------
        # Sıradaki diğer noktaya geç
        Pn = NeighborPts[i]
       
        # If Pn was labelled NOISE during the seed search, then we know it's not a branch point 
        # (it doesn't have enough neighbors), so make it a leaf point of cluster C and move on.
        # --------------------------------------------------------------------------------------------------
        # Eğer Pn noktası merkez araştırması sırasında gürültü olarak etiketlenirse,
        # o artık bir dallanma noktası olamaz(yeterli komşusu yoktur)
        # böylece bu nokta kümeye ait bir sınır değeri (leaf/boundry) olur ve sonraki değerden devam edilir.
        if cluster_labels[Pn] == -1:
            cluster_labels[Pn] = C
        
        # Otherwise, if Pn isn't already claimed, claim it as part of C.
        # --------------------------------------------------------------------------------------------------
        # Diğer durumda, Pn henüz ele alınmamışsa, bu noktayı da kümeye dahil edelim ve buradan dallanma olacak mı bakalım.
        elif cluster_labels[Pn] == 0:
            cluster_labels[Pn] = C
            
            # Find all the neighbors of Pn
            # --------------------------------------------------------------------------------------------------
            # Pn'ye ait tüm komşu noktaları bulalım
            PnNeighborPts = find_neighbors(D, Pn, eps, cluster_labels)
            
            # If Pn has at least MinPts neighbors, it's a branch point!
            # Add all of its neighbors to the FIFO queue to be searched.
            # --------------------------------------------------------------------------------------------------
            # Eğer Pn en az MinPts kadar komşuya sahipse, o bir dallanma noktasıdır.
            # Onun tüm komşularını da FIFO sırasına ekleyip devam edelim.
            if len(PnNeighborPts) >= MinPts:
                NeighborPts = np.unique(np.concatenate((NeighborPts,PnNeighborPts),0))
                for i in NeighborPts:
                    #print("****")
                    #print("new neighbors")
                    #print(i)
                    cluster_labels[i] = C
                    #print("****")
                
            # If Pn *doesn't* have enough neighbors, then it's a leaf point.
            # Don't queue up it's neighbors as expansion points.
            #else:
                # Do nothing                
                #NeighborPts = NeighborPts 
            # Eğer Pn yeteri kadar komşuya sahip değilse, bu durumda bir sınır değeri (leaf/boundry) olur.
            # Onun komşularını sıraya alıp araştırmamıza gerek kalmadı.
            # else:
                # Birşey yapmaya gerek yok                
                # NeighborPts = NeighborPts 
        
        # Advance to the next point in the FIFO queue.
        # --------------------------------------------------------------------------------------------------
        # FIFO sırasındaki diğer noktaya geçelim.
        i += 1        
    
    # We've finished growing cluster C.
    # --------------------------------------------------------------------------------------------------
    # C kümesini genişletmeyi tamamladık.


def find_neighbors(D, P, eps, cluster_labels):
    '''
    Find all points in dataset `D` within distance `eps` of point `P`.
    
    This function calculates the distance between a point P and every other 
    point in the dataset, and then returns only those points which are within a
    threshold distance `eps`.
    (We use euclidean distance in this application.)
    # --------------------------------------------------------------------------------------------------
    `D` veri setinde `P` noktasına `eps` kadar mesafede bulunan tüm noktaları bul.
    
    Bu fonksiyon P noktası ile veri setindeki diğer tüm noktalar arasındaki mesafeyi hesaplar,
    ve sadece eşik değeri olan `eps` sınırları içerisindeki döndürür. 
    (Biz bu uygulamada euclidean distance kullanıyoruz.)
    
    '''
    neighbors = []
    
    
    for Pn in range(0, len(D)):
            
        # If the datapoint is not included in another class AND 
        # if the distance is below the threshold, add it to the neighbors list.
        # --------------------------------------------------------------------------------------------------
        # Eğer veri noktası başka bir sınıfa dahil değilse VE
        # uzaklık eşik değeri altındaysa onu neighbors listesine ekle.
        
        # When np.linalg.norm() is called on an array-like input without any additional arguments,
        # the default behavior is to compute the L2 norm on a flattened view of the array.
        # This is the square root of the sum of squared elements and  
        # can be interpreted as the length of the vector in Euclidean space.
        # Shortly, we use Euclidean distance.
        # --------------------------------------------------------------------------------------------------
        # np.linalg.norm() ekstra bir argüman olmadan sadece bir array girdisiyle kullanıldığında,
        # varsayılan olarak o array'in flatten edilmiş hali üzerinde L2 normalizasyonu uygular.
        # Bu ise array elemanlarının karelerinin toplamının kareköküne eşittir ve 
        # Euclidean uzayında vektör uzunluğu olarak yorumlanabilir.
        # Kısaca Euclidean distance kullanıyoruz.
        
        not_clustered = (cluster_labels[Pn] == 0) or (cluster_labels[Pn] == -1)
        smaller_than_eps = np.sum(np.linalg.norm(D.iloc[P] - D.iloc[Pn])) < eps
        
        if not_clustered and smaller_than_eps:
           neighbors.append(Pn)
            
    return neighbors


# In[41]:


# HOW TO SELECT EPS

# Once you have the appropriate minPts, in order to determine the optimal eps, follow these steps:
# Let's say minPts = 20
# 1. For every point in dataset, compute the distance of it's 20th nearest neighbor.
# (generally we use euclidean distance, but you can experiment with different distance metrics).
# 2. Sort the distances in the increasing order.
# 3. Plot the chart of distances on Y-axis v/s the index of the datapoints on X-axis.
# 4. Observe the sudden increase or what we popularly call as an 'elbow' or 'knee' in the plot. 
# Select the distance value that corresponds to the 'elbow' as optimal eps.
# --------------------------------------------------------------------------------------------------

# EPS'İ NASIL SEÇEBİLİRİZ?

# Uygun minPts değerine sahip olduğumuzda,optimal eps değerini belirlemek için şu adımları takip edebiliriz:
# minPts = 20 olsun
# 1. Veri setindeki her bir nokta için, o noktaya ait 20'nci yakınlıktaki komşusunu hesapla.
# (genelde euclidean distance kullanılır, farklı uzaklık metrikleri de denenebilir.).
# 2. Uzaklıkları artan sırada sırala.
# 3. Y eksenindeki uzaklıklar ile X eksenindeki veri noktalarının indekslerini çizime dök.
# 4. Çizimdeki ani artış noktasını ya da popüler isimleriyle 'dirsek' ya da 'diz' noktasını bul.
# Bu noktadaki değeri optimal EPS değeri olarak al.


# In[42]:


def calculate_eps(D, MinPts):
    
    # 1 ------------------------------------------
    minPts_distances = []
    data_length = len(D)

    for i in range(0, data_length):
        distances = []
        for j in range (0, data_length):
            distances.append(np.linalg.norm(D.iloc[i] - D.iloc[j]))
        sorted_distances = np.sort(distances)
        minPts_distances.append(sorted_distances[MinPts-1])
                
    
    # 2 ------------------------------------------
    y_axis = np.sort(minPts_distances)
    print("--- y_axis ---")
    print(y_axis)
    
    # 3 ------------------------------------------
    x_axis = np.arange(0, data_length)
    print("--- x_axis ---")
    print(x_axis)
    
    
    plt.plot(x_axis, y_axis)
    plt.ylabel("Distances")
    plt.xlabel("Indexes")
    plt.show()
    
    return minPts_distances
    


# In[43]:


calculate_eps(DATA_FRAME, 20)


# In[862]:


dbscan(DATA_FRAME, 2.5, 20)


# In[ ]:




