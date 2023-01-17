#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


books = pd.read_csv("C:/Users/raamn/Desktop/BX-Books.csv",sep = ';',error_bad_lines = False,encoding = 'latin')


# In[3]:


books.info()


# In[4]:


books.head()


# In[5]:


books.columns


# In[6]:


books = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']]
books.head()


# In[7]:


books.rename(columns = {'Book-Title':'title','Book-Author':'author','Year-Of-Publication':'year','Publisher':'publisher'}, inplace = True)
books.head()


# In[8]:


users = pd.read_csv("C:/Users/raamn/Desktop/BX-Users.csv",sep = ';',error_bad_lines = False,encoding = 'latin-1')


# In[9]:


users.head()


# In[10]:


users.rename(columns = {'User-ID':'user_id','Location':'location','Age':'age'},inplace = True)
users.head()


# In[11]:


ratings = pd.read_csv("C:/Users/raamn/Desktop/BX-Book-Ratings.csv",sep =';',error_bad_lines = False, encoding = 'latin-1')
ratings.head()


# In[12]:


ratings.rename(columns = {'User-ID':'user_id','Book-Rating':'rating'},inplace = True)
ratings.head(2)


# In[13]:


books.shape


# In[14]:


users.shape


# In[15]:


ratings.shape


# In[18]:


x = ratings['user_id'].value_counts() > 200
x.shape


# In[19]:


y = x[x].index
print(y.shape)
y


# In[20]:


ratings = ratings[ratings['user_id'].isin(y)]
ratings.shape


# In[21]:


books_with_ratings = ratings.merge(books,on = 'ISBN')
books_with_ratings


# In[22]:


number_rating = books_with_ratings.groupby('title')['rating'].count().reset_index()
number_rating


# In[23]:


number_rating.rename(columns={'rating':'number_of_ratings'},inplace = True)


# In[24]:


final_rating = books_with_ratings.merge(number_rating,on='title')
final_rating


# In[25]:


final_rating = final_rating[final_rating['number_of_ratings']>=50]
final_rating.shape


# In[26]:


final_rating.drop_duplicates(['user_id','title'],inplace = True)


# In[27]:


final_rating.shape


# In[39]:


book_pivot = final_rating.pivot_table(columns = 'user_id',index ='title',values = 'rating')


# In[52]:


book_pivot.fillna(0,inplace = True)
book_pivot


# In[53]:


from scipy.sparse import csr_matrix


# In[54]:


book_sparse = csr_matrix(book_pivot)


# In[55]:


type(book_sparse)


# In[58]:


from sklearn.neighbors import NearestNeighbors
model = NearestNeighbors(algorithm = 'brute')


# In[59]:


model.fit(book_sparse)


# In[64]:


distances, suggestions = model.kneighbors(book_pivot.iloc[237,:].values.reshape(1,-1),n_neighbors = 6)


# In[65]:


distances


# In[66]:


suggestions


# In[69]:


for i in range(len(suggestions)):
    print(book_pivot.index[suggestions[i]])


# In[70]:


book_pivot.index[240]


# In[71]:


np.where(book_pivot.index == 'Harry Potter and the Prisoner of Azkaban (Book 3)')


# In[72]:


np.where(book_pivot.index == 'Harry Potter and the Prisoner of Azkaban (Book 3)')[0][0]


# In[75]:


def recommend_book(book_name):
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distances, suggestions = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1),n_neighbors = 6)
    
    for i in range(len(suggestions)):
        if i == 0:
            print('The suggestions for the book',book_name,'are:')
        if not i:
            print(book_pivot.index[suggestions[i]])
            


# In[76]:


recommend_book('The Cradle Will Fall')

