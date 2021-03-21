#!/usr/bin/env python
# coding: utf-8

# # EDA on IPL dataset (Task 5)

# In[1]:


#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, precision_recall_curve
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Loading dataset
data = pd.read_csv("matches.csv")
Data = pd.read_csv("deliveries.csv")
data.head(7)


# In[3]:


Data.head(7)


# In[4]:


each_season_data=data[['id','season','winner']]
merge_data=Data.merge(each_season_data,how='inner',left_on='match_id',right_on='id')


# In[5]:


merge_data.head(10)


# final_matches=data.drop_duplicates(subset=['season'], keep='last')
# final_matches[['season','winner']].reset_index(drop=True).sort_values('season')

# # Winners season-wise

# In[6]:


final_matches=data.drop_duplicates(subset=['season'], keep='last')
final_matches[['season','winner']].reset_index(drop=True).sort_values('season')


# # Total IPL Championships till 2019

# In[7]:


final_matches["winner"].value_counts()


# In[8]:


plt.figure(figsize = (20,10))
sns.countplot('season',data=data,palette="rocket")
plt.title("Number of Matches played each season",fontsize=20)
plt.xlabel("season",fontsize=15)
plt.ylabel('Number of Matches',fontsize=15)
plt.show()


# # Percentage Wins on batting/bowling first

# In[9]:


data['win_by']=np.where(data['win_by_runs']>0,'Team Batting first','Team Bowling first')
Win=data.win_by.value_counts()
labels=np.array(Win.index)
sizes = Win.values
colors = ['#00B000', '#1A8072']
sns.color_palette("crest", as_cmap=True)
plt.figure(figsize = (10,8))
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.11f%%', shadow=True,startangle=0)
plt.title('% Wins',fontsize=20)
plt.axis('equal')
plt.show()


# # Total matches won by teams batting/bowling first each season

# In[10]:


plt.figure(figsize = (20,10))
sns.countplot('season',hue='win_by', data=data, palette='crest')
plt.title("Numbers of matches won by batting and bowling first ",fontsize=20)
plt.xlabel("Season",fontsize=15)
plt.ylabel("Count",fontsize=15)
plt.show()


# # Total matches won by indivisual teams

# In[11]:


plt.figure(figsize = (20,10))
sns.countplot(x='winner',data=data, palette='viridis')
plt.title("Numbers of matches won by any team ",fontsize=20)
plt.xticks(rotation=50)
plt.xlabel("Teams",fontsize=15)
plt.ylabel("No of wins",fontsize=15)
plt.show()


# # Toss impact

# In[12]:


plt.figure(figsize = (20,10))
sns.countplot('season',hue='toss_decision',data=data,palette="cubehelix")
plt.title("Matches won by Toss result ",fontsize=20)
plt.xlabel("Season",fontsize=15)
plt.ylabel("Count",fontsize=15)
plt.show()


# In[13]:


Tossdecision=data.toss_decision.value_counts()
labels=np.array(Tossdecision.index)
length = Tossdecision.values
colors = ['#F70F00', '#A08072']
plt.figure(figsize = (10,8))
plt.pie(length, labels=labels, colors=colors,
        autopct='%1.11f%%', shadow=True,startangle=-90)
plt.title('Toss result',fontsize=20)
plt.axis('equal')
plt.show()


# # Toss impact in finals

# In[14]:


Toss=final_matches.toss_decision.value_counts()
labels=np.array(Toss.index)
sizes = Toss.values
colors = ['#CC7070', '#FFA482']
plt.figure(figsize = (10,8))
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True,startangle=90)
plt.title('Toss Result in finals',fontsize=20)
plt.axis('equal')
plt.show()


# # Man of the match in finals

# In[15]:


final_matches[['season','winner','player_of_match']].reset_index(drop=True).sort_values('season')


# In[ ]:





# # Total fours by each team

# In[16]:


fours=merge_data[merge_data['batsman_runs']==4]
fours.groupby('batting_team')['batsman_runs'].agg([('runs by fours','sum'),('fours','count')])


# In[17]:


ax=fours.groupby('season')['batsman_runs'].agg([('four','count')]).reset_index().plot('season','four',kind='bar',color = 'blue')
plt.title("Fours hit in each season ",fontsize=20)
plt.xticks(rotation=40)
plt.xlabel("season",fontsize=15)
plt.ylabel("Fours",fontsize=15)
plt.show()


# # Total sixes by each team

# In[18]:


six_data=merge_data[merge_data['batsman_runs']==6]
six_data.groupby('batting_team')['batsman_runs'].agg([('runs by six','sum'),('sixes','count')])


# In[19]:


batsman_six=six_data.groupby('batsman')['batsman_runs'].agg([('six','count')]).reset_index().sort_values('six',ascending=0)
ax=batsman_six.iloc[:10,:].plot('batsman','six',kind='bar',color='purple')
plt.title("Sixes hit by indivisual players ",fontsize=20)
plt.xticks(rotation=50)
plt.xlabel("Player name",fontsize=15)
plt.ylabel("No of six",fontsize=15)
plt.show()


# In[20]:


super_overs = data.groupby("season")["result"].value_counts()
super_overs


# # Superovers played in each season

# In[21]:


plt.figure(figsize = (18,10))
sns.countplot('season',hue='result',data=data,palette='afmhot')
plt.title("Results of matches in each season ",fontsize=20)
plt.xlabel("Season",fontsize=15)
plt.ylabel("Count",fontsize=15)
plt.show()


# # Man of the match winners

# In[22]:


top_players = data.player_of_match.value_counts()[:10]
fig, ax = plt.subplots()
ax.set_ylim([0,20])
ax.set_ylabel("Count")
ax.set_title("Top player of the match Winners")
top_players.plot.bar()
sns.barplot(x = top_players.index, y = top_players, orient='v', palette="magma");
plt.show()


# # Top 10 run scorers

# In[23]:


batsman_score=Data.groupby('batsman')['batsman_runs'].agg(['sum']).reset_index().sort_values('sum',ascending=False).reset_index(drop=True)
batsman_score=batsman_score.rename(columns={'sum':'batsman_runs'})
print("*** Top 10 Leading Run Scorer in IPL ***")
batsman_score.iloc[:10,:]


# # Top 10 wicket takers

# In[24]:


wicket_data=Data.dropna(subset=['dismissal_kind'])
wicket_data=wicket_data[~wicket_data['dismissal_kind'].isin(['run out','retired hurt','obstructing the field'])]
wicket_data.groupby('bowler')['dismissal_kind'].agg(['count']).reset_index().sort_values('count',ascending=False).reset_index(drop=True).iloc[:10,:]


# In[ ]:




