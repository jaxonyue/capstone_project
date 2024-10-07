# Social Insider Prediction Problem

Capstone project delivered by Kelly Tong, Annie (Xueqing) Wu, Harry (Haochong) Xia, Jaxon Yue

#### Description
This Github repository serves as the primarily record of our capstone team project. The project cooperates with the Social Insider company to accomplish 2 main goals: 1. developing an algorithm to predict user behavior (subscribe/purchase to Social Insider services) base on a series of events conducted by the user; 2. predicting the follower count and average engagement rate of social media pages based on historical performance data. 

#### Overall Timeline

![timeline](https://github.com/user-attachments/assets/13acd1ba-d3a2-4f16-82e2-ceb16a748575)

#### Important Files Included: 
- Data & Files Folder:
    - socialinsider_events_2024.csv data files which are labeled with the associated month. Ex. socialinsider_events_2024-05.csv is the event data in May 2024. 
    - Social_Insider_Data_Pipeline.ipynb: a jupyter notebook that involves all processes for transforming original event-level data to user-level data, which serves as the data preprocessing stage before modeling.
- exploratory_analysis Folder:
    - Socialinsider Exploratory Data Analysis.pdf: a report that includes main details of our exploratory analysis outcomes. The report explains data insights with visualizations as well as pipeline transformation process. 
 
#### Example User Journey on a Social Insider Platform

![whiteboard_exported_image](https://github.com/user-attachments/assets/1e245042-408c-42d0-8607-1b02f862e7bb)
 
#### Exploratory Analysis Summary

`1. Data Overview: `

<img width="759" alt="截屏2024-10-08 04 22 17" src="https://github.com/user-attachments/assets/aaaa0f6a-8624-4ce8-89f9-59d87a85de50">

`2. Example Visualizations: `

More visualizations and exploratory analysis details can be fold in "Social Insider Exploratory Data Analysis" file. 

![visual1](https://github.com/user-attachments/assets/69135750-9b38-4841-940e-032b8df9da71)

![visual2](https://github.com/user-attachments/assets/7fca8f0f-d0b8-45cb-9975-f378c0673d82)

`3. Pipeline Transformation: `
A simplified diagram for explaining the pipeline transformation:

<img width="1161" alt="截屏2024-10-08 04 20 17" src="https://github.com/user-attachments/assets/a4c18974-8c1b-4da8-a0f5-a82b57340b1b">

Based on our EDA insights, we have done data cleaning and feature engineering, in which we have built a pipeline for transforming the original event-level data to user-level data. In the original data, each row is representing each event done by the user and its associated information such as load time, platform etc. In the transformed data, each row is representing behaviors and features related to every unique user. There are in total 45 features that have been transformed by the pipeline currently. These mainly include conversion, country, load time, count of specific events, count of each specific platform and count for each type of view.
