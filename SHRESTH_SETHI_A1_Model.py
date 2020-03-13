#!/usr/bin/env python
# coding: utf-8

# In[7]:


#%%timeit

# Student Name : Shresth Sethi
# Cohort       : FMSBA5 - Valencia

#


################################################################################
# Import Packages
################################################################################



# Importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import sklearn.linear_model
from ipywidgets import interactive
from IPython.display import display, HTML
import statsmodels.formula.api as smf



################################################################################
# Load Data
################################################################################



# Loading DataSet Apprentice Chef
original_df = pd.read_excel("Apprentice_Chef_Dataset.xlsx")
original_description = pd.read_excel('Apprentice_Chef_Data_Dictionary.xlsx')



################################################################################
# Feature Engineering
################################################################################



#Creating an empty list to store the values
empty_lst = []


## This loop is created to split the email id and domain id
# looping over each email address using iterrows
for index, col in original_df.iterrows():
    
    #separating email using @ symbol
    split_email = original_df.loc[index, 'EMAIL'].split(sep = '@') 
    
    # appending the split_email over empty list to store values
    empty_lst.append(split_email)
    


# Converting empty_lst into a DataFrame 
email_df = pd.DataFrame(empty_lst)



# Adding the domain column in df
email_df.columns = ['ID', 'DOMAIN']



# Concatenating email_df "DOMAIN" column with original_df
original_df = pd.concat([original_df, email_df.loc[:,'DOMAIN']], axis = 1)



## Catagorizing the domains into professional, personal, and junk
# Creating list for professional email domain as specified in case
professional_email = ['@mmm.com', '@amex.com', '@apple.com','@boeing.com',
                      '@caterpillar.com', '@chevron.com', '@cisco.com',
                      '@cocacola.com','@disney.com', '@dupont.com',
                      '@exxon.com', '@ge.org','@goldmansacs.com', 
                      '@homedepot.com', '@ibm.com', '@intel.com',
                      '@jnj.com', '@jpmorgan.com', '@mcdonalds.com',
                      '@merck.com', '@microsoft.com', '@nike.com', 
                      '@pfizer.com', '@pg.com', '@travelers.com',
                      '@unitedtech.com', '@unitedhealth.com', 
                      '@verizon.com', '@visa.com', '@walmart.com']



# Creating list for personal email domain as specified in case
personal_email     = ['@gmail.com', '@yahoo.com', '@protonmail.com']



# Creating list for junk email domain as specified in case
junk_email         = ['@me.com', '@aol.com', '@hotmail.com', '@live.com',
                      '@msn.com', '@passport.com']



## This section is used to assign the catogries made with the domain
# Creating an empty list for the loop to store value
empty_lst = []



# Assigning domains to get the categorical data
for i in original_df['DOMAIN']:
    
        if   '@'+ i in professional_email:
             empty_lst.append('professional')
            
        elif '@' + i in personal_email:
             empty_lst.append('personal')

        elif '@' + i in junk_email:
             empty_lst.append('junk')
            
        else:
             print('Unknown')

                

# Creating a new column in the original dataset to store the assigned values
original_df['DOMAIN_GRP'] = pd.DataFrame(empty_lst)



# Using get_dummies to encode the "DOMAIN_GRP" column
domain_one_hot = pd.get_dummies(original_df['DOMAIN_GRP'])



# Dropping the column "DOMAIN_GRP" we don't need it
original_df          = original_df.drop('DOMAIN_GRP', axis = 1)



# Joining one hot encoding with the df
original_df          = original_df.join([domain_one_hot])



# Setting threshold values for outlier analysis
MEDIAN_MEAL_RATING_at1           = 4
AVG_CLICKS_PER_VISIT_hi          = 19
AVG_CLICKS_PER_VISIT_lo          = 8
LARGEST_ORDER_SIZE_hi            = 10
LARGEST_ORDER_SIZE_lo            = 1
AVG_TIME_PER_SITE_VISIT_hi       = 200
CONTACTS_W_CUSTOMER_SERVICE_hi   = 9
CONTACTS_W_CUSTOMER_SERVICE_lo   = 3
UNIQUE_MEALS_PURCH_hi            = 8
MOBILE_NUMBER_at                 = 1



# Selecting Median Meal Rating outliers
original_df['MEDIAN_MEAL_RATING_out_1'] = 0
condition_at_1 = original_df.loc[0:,'MEDIAN_MEAL_RATING_out_1'][original_df['MEDIAN_MEAL_RATING'] == MEDIAN_MEAL_RATING_at1]
original_df['MEDIAN_MEAL_RATING_out_1'].replace(to_replace = condition_at_1,
                                                value      = 1,
                                                inplace    = True)



# Selecting Avg Clicks Per Visits outliers
original_df['AVG_CLICKS_PER_VISIT_out'] = 0
condition_hi = original_df.loc[0:,'AVG_CLICKS_PER_VISIT_out'][original_df['AVG_CLICKS_PER_VISIT'] > AVG_CLICKS_PER_VISIT_hi]
condition_lo = original_df.loc[0:,'AVG_CLICKS_PER_VISIT_out'][original_df['AVG_CLICKS_PER_VISIT'] < AVG_CLICKS_PER_VISIT_lo]

original_df['AVG_CLICKS_PER_VISIT_out'].replace(to_replace = condition_hi,
                                                value      = 1,
                                                inplace    = True)
original_df['AVG_CLICKS_PER_VISIT_out'].replace(to_replace = condition_lo,
                                                value      = 1,
                                                inplace    = True)



# Selecting Largest Order Size outliers
original_df['LARGEST_ORDER_SIZE_out'] = 0
condition_hi = original_df.loc[0:,'LARGEST_ORDER_SIZE_out'][original_df['LARGEST_ORDER_SIZE'] > LARGEST_ORDER_SIZE_hi]
condition_lo = original_df.loc[0:,'LARGEST_ORDER_SIZE_out'][original_df['LARGEST_ORDER_SIZE'] < LARGEST_ORDER_SIZE_lo]

original_df['LARGEST_ORDER_SIZE_out'].replace(to_replace = condition_hi,
                                              value      = 1,
                                              inplace    = True)
original_df['LARGEST_ORDER_SIZE_out'].replace(to_replace = condition_lo,
                                              value      = 1,
                                              inplace    = True)



# Selecting Avg Time Per Visits outliers
original_df['AVG_TIME_PER_SITE_VISIT_out'] = 0
condition_hi = original_df.loc[0:,'AVG_TIME_PER_SITE_VISIT_out'][original_df['AVG_TIME_PER_SITE_VISIT'] > AVG_TIME_PER_SITE_VISIT_hi]
original_df['AVG_TIME_PER_SITE_VISIT_out'].replace(to_replace = condition_hi,
                                                   value      = 1,
                                                   inplace    = True)



# Selecting Contacts With Customer Service outliers
original_df['CONTACTS_W_CUSTOMER_SERVICE_out'] = 0
condition_hi = original_df.loc[0:,'CONTACTS_W_CUSTOMER_SERVICE_out'][original_df['CONTACTS_W_CUSTOMER_SERVICE'] > CONTACTS_W_CUSTOMER_SERVICE_hi]
condition_lo = original_df.loc[0:,'CONTACTS_W_CUSTOMER_SERVICE_out'][original_df['CONTACTS_W_CUSTOMER_SERVICE'] < CONTACTS_W_CUSTOMER_SERVICE_lo]

original_df['CONTACTS_W_CUSTOMER_SERVICE_out'].replace(to_replace = condition_hi,
                                                       value      = 1,
                                                       inplace    = True)
original_df['CONTACTS_W_CUSTOMER_SERVICE_out'].replace(to_replace = condition_lo,
                                                       value      = 1,
                                                       inplace    = True)



# Selecting Unique Meals Purchased outliers
original_df['UNIQUE_MEALS_PURCH_out'] = 0
condition_hi = original_df.loc[0:,'UNIQUE_MEALS_PURCH_out'][original_df['UNIQUE_MEALS_PURCH'] > UNIQUE_MEALS_PURCH_hi]
original_df['UNIQUE_MEALS_PURCH_out'].replace(to_replace = condition_hi,
                                              value      = 1,
                                              inplace    = True)



# Selecting Mobile Number at 1 people giving mobile number and excluding people with landline
original_df['MOBILE_NUMBER_at'] = 0
condition_at = original_df.loc[0:,'MOBILE_NUMBER_at'][original_df['MOBILE_NUMBER'] == MOBILE_NUMBER_at]
original_df['MOBILE_NUMBER_at'].replace(to_replace = condition_at,
                                        value      = 1,
                                        inplace    = True)



# Calculating fields with AVG_PREP_VID_TIME
original_df['avg_vid_time_rating']       = original_df['AVG_PREP_VID_TIME']*original_df['MEDIAN_MEAL_RATING']
original_df['avg_vid_time_click']        = original_df['AVG_PREP_VID_TIME']/original_df['AVG_CLICKS_PER_VISIT']
original_df['avg_prep_total_photo']      = original_df['AVG_PREP_VID_TIME']*original_df['TOTAL_PHOTOS_VIEWED']
original_df['avg_vid_time_larg_order']   = original_df['AVG_PREP_VID_TIME']*original_df['LARGEST_ORDER_SIZE']
original_df['avg_vid_customer_service']  = original_df['AVG_PREP_VID_TIME']*original_df['CONTACTS_W_CUSTOMER_SERVICE']



# Calculating fields with TOTAL_MEALS_ORDERED
original_df['meals_ordered_rating']      = original_df['TOTAL_MEALS_ORDERED']*original_df['MEDIAN_MEAL_RATING']
original_df['meals_ordered_click']       = original_df['TOTAL_MEALS_ORDERED']/original_df['AVG_CLICKS_PER_VISIT']
original_df['meals_ordered_photo']       = original_df['TOTAL_MEALS_ORDERED']*original_df['TOTAL_PHOTOS_VIEWED']
original_df['Frequency_orders']          = 1/original_df['TOTAL_MEALS_ORDERED']



# Median rating given by a customer over the average clicks
original_df['rating_click_per_visit']    = original_df['MEDIAN_MEAL_RATING']/original_df['AVG_CLICKS_PER_VISIT']



# Taking log of the revenue
original_df['Ln_revenue']                = np.log(original_df['REVENUE'])



################################################################################
# Train/Test Split
################################################################################



# Creating X variable for the model
x_val_log = ['avg_vid_time_rating',           
            'rating_click_per_visit',             
            'MEDIAN_MEAL_RATING_out_1',        
            'avg_vid_time_larg_order',           
            'avg_vid_customer_service',                   
            'MASTER_CLASSES_ATTENDED',            
            'meals_ordered_photo',          
            'avg_prep_total_photo',              
            'TOTAL_PHOTOS_VIEWED',                      
            'CONTACTS_W_CUSTOMER_SERVICE_out',   
            'AVG_CLICKS_PER_VISIT',
            'Frequency_orders',
            'UNIQUE_MEALS_PURCH',
            'PACKAGE_LOCKER',
            'UNIQUE_MEALS_PURCH_out',
            'LARGEST_ORDER_SIZE_out',
            'junk',
            'personal',
            'professional'
            ]



# train_test_split
# training data on the x variable that has statisticaly significant values except the tareget variable
chef_data = original_df.loc[: , x_val_log]



# target data on the y variable that has the target that we need to achieve
chef_target = original_df.loc[:, "Ln_revenue"]



# test_size = 0.25 and randome_state = 222
X_train, X_test, y_train, y_test = train_test_split(chef_data,
                                                    chef_target,
                                                    test_size = 0.25,
                                                    random_state = 222)



################################################################################
# Final Model (instantiate, fit, and predict)
################################################################################



# Instantiating the final model, I am using Linear Regression for my final approach
final_model = sklearn.linear_model.LinearRegression()



# Model fitting on the training data
final_fit= final_model.fit(X_train, y_train)



# Model prediction on test data
final_pred = final_model.predict(X_test)



################################################################################
# Final Model Score (score)
################################################################################



# Scoring final model on the traing and testing set
# Printing the score for the model
print('Training Score:', final_model.score(X_train, y_train).round(3))
print('Testing Score:',  final_model.score(X_test, y_test).round(3))



# Saving the results in variable test_score
test_score = final_model.score(X_test, y_test)



# %%
