#!/usr/bin/env python
# coding: utf-8

# In[1]:


import platform  # imports go at the top of your script
import os  # start with imports from the standard library followed by a space

import numpy as np  # then import custom/user packages
import pandas as pd  # you can use shorter names for easy reference
import matplotlib.pyplot as plt  # np, pd, and plt are common for these packages

# this is a IPython magic command, you may ignore it
get_ipython().run_line_magic('matplotlib', 'inline')

print(f"My Python version is: {platform.python_version()}")
print(f"My Numpy version is: {np.__version__}")
print(f"My Pandas version is: {pd.__version__}")


# # Introduction to Pandas
# 
# <img src="https://pandas.pydata.org/pandas-docs/stable/_static/pandas.svg" alt="pandas logo" width="200"/>
# 
# ## Lets get started
# 
# Time to get started! The following exercises will introduce you to data inspection and mangling with the Python package Pandas. The theme of this section is Data Reading & Manipulation. You will learn to:
# -	Work with DataFrames
# -	Read and write files
# -	Select Data
# -	Filtering Data (Missing Values)
# -	Manipulate Data (Sort, Group)
# 
# ````{admonition} Tip
# :class: tip
# We put a lot of links to the Pandas [documentation](https://pandas.pydata.org/pandas-docs/stable/), this is because I want you to get used to looking at the documentation to solve a problem. If you want to you can also keep the [cheat-sheet](http://pandas.pydata.org/Pandas_Cheat_Sheet.pdf) for this session nearby and hang it above your bed. Remember to look up functions and methods (``ctrl+I``) in Spyder or type:
# 
# ```bash
# ?pd.read_csv
# # or
# help(pd.read_csv)
# ````
# 
# > **Objective:**
# >
# > Your research objective is to see how the total lifted weight of lifters in the IPF in the world championships has developed over time, to answer this question we will use a real dataset, but first we have to go over the basics.
# >
# 
# ### Assignment 0
# 
# Remember the dictionary *lifters* with all the information on the top lifters in the International Powerlifting Federation of the pre-assignment? It is convenient to put the type of data arrangement in the name (dic for dictionairy, li for lists, df for dataframes) 
# 
# ```python
# names = ["Andrzej Stanaszek", "Ray Williams", "Sergey Fedosienko", 
#          "Don Reinhoudt", "Dennis Cornelius", "Brett Gibbs", "John Haack"]
# ages = [27, 30, 33, 28, 35, 27, None]
# totalkg = [592.0, 585.5, 584.5, 567.5, 563.0, 555.5, 543.5]
# 
# dic_lifters = {"Name": names, "Age": ages, "Totalkg": totalkg} # make it a dict
# 
# ```
# 
# - **Convert the dictionary into a DataFrame (df_lifters) using the [DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) class from the Pandas package and print to check.**

# In[23]:


names = ["Andrzej Stanaszek", "Ray Williams", "Sergey Fedosienko", 
         "Don Reinhoudt", "Dennis Cornelius", "Brett Gibbs", "John Haack"]  # [] => list constructor
ages = [27, 30, 33, 28, 35, 27, None]
totalkg = [592.0, 585.5, 584.5, 567.5, 563.0, 555.5, 543.5]

dic_lifters = {"Name": names, "Age": ages, "Totalkg": totalkg}  # {} => dict constructor

df_lifters = pd.DataFrame(dic_lifters)
print(df_lifters)


# ## Indexing with .loc and .iloc
# 
# The result of your print statement already looks pretty nice. What you should notice is that we now get an index next to our dataframe, by default these are integers. In this case it's just an integer index because we didn't specify anything. This could also have been, for example, dates or IDs. The index is a very powerful tool, especially when you start slicing your data and merging DataFrames. 
# 
# Which bring us to the next advantage of DataFrames: slicing. You can index columns in a dataframe just like you would with a dictionary, but now you can also select rows.
# ```python
# df_lifters["Name"]  # returns the name column as a Series
# df_lifters.Name  # returns the name column as a Series
# df_lifters[["Name"]]  # returns the name column as a DataFrame
# df_lifters[1:5]  # returns the second till fifth row
# df_lifters[[True, False, True, False, False, True, False]]  # booleans indices return rows
# ```
# 
# **As a mini-challenge, play around with this in the console. If you don't hit an error, you haven't played around enough!**
# 
# However, if you try to slice the row and the column at the same time you will get an error. This is because Pandas does not know if you are using integer slices or 'names' (remember that the index can be anything).
# 
# To do so we need a location indexer. Pandas provides two options here: ``.iloc`` and ``.loc`` ([docs](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html)): 
# - iloc gives you traditional Python slicing \[start:stop:step, start:stop:step\] for \[rows, columns\]
# - loc gives you label based/boolean indexing \[name lifter:name other lifter, Totalkg\] for \[rows, columns\]
# 
# **Some important gotcha's:** 
# - As a mnemonic you can remember that the i in iloc stands for integer and thus refers to numbers.
# - The syntax is dataframe.loc\[...,...\], with square brackets, not with parentheses (it's not a function call).
# - Iloc is incluse:exclusive, so you will slice up till -but not including- the last index. This is [useful](http://www.cs.utexas.edu/users/EWD/transcriptions/EWD08xx/EWD831.html) because the size of the resulting array is the same as exclusive-inclusive, but it does take some training.
# - However, loc includes both the start and stop index, this is useful because you are often looking for very specific indices with loc.
# - If you ask for a single column or row you get a Series object. If you want a dataframe instead, you can prevent this by putting the single index in a list object (like above) by using square brackets \[...\].
# - If you want to get all the rows of columns, you can use ``:`` .
# - Like with most operations in Python you will get a view (saves memory) most of the times and not a copy (use [.copy()](https://docs.python.org/3/library/copy.html) to get a copy).
# 
# ### Assignment 1
# 
# You already know traditional Python indexing from DataCamp so that seems like a good place to start.
# 
# - **Select and print the first 2 lifters using .iloc.**
# 
# - **Select and print the last 2 lifters using .iloc.**
# 
# - **Select and print the names of all lifters as a DataFrame using .iloc.**
# 
# - **Select and print the names of all lifters as a DataFrame using .iloc, make sure you get a DataFrame.**
# 
# You should get something like this:

# In[4]:


print(df_lifters.iloc[:2, :], "\n")  # print with an extra newline for readability
print(df_lifters.iloc[-2:, :], "\n") # In python, "\n" is the same as an enter
print(df_lifters.iloc[:, 0], "\n")
print(df_lifters.iloc[:, [0]], "\n")


# ## (Re)setting the index
# 
# By default, the index is an array of integers. However, you can pretty much set anything to be the index. For example, you could use the names of the lifters as the index, or maybe their birthdate. Setting the index is done with ``.set_index()`` (who would have guessed). Resetting the index can be done with ``.reset_index()``. 
# 
# This method places the index column by default back in your DataFrame and gives you an integer index again. However, you can specify certain key arguments in this method to change this, to drop the index: ``drop=True``
# 
# Setting and resetting the index is a very powerful way to interact with your DataFrame. It is also a common source for errors, so always be aware of your index! (if you are completely lost, just run your initital DataFrame again and start over)
# 
# ### Assignment 2
# 
# Now you should try the same thing with .loc. Remember that .loc includes both the start and stop index.
# 
# - **Set "Name" as the [index](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.set_index.html) of your dataframe and print the result.**
# 
# - **Select and print the first 2 lifters using .loc.**
# 
# - **Select and print the last 2 lifters using .loc.**
# 
# - **Select and print the Totalkg of all lifters as a DataFrame using .loc.**
# 
# You should get something like this:

# In[24]:


df_lifters.set_index("Name", inplace=True)  # same as df_lifters = df_lifters.set_index("Name")
print(df_lifters.loc["Andrzej Stanaszek":"Ray Williams", :], "\n")
print(df_lifters.loc["Brett Gibbs":"John Haack", :], "\n")
print(df_lifters.loc[:, ["Totalkg"]], "\n")


# ## Copy
# 
# We already briefly mentioned that slicing in Pandas (and most other Python objects) returns a view not a copy. This might be a little counter intuitive if you're not familiar with general purpose programming languages (MATLAB is not). We do not want to mess with our lifter_df, so we will make a new dataframe for this assignment with three columns, ranging from 1-10:
# ```python
# df1 = pd.DataFrame({"X": list(range(10)), "Y": list(range(10)), "Z": list(range(10))})
# ```
# 
# ### Assignment 3
# 
# - **Make a slice of the first five rows using .iloc or .loc and assign it to a new variable.**
# 
# - **Select all samples with .iloc or .loc and set all samples in the new variable to 0 and print the DataFrame.**
# 
# - **Now print the original DataFrame. What do you notice?**
# 
# You should get something like this:

# In[25]:


df1 = pd.DataFrame({"X": list(range(10)), "Y": list(range(10)), "Z": list(range(10))})
df2 = df1.iloc[:5, :]
df2.loc[:] = 0
print("Sliced df:\n", df2, "\n") # \n gives you an empty line after your print statement for readability
print("Original df:\n", df1)


# ```{warning}
# Oh no, you've not only altered df1, but also df2. This is because the slicing operation gave you a view into the DataFrame, but not a copy of the data. Again, this saves a lot of memory, but it can mess up your data! Luckily Pandas gives us a warning when we try to do this!
# ```
# 
# To prevent this problem you can use the ``.copy()`` method which returns you a copy and not a view.
# 
# Note: whether Pandas returns a copy or a view is actually a pretty delicate topic, but just assume you get a view and use ``.copy()`` when you plan on changing the contents of the DataFrame.
# 
# ## Accessors
# 
# We often use mixed data in Pandas. What if I want to apply a [string method](https://docs.python.org/3/library/stdtypes.html#string-methods) to an entire row or column? For this we have Accessors, they provide an interface to additional methods, specifically for: strings, categorical data, and datetime-like data in a way that they can be applied to the entire column.
# 
# For example, if we want to remove all whitespace from the name column we would use ``.replace(" ", "")`` on every string. 
# We *could* loop over every row using the ``.iterrows()``, or we *could* write a lambda function. However, the string accessor makes this very easy for us and it vectorizes the operation:
# 
# ```python
# df_lifters["Name"].str.replace(" ", "")  # removes all whitespace from entire column
# df_lifters.columns.str.replace(" ", "")  # removes all whitespace from column names
# ```
# ```{admonition} Tip
# :class: tip
# Think about this example and maybe play around with it. Do these lines change the underlying data, or just return a view with changed data? What is the difference between the two?
# ```
# 
# Cleaning up strings is a common operation in data science. Always check (your column names) for unwanted whitespace!
# 
# ### Assignment 4
# 
# Consider an entry like this: {"Name": "ALEXEY Kuzmin", "Age": 34, "Totalkg": 527.25}. We can add it to the dataframe using the [concat](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html) method: 
# 
# ```python
# df_lifters_to_add = pd.DataFrame({"Name": "ALEXEY Kuzmin", "Age": 34, "Totalkg": 527.25}, index=[0])
# df_lifters = pd.concat([df_lifters, df_lifters_to_add], ignore_index=True) # overwrite df_lifters to be the new dataframe where the new lifter is added
# ```
# 
# His first name is fully capitalized unlike all the other names in the DataFrame, this makes you a little more nervous than you would like to admit. Using Python's string methods this can easily be fixed by doing a ``.title()`` [\[docs\]](https://docs.python.org/3/library/stdtypes.html#str.title). However, we would need to loop over the entire Series, which would be slow. Luckily, we can use the string accessor ``.str``:
# ````{margin}
# ```{admonition} Tip
# :class: tip
# You are about to use your first method that accepts the inplace keyword argument. Set it to True to make everything more readable!
# ```
# ````
# - **First, [reset](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reset_index.html) the index with the appropriate method so the name column is a column again.**
# - **Add Alexey to the DataFrame. Do you understand what the ignore_index does?**
# - **Use the string accessor on the "Name" column to make sure all names and surnames start with a capitalized letter.**
# 
# You should get something like this:

# In[26]:


df_lifters.reset_index(inplace=True)  # equal to df_lifters = df_lifters.reset_index()
df_lifters_to_add = pd.DataFrame({"Name": "ALEXEY Kuzmin", "Age": 34, "Totalkg": 527.25}, index=[0])
df_lifters = pd.concat([df_lifters, df_lifters_to_add], ignore_index=True) # overwrite df_lifters to be the new dataframe where the new lifter is added
df_lifters["Name"] = df_lifters["Name"].str.title()
df_lifters  # sometimes I don't use print, this is a feature of IPython/Jupyter


# ### Assignment 5
# 
# ````{margin}
# ```{admonition} Tip
# :class: tip
# Use the string accessor method, use the keyword argument expand=True to put your answer in two columns!
# ```
# ````
# - **Make a column for first name and one for surname with the ``.split()`` method. [[docs]](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.split.html)**
# - **Use the string accessor on the "Firstname" and "Surname" columns to make sure there are no whitespaces in the columns**
# - **Change the Name column name to Fullname. Use the ``.rename()`` method (look it up).**
# - **Rearange the columns so they're equal to ["Fullname", "Firstname", "Surname", "Age", "Totalkg"].**
# 
# You should get something like this:

# In[27]:


df_lifters[["Firstname", "Surname"]] = df_lifters["Name"].str.split(expand=True)

df_lifters["Firstname"] = df_lifters["Firstname"].str.replace(" ", "")
df_lifters["Surname"] = df_lifters["Surname"].str.replace(" ", "")
df_lifters.rename(columns={"Name": "Fullname"}, inplace=True)

df_lifters = df_lifters[["Fullname", "Firstname", "Surname", "Age", "Totalkg"]]
df_lifters


# ## Sorting and dropping
# 
# Pandas can also sort your rows for you. Right now, our DataFrame is starting to look like a mess. We can make it look a little more intuitive with the ``.sort_values()`` method. This method takes one or more columns and sorts the DataFrame based on the content those columns. **Look up the documentation for this method, do you understand it?** 
# 
# The age of John Haack is missing. We can choose to fill in the missing value or to just remove him from the DataFrame entirely. We decide that it's too much effort to look up his age (sorry John) and we remove him from the DataFrame. We can use the built-in ``.dropna()`` (alternatively you could use ``.fillna()`` to fill the gaps) method. **Look up the documentation for this method, do you understand it?** 
# 
# Alternatively, when you have duplicates in your data you might want to remove those. The built-in ``.drop_duplicates()`` method comes to the rescue here. **Look up the documentation for this method, do you understand it?** 
# 
# Finally, we can also remove the age column entirely if it does not serve any purpose. By doing this we free up some memory and we have one less column to worry about. For this you can use the ``.drop()`` method in which you specificy if you want to drop a row (axis=0) or a column (axis=1). **Look up the documentation for this method, how does it work?** 
# 
# You can also chain methods if you would like. If a method returns a DataFrame you can use another DataFrame method on it, e.g.:
# ```python
# # First drop all NaNs and then sort the values by Totalkg:
# df_lifters = df_lifters.dropna()  # alternatively use df_lifters.dropna(inplace=True)
# df_lifters = df_lifters.sort_values(by="Totalkg", ascending=False)
# 
# # Same thing with method chaining:
# df_lifters = df_lifters.dropna().sort_values(by="Totalkg", ascending=False)
# ```
# 
# ### Assignment 6
# 
# - **First sort all the data by Totalkg score, make sure the Totalkg is on top of your DataFrame. Print out the DataFrame. What do you notice?**
# 
# - **Remove all rows that contain missing values. Print out the DataFrame. What happened to the index?**
# 
# - **Drop the age column entirely. Print out the DataFrame.**
# 
# - **For the following task you have to use method chaining: sort the data by Firstname (alphabetically), remove the Fullname column and reset the index. Print out the DataFrame.**
# 
# You should get something like this:

# In[28]:


df_lifters.sort_values(by="Totalkg", ascending=True, inplace=True)
print(df_lifters, "\n") # \n gives you an empty line after your print statement for readability

df_lifters.dropna(inplace=True)
print(df_lifters, "\n")

df_lifters.drop("Age", axis=1, inplace=True)
print(df_lifters, "\n")

df_lifters = df_lifters.sort_values(by="Firstname", ascending=True).drop('Fullname', axis=1).reset_index(drop=True)
print(df_lifters, "\n")

