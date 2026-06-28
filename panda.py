import pandas as pd

#series is a 1d labeled array
data=[100,102,104]
series= pd.Series(data, index=["a","b","c"])#constructor not function
print("series", series)
print("series with a particular label", series.loc["b"])

#change loc using integer and index
series.loc["c"]=201
print("series", series)
series.iloc[2]=200
print("series", series)

#to print series with condition
print("element greater then 100:",  series[series>100])

#dictionary
death={"year1":200, "year2":302, "year3":210}
series=pd.Series(death)
#updating
series.loc["year3"]=211
print("series", series)


#dataframe
data= {"name":["arush", "peter", "parth", "aditya"],
       "age":[18,13,17,19]}
df=pd.DataFrame(data, index=["a","b","c","d"])
print("dataframe",df)
print("at a perticular location", df.loc["b"])

#add new column
df["job"]= ["code", "bring coffe", "eat", "protect"]
print("updates dataframe\n", df)

#add new row
new_row=pd.DataFrame([{"name": "sandy", "age":15, "job": "manager"}])
df=pd.concat([df,new_row])
print("with updated row\n", df)