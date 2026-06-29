import pandas as pd
df=pd.read_csv("pokemon.csv")
print(df)

#selection by column
print("selection by column\n", df["Name"])
print("selection of multiple comlumns\n" ,df[["Name", "Type 1"]])
#selection by row
print("selection by row ", df.loc[0])
print("selection by row for particular column part", df.loc[0, ["Name"]])

# to show user inputed pokeman stats
# df=pd.read_csv("pokemon.csv",index_col="Name")
# pokemon=input("enter pokemon name ")
# print("stats are...\n ", df.loc[pokemon])

#filtering- to output data based on particular condition
tall_pokemon=df[df["Type 1"]=="Grass"]
print(tall_pokemon)

#data cleaning
#droping column when not needed
df_2=df.drop(columns=["Type 1", "index_name" ])
# df_2=df.dropna(subset=["Type 2"])
df_2=df.fillna({"Type 2": "None"})#this fills the nan place
print("new dataframe", df_2)
#standerizing text
df["Name"]=df["Name"].str.lower()
print("lower case names", df)