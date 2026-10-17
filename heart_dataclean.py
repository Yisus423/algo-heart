import pandas as pd 
df = pd.read_csv('heart_disease_risk_2026.csv')
print("-"*20,"DATASET ORIGINAL","-"*20)
df.info() 

df["sex"]= df["sex"].apply(lambda x: 1 if x == 2 else 0) # cambiamos a int
df.drop(['patient_id','smoker_status','chest_pain_type',
'hba1c','bmi','alcohol_units_per_week','sleep_hours','stress_score','diet_quality_score','st_depression', 'cholesterol_total'],axis=1,inplace=True)

print("-"*20,"DATASET LIMPIO","-"*20)
df.info()

df.to_csv('heart_clean.csv',index=False )
print("El archivo fué guardado en heart_clean.csv")
