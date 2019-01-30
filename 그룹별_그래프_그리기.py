fig, ax = plt.subplots(figsize=(8,6))
result.groupby('측정소명').plot(x="datetime", y="PM10",ax=ax) 
