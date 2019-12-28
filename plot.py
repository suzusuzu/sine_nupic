import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('log.csv')

plt.figure(figsize=(12, 4))
plt.plot(df['true'], label='true')
plt.plot(df['predict'], label='predict')
plt.title('one step prediction')
plt.xlabel('time')
plt.legend()
plt.savefig('fig.png')