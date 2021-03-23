
import matplotlib.pyplot as plt
import pandas as pd

plot_1_df = pd.read_excel("Results/Results_CNN1/cnn_1_loss_time.xlsx")
plot_2_df = pd.read_excel("Results/Results_CNN1/cnn_2_loss_time.xlsx")
plot_1_df.rename( columns={'Unnamed: 0':'Epoch', 'Loss on epoch':'CNN_ver_1'}, inplace=True )
plot_2_df.rename( columns={'Unnamed: 0':'Epoch', 'Loss on epoch':'CNN_ver_2'}, inplace=True )
print(plot_1_df)
ax = plt.gca()
plot_1_df.plot(kind='line', x='Epoch', y='CNN_ver_1', ax=ax)
plot_2_df.plot(kind='line', x='Epoch', y='CNN_ver_2', color='red', ax=ax)
plt.ylabel("Błąd funkcji celu")
plt.show()
