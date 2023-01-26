from src.pairplot import make_pair_plot
import seaborn as sns


data = sns.load_dataset('diamonds')
print(data.head())
modified_df, labeled_dict = make_pair_plot(data,convert_categorical=True, categories_boundary=10, rotation=30, save_path='../data/diamonds.jpg',title='Diamonds dataset', return_data=True,show_plot=False,verbose=True)
print(modified_df)
print(labeled_dict)
