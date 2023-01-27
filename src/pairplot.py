import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Tuple, List


# %%
def make_pair_plot(
        df: pd.DataFrame,
        convert_categorical: bool = False,
        categories_boundary: int = None,
        label_dict=None,
        rotation: int = 0,
        save_path: str = None,
        title: str = None,
        fig_size: Tuple[int, int] = None,
        return_data: bool = False,
        show_plot: bool = True,
        verbose=False,
) -> Union[None, Tuple[pd.DataFrame, dict]]:
    """
    This function creates a pair plot of the input dataframe.
    Pair plot is a good way to visualize the relationship between all the features of a dataframe.

    Parameters:
    :param df: pandas dataframe
    :param convert_categorical: it will automatically convert non-numeric columns to categorical-columns
    :param categories_boundary: convert numeric categorical columns to categorical
    :param label_dict: dictionary containing mapping of the categorical columns
    :param rotation: x_ticks rotation
    :param save_path: path to save the plot
    :param title: title of the SUP-PLOT
    :param fig_size: size of the figure
    :param return_data: if user wants converted_df and its dictionary
    :param show_plot: to display drawn plot
    :param verbose: to show details of process

    :return: None|tuple(pd.DataFrame,dict)

    Example:
        preprocessing of data before passing it if it contains categorical cols is as follows
        >>> data = sns.load_dataset('diamonds')
        >>> data = data [['carat', 'cut', 'clarity', 'depth', 'price']]
        >>> cut = pd.Categorical(data.cut)
        >>> data.cut = cut.codes
        >>> clarity = pd.Categorical(data.clarity)
        >>> data.clarity = clarity.codes
        >>> label_dict={'cut':cut.categories,'clarity':clarity.categories}
        >>> make_pair_plot(data,categories_boundary=10,label_dict=label_dict, verbose=False)
        showing plot
        >>> make_pair_plot(data, convert_categorical=True, categories_boundary=10, rotation=30, verbose=False)
        showing plot
    """
    # creating a copy of the dataframe
    df = df.copy()

    # initializing label_dictionary for categorical values
    if label_dict is None:
        label_dict = {}

    # initializing categorical column as a set as it might be repeated in the below two steps
    cat_cols = set()

    numeric_dtypes = [
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "float32",
        "float64",
    ]

    # automatically converting categorical cols to numerical
    if convert_categorical:
        # selecting all cols except numeric types
        cat_cols = cat_cols.union(df.select_dtypes(exclude=numeric_dtypes).columns)
        for cat_col in cat_cols:
            # converting them to numeric and storing their mapping
            classes = pd.Categorical(df[cat_col])
            df[cat_col] = classes.codes
            label_dict[cat_col] = classes.categories

    # selecting only numeric types
    df = df.select_dtypes(numeric_dtypes)

    columns = df.columns

    # getting categorical cols names from numerical columns according to given boundary
    if categories_boundary:
        cat_cols = cat_cols.union(df.columns[(df.nunique() <= categories_boundary)])

    if verbose:
        print("Categorical columns =", cat_cols)

    # making a fig suitable for plot_number of columns
    fig_size = (8 * len(columns), 8 * len(columns)) if fig_size is None else fig_size
    title_size = 30 if fig_size is None else 5
    figure, ax = plt.subplots(len(columns), len(columns), figsize=fig_size)

    # setting title of plot
    figure.suptitle(
        title if title is not None else "PAIR PLOT", fontsize=title_size * len(columns)
    )

    plot_number = 1
    # running loop for every col as a row
    for row, rx in zip(columns, ax):
        # running loop for every col as a column
        for col, cx in zip(columns, rx):
            if col in cat_cols:
                # setting rotation of xticks
                cx.set_xticklabels(labels=[], rotation=rotation)
            labeled_plot = False

            if (
                    col == row
            ):  # if the col is a categorical column then it shows its value count
                if col in cat_cols:
                    sns.countplot(
                        x=df[col].apply(
                            lambda x: label_dict[col][x] if col in label_dict else x
                        ),
                        ax=cx,
                    )
                else:  # it shows distribution of data if column is not a categorical one
                    sns.histplot(
                        df[row], stat="density", alpha=0.5, ax=cx, color="darkblue"
                    )
                    sns.kdeplot(df[row], ax=cx, color="darkblue")
                    # sns.histplot(df[row], ax=cx,kde=True,color='darkblue')
                    cx.set(xlabel="", ylabel="")

            elif col in cat_cols:
                # if cols are different but both are categorical then showing counts of column with hue set as row
                if row in cat_cols:
                    sns.countplot(
                        x=df[col].apply(
                            lambda x: label_dict[col][x] if col in label_dict else x
                        ),
                        hue=df[row].apply(
                            lambda x: label_dict[row][x] if row in label_dict else x
                        ),
                        ax=cx,
                    )
                # if only the col is a categorical column then showing a box plot between them
                else:
                    sns.boxplot(
                        y=df[row],
                        x=df[col].apply(
                            lambda x: label_dict[col][x] if col in label_dict else x
                        ),
                        ax=cx,
                    )
            # if only the row is a categorical column then showing a violin plot between them
            elif row in cat_cols:
                sns.violinplot(
                    y=df[row].apply(
                        lambda x: label_dict[row][x] if row in label_dict else x
                    ),
                    x=df[col],
                    orient="horizontal",
                    ax=cx,
                )
                labeled_plot = True
            else:  # if both row and col are numerical than showing a regression plot between them
                sns.regplot(df, x=col, y=row, ax=cx, color="darkblue")

            # Some plots are already labeled but if it's not labeled, we make sure to label it for better readability and understanding of the data.
            if not labeled_plot:
                cx.set(xlabel=col, ylabel=row)
            # showing processes info if required
            if verbose:
                print(f"Done plot {plot_number}/{len(columns) * len(columns)}")
            # increasing the number of plots drawn
            plot_number += 1
    if save_path:
        print(f"saving plot as {save_path}")
        plt.savefig(save_path, dpi=300)
    if show_plot:
        print("showing plot")
        plt.show()
    else:
        plt.close()
    # returning data and label dictionary
    if return_data:
        return df, label_dict
