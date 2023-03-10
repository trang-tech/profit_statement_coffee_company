draw_heatmap
def draw_heatmap(df, columns, cmap='coolwarm'):
    corr_matrix = df[columns].corr()
    sns.heatmap(corr_matrix, cmap=cmap)
    plt.show()