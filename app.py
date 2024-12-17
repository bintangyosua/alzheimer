import marimo

__generated_with = "0.9.32"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def __():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        # Melakukan Klasifikasi Penyakit Alzeimer dengan Algoritma `Xgboost` dan `KNN`

        This project is a classification analysis of Alzheimer's disease using two machine learning algorithms: XGBoost and KNN. Below is a breakdown of the steps involved, organized by section.

        1. Panky Bintang Pradana Yosua (H1D022077)
        2. Sarah Shiba Huwaidah (H1D023044)
        3. Isma Fadhilatizzahra (H1D023107)
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Import Libraries

        This section includes necessary Python libraries for data manipulation, visualization, and machine learning modeling:

        - Data Manipulation & Visualization: `numpy`, `pandas`, `matplotlib.pyplot`, `seaborn`, and `altair`
        - Preprocessing: `LabelEncoder`, `StandardScaler`
        - Model Selection & Evaluation: `train_test_split`, `accuracy_score`, `classification_report`, `confusion_matrix`, `ConfusionMatrixDisplay`
        - Machine Learning Models: `KNeighborsClassifier`, `SVC`, `RandomForestClassifier`, `GradientBoostingClassifier`, `ExtraTreesClassifier`, `XGBClassifier`
        - Dimensionality Reduction: `PCA`, `KernelPCA`, `TSNE`, `LocallyLinearEmbedding`, `MDS`, `UMAP`
        """
    )
    return


@app.cell(hide_code=True)
def __():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import altair as alt

    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
    from sklearn.decomposition import PCA, KernelPCA
    from sklearn.manifold import TSNE, LocallyLinearEmbedding, MDS
    from imblearn.over_sampling import SMOTE

    import xgboost as xgb
    return (
        ConfusionMatrixDisplay,
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        KNeighborsClassifier,
        KernelPCA,
        LabelEncoder,
        LocallyLinearEmbedding,
        MDS,
        NearestNeighbors,
        PCA,
        RandomForestClassifier,
        SMOTE,
        SVC,
        StandardScaler,
        TSNE,
        accuracy_score,
        alt,
        classification_report,
        confusion_matrix,
        np,
        pd,
        plt,
        sns,
        train_test_split,
        xgb,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## 1. Exploratory Data Analysis""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Read the dataset

        The dataset is read in and initially cleaned by dropping unnecessary columns (DoctorInCharge and PatientID). Key characteristics of the data are:

        - Primary Key: `PatientID`
        - Categorical Columns: `Gender`, `Ethnicity`, `EducationLevel`, etc.
        - Continuous Columns: `Age`, `BMI`, `AlcoholConsumption`
        """
    )
    return


@app.cell(hide_code=True)
def __(pd):
    df = pd.read_csv('alzheimers_disease_data.csv')
    df = df.drop(['DoctorInCharge', 'PatientID'], axis=1)
    df
    return (df,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Statistical Summary

        Information and summary statistics for each column are provided, along with a count of each unique diagnosis in the Diagnosis column.
        """
    )
    return


@app.cell(hide_code=True)
def __(df):
    df.describe().reset_index()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Number of Diagnoses for Each Label""")
    return


@app.cell(hide_code=True)
def __(alt, df, mo):
    grouped_diagnosis = df['Diagnosis'].value_counts().rename_axis('Diagnosis').reset_index(name='Count')
    diagnosis_bar = alt.Chart(grouped_diagnosis).mark_bar().encode(
        x='Diagnosis:N',
        y='Count:Q'
    )

    mo.ui.altair_chart(diagnosis_bar)
    return diagnosis_bar, grouped_diagnosis


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ### Correlation Analysis

        Allows you to select continuous variables to generate scatter plots. This helps visualize the relationships between variables in the dataset.
        """
    )
    return


@app.cell(hide_code=True)
def __(df, pd):
    columns = df.columns
    continuous_columns = [
        col for col in columns if (pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 10)
    ]
    return columns, continuous_columns


@app.cell(hide_code=True)
def __(continuous_columns, mo):
    x_scatter = mo.ui.dropdown(continuous_columns, value=continuous_columns[0])
    y_scatter = mo.ui.dropdown(continuous_columns, value=continuous_columns[1])
    return x_scatter, y_scatter


@app.cell(hide_code=True)
def __(mo, x_scatter, y_scatter):
    mo.md(f"""
    #### Scatter Plot Continuous Variables

    Choosen variables to be in `scatterplot`

    x: {x_scatter} &nbsp;
    y: {y_scatter}

    """)
    return


@app.cell(hide_code=True)
def __(alt, df, mo, x_scatter, y_scatter):
    # brush = alt.selection_point(encodings=["x"])

    scatter_bar_chart = (alt.Chart(df)
                          .mark_point()
                          .encode(
                              x=x_scatter.value,
                              y=y_scatter.value,
                              color='Diagnosis')
                          # .add_params(brush)
                         )
    scatter_bar_chart = mo.ui.altair_chart(scatter_bar_chart)
    return (scatter_bar_chart,)


@app.cell(hide_code=True)
def __(mo, scatter_bar_chart):
    mo.vstack([scatter_bar_chart, scatter_bar_chart.value.head()])
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""### Make Your Own Visualization""")
    return


@app.cell(hide_code=True)
def __(df, mo):
    mo.ui.data_explorer(df)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ### Heatmap Correlation

        The correlation matrix between variables is displayed as a heatmap. This visual helps in identifying highly correlated features.
        """
    )
    return


@app.cell
def __(df):
    correlation_matrix = df.corr()
    correlation_melted = correlation_matrix.reset_index().melt(id_vars='index')
    correlation_melted.columns = ['Variable1', 'Variable2', 'Correlation']
    return correlation_matrix, correlation_melted


@app.cell
def __(alt, correlation_melted):
    alt.Chart(correlation_melted).mark_rect().encode(
        x='Variable1:O',
        y='Variable2:O',
        color='Correlation:Q',
        tooltip=['Variable1', 'Variable2', 'Correlation']
    ).properties(
        width=420,
        height=420,
        title='Correlation Heatmap'
    ).interactive()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Data Preprocessing 

        #### Split the Dataset

        The dataset is split into input features `X` and target variable `y`, and further divided into training and testing sets.
        """
    )
    return


@app.cell
def __(df):
    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']
    return X, y


@app.cell
def __(SMOTE, X, alt, mo, pd, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_resampled_df = pd.DataFrame(X_resampled, columns=X_resampled.columns)
    y_resampled_df = pd.DataFrame(y_resampled, columns=['Diagnosis'])

    new_df = pd.concat([X_resampled_df, y_resampled_df], axis=1)

    balanced_df = new_df['Diagnosis'].value_counts().rename_axis('Diagnosis').reset_index(name='Count')
    balanced_diagnosis_bar = alt.Chart(balanced_df).mark_bar().encode(
        x='Diagnosis:N',
        y='Count:Q'
    )

    mo.ui.altair_chart(balanced_diagnosis_bar)
    return (
        X_resampled,
        X_resampled_df,
        balanced_df,
        balanced_diagnosis_bar,
        new_df,
        smote,
        y_resampled,
        y_resampled_df,
    )


@app.cell
def __(X_resampled, train_test_split, y_resampled):
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, random_state=42, test_size=0.2)
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Normalized Dataset

        The data is scaled using `StandardScaler` to ensure consistent scales across features. This normalization step is critical for algorithms like `KNN`, which rely on distance-based calculations.
        """
    )
    return


@app.cell(hide_code=True)
def __(StandardScaler, X_test, X_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_test_scaled, X_train_scaled, scaler


@app.cell
def __(X_train_scaled):
    print(X_train_scaled)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## 2. Model Construction""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Make Model

        The project includes two classification models: `XGBoost` and `KNN`. A slider allows the user to select the number of neighbors for `KNN`.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    neighbors = mo.ui.slider(3, 99, 2, 7)
    mo.md(f"""
        Enter number of neighbors for KNN (neighbors value): {neighbors}
    """)
    return (neighbors,)


@app.cell(hide_code=True)
def __(mo, neighbors):
    mo.md(f"Number of Neighbors: {neighbors.value}")
    return


@app.cell
def __():
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from scipy.stats import uniform
    return GridSearchCV, RandomizedSearchCV, uniform


@app.cell
def __(GridSearchCV, X_train_scaled, xgb, y_train):
    xgb_model = xgb.XGBClassifier()

    # Parameter Grid
    param_grid = {
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1],
    }

    # Grid Search
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
    grid_search.fit(X_train_scaled, y_train)

    print("Best Params:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)
    return grid_search, param_grid, xgb_model


@app.cell
def __(GridSearchCV, KNeighborsClassifier, X_train_scaled, y_train):
    knn_model = KNeighborsClassifier()

    knn_param_grid = {
        'n_neighbors': [i for i in range(3, 100, 2)]
    }

    knn_grid_search = GridSearchCV(estimator=knn_model, param_grid=knn_param_grid, cv=3, scoring='accuracy', verbose=1)
    knn_grid_search.fit(X_train_scaled, y_train)

    print("Best Params:", knn_grid_search.best_params_)
    print("Best Score:", knn_grid_search.best_score_)
    return knn_grid_search, knn_model, knn_param_grid


@app.cell
def __(
    KNeighborsClassifier,
    X_train_scaled,
    grid_search,
    neighbors,
    xgb,
    y_train,
):
    models = {
        "XGBoost": xgb.XGBClassifier(
            colsample_bytree=grid_search.best_params_['colsample_bytree'],
            learning_rate=grid_search.best_params_['learning_rate'],
            max_depth=grid_search.best_params_['max_depth'],
            n_estimators=grid_search.best_params_['n_estimators'],
            subsample=grid_search.best_params_['subsample']),
        "KNN": KNeighborsClassifier(n_neighbors=neighbors.value),
    }

    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
    return model, model_name, models


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""<!-- ### Decision Boundary Visualization -->""")
    return


@app.cell(hide_code=True)
def __():
    # def dimensional_reduction(model, X_scaled):
    #     X_scaled = model.fit_transform(X_scaled)

    #     return X_scaled
    return


@app.cell(hide_code=True)
def __():
    # def decision_boundary(model, X_scaled, y, model_name, ax):
    #     """
    #     Function to plot the decision boundary of a given model using 2D data.

    #     Parameters:
    #     model: The classifier (e.g., XGBoost, KNN).
    #     X_train_2d: 2D transformed training data (from any dimensionality reduction technique).
    #     X_test_2d: 2D transformed testing data (from any dimensionality reduction technique).
    #     y_train: Training labels.
    #     y_test: Testing labels.
    #     model_name: Name of the model (e.g., 'XGBoost', 'KNN').
    #     ax: The axis object for plotting.

    #     Returns:
    #     ax: The updated axis with the decision boundary plot.
    #     """
    #     # Fit the model
    #     model.fit(X_scaled, y)

    #     # Set up the mesh grid for plotting decision boundaries
    #     x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    #     y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    #     xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
    #                          np.arange(y_min, y_max, 0.1))

    #     # Predict the label for each point on the mesh grid
    #     Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    #     Z = Z.reshape(xx.shape)

    #     # Plot decision boundary on the provided axis
    #     contour = ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    #     scatter_test = sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=y, ax=ax, palette="coolwarm")

    #     # Add labels and title
    #     ax.set_xlabel('Component 1')
    #     ax.set_ylabel('Component 2')
    #     ax.set_title(f'Decision Boundary of {model_name} with {model_name}')
    #     ax.legend(title='Survived', loc='upper right')

    #     return ax  # Return the updated axis
    return


@app.cell(hide_code=True)
def __():
    # X_scaled = np.vstack((X_train_scaled, X_test_scaled))

    # pca_model = PCA(n_components=2)
    # manifold_models = {
    #     # 'pca': PCA(n_components=2),
    #     # 'lle': LocallyLinearEmbedding(n_components=2),
    #     # 'kernel_pca': KernelPCA(n_components=2),
    #     # 'mds-auto': MDS(n_components=2, normalized_stress='auto'),
    #     # 'mds-false': MDS(n_components=2, normalized_stress=False),
    #     'tsne': TSNE(n_components=2),
    #     'umap': UMAP(n_components=2)
    # }
    return


@app.cell(hide_code=True)
def __():
    # X_2d = {}
    # for dimred_name, dimred_model in manifold_models.items():
    #     X_2d[dimred_name] = dimensional_reduction(dimred_model, X_scaled)
    return


@app.cell(hide_code=True)
def __():
    # decision_boundary_models = {
    #     'XGBoost': xgb.XGBClassifier(),
    #     'knn': KNeighborsClassifier(n_neighbors=77)
    # }
    return


@app.cell(hide_code=True)
def __():
    # fig, axes = plt.subplots(nrows=2, ncols=len(manifold_models), figsize=(15, 10))

    # # Flatten the axes array for easy indexing
    # axes = axes.flatten()

    # for i, (model_name_dr, model_dr) in enumerate(decision_boundary_models.items()):
    #     for j, (dimred_name_dr, dimred_model_dr) in enumerate(manifold_models.items()):
    #         X_2d_dr = X_2d[dimred_name_dr]
    #         ax = axes[i * len(manifold_models) + j]
    #         decision_boundary(model_dr, X_2d_dr, y, model_name_dr, ax)

    #         ax.set_title(f'{model_name_dr} with {dimred_name_dr}')

    # plt.tight_layout()
    # plt.show()
    return


@app.cell
def __():
    # df['Diagnosis'].value_counts()
    return


@app.cell(hide_code=True)
def __():
    # xg_boost_scatter_bar_chart = (
    #     alt.Chart(
    #         pd.DataFrame(
    #             np.column_stack(
    #                 (X_2d['tsne'], y)
    #             ),
    #             columns=['Component 1', 'Component 2', 'Diagnosis']
    #         )
    #     )
    #     .mark_point()
    #     .encode(
    #         x='Component 1',
    #         y='Component 2',
    #         color='Diagnosis'
    #     )
    # )

    # mo.ui.altair_chart(xg_boost_scatter_bar_chart)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## 3. Model Evaluation""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Confusion Matrix

        Confusion matrices for both `XGBoost` and `KNN` are generated to provide insight into the true positives, false positives, true negatives, and false negatives.
        """
    )
    return


@app.cell(hide_code=True)
def __(ConfusionMatrixDisplay, confusion_matrix, plt):
    def display_confusion_matrix(y_test, y_pred, model):
        cm = confusion_matrix(
            y_test, y_pred, 
            labels=model.classes_
        )
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=model.classes_)
        disp.plot()
        plt.show()
    return (display_confusion_matrix,)


@app.cell(hide_code=True)
def __(X_test_scaled, models):
    y_pred = {}
    y_pred['XGBoost'] = models['XGBoost'].predict(X_test_scaled)
    y_pred['KNN'] = models['KNN'].predict(X_test_scaled)
    return (y_pred,)


@app.cell(hide_code=True)
def __(alt, confusion_matrix, pd):
    def altair_confusion_matrix(y_test, y_pred, model_name):
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=['False', 'True'], columns=['False', 'True'])
        cm_df = cm_df.reset_index().melt(id_vars='index', var_name='Predicted', value_name='Count')
        cm_df.rename(columns={'index': 'Actual'}, inplace=True)

        heatmap = alt.Chart(cm_df, height=400, width=400).mark_rect().encode(
            x=alt.X('Predicted:N', title='Predicted Label', axis=alt.Axis(labelFontSize=14, titleFontSize=16, labelAngle=0)),
            y=alt.Y('Actual:N', title='Actual Label', axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
            color=alt.Color('Count:Q', scale=alt.Scale(scheme='blues'), title='Count'),
            tooltip=[alt.Tooltip('Actual:N'), alt.Tooltip('Predicted:N'), alt.Tooltip('Count:Q')]
        ).properties(
            title=f'Confusion Matrix Heatmap of {model_name} (True vs False)'
        )

        text_annotations = heatmap.mark_text(
            align='center',
            baseline='middle',
            fontSize=20,
        ).encode(
            color=alt.value('black'),
            text='Count:Q'
        )

        return heatmap + text_annotations
    return (altair_confusion_matrix,)


@app.cell(hide_code=True)
def __(alt, altair_confusion_matrix, mo, y_pred, y_test):
    cm_xgb = altair_confusion_matrix(y_test, y_pred['XGBoost'], 'XGBoost')
    cm_knn = altair_confusion_matrix(y_test, y_pred['KNN'], 'KNN')

    mo.ui.altair_chart(alt.hconcat(cm_xgb, cm_knn))
    return cm_knn, cm_xgb


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ### Model Report and Metrics

        Each model’s accuracy and classification report are printed for further analysis.
        """
    )
    return


@app.cell(hide_code=True)
def __(accuracy_score, classification_report, mo):
    def evaluate_model_report(y_test, y_pred, model_name):
        with mo.redirect_stdout():
            accuracy = "{:.2f}%".format(round(accuracy_score(y_test, y_pred) * 100, 2))
            print(f'Accuracy score of {model_name}: {accuracy}')
            print(classification_report(y_test, y_pred))
    return (evaluate_model_report,)


@app.cell(hide_code=True)
def __(evaluate_model_report, models, y_pred, y_test):
    for key, value in models.items():
        evaluate_model_report(y_test, y_pred[key], key)
    return key, value


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Interactive Barplot of XGBoost and KNN accuracies""")
    return


@app.cell(hide_code=True)
def __(accuracy_score, alt, mo, pd, y_pred, y_test):
    xgb_accuracy = "{:.2f}".format(round(accuracy_score(y_test, y_pred['XGBoost']) * 100, 2))
    knn_accuracy = "{:.2f}".format(round(accuracy_score(y_test, y_pred['KNN']) * 100, 2))

    results_df = pd.DataFrame(
        {
            'Model': ['XGBoost', 'KNN'],
            'Accuracy': [float(xgb_accuracy), float(knn_accuracy)]
        }
    )

    mo.ui.altair_chart(
        alt.Chart(results_df, width=300)
        .mark_bar(fill='#4c78a8')
        .encode(
            x=alt.X('Model:O', axis=alt.Axis(labelAngle=0)),
            y='Accuracy'
        )
    )
    return knn_accuracy, results_df, xgb_accuracy


@app.cell(hide_code=True)
def __(knn_accuracy, mo, xgb_accuracy):
    mo.md(
        f"""
        ## 5. Kesimpulan

        Model dengan Xgboost mendapatkan akurasi **{xgb_accuracy}%**, sedangkan KNN hanya memperoleh akurasi **{knn_accuracy}%**
        """
    )
    return


if __name__ == "__main__":
    app.run()
