import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("Comprehensive Data Analysis App")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Show dataset preview
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Show basic info
    st.subheader("Summary Statistics")
    st.write(df.describe(include='all'))

    # Check for missing values
    st.subheader("Missing Values")
    missing_values = df.isnull().sum()
    st.write(missing_values[missing_values > 0])

    # Option to remove rows with missing values
    if st.checkbox("Remove rows with missing values"):
        df = df.dropna()
        st.success("Rows with missing values have been removed.")
        st.write(df.head())

    # Data type conversion suggestions
    st.subheader("Data Type Conversion Suggestions")
    for column in df.columns:
        if pd.api.types.is_object_dtype(df[column]):
            unique_values = df[column].nunique()
            if unique_values < 20:  # Suggest converting categorical columns with few unique values
                st.write(f"Consider converting '{column}' to category type.")

    # Column selection
    st.subheader("Column-wise Analysis")
    column = st.selectbox("Select a column for analysis", df.columns)

    if pd.api.types.is_numeric_dtype(df[column]):
        st.write(f"Summary of {column}:")
        st.write(df[column].describe())

        # Histogram
        fig, ax = plt.subplots()
        df[column].hist(ax=ax, bins=20)
        ax.set_title(f"Histogram of {column}")
        st.pyplot(fig)

        # Boxplot
        fig, ax = plt.subplots()
        sns.boxplot(x=df[column], ax=ax)
        ax.set_title(f"Boxplot of {column}")
        st.pyplot(fig)

        # Density Plot
        fig, ax = plt.subplots()
        sns.kdeplot(df[column], ax=ax)
        ax.set_title(f"Density Plot of {column}")
        st.pyplot(fig)

    else:
        st.write(f"Value counts of {column}:")
        st.write(df[column].value_counts())

        # Countplot
        fig, ax = plt.subplots()
        sns.countplot(y=df[column], ax=ax)
        ax.set_title(f"Countplot of {column}")
        st.pyplot(fig)

    # Correlation heatmap for numeric columns
    if df.select_dtypes(include='number').shape[1] > 1:
        st.subheader("Correlation Heatmap")
        correlation = df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

    # Pairplot for numeric columns
    if df.select_dtypes(include='number').shape[1] > 1:
        if st.checkbox("Show Pairplot"):
            st.subheader("Pairplot of Numeric Features")
            pairplot_fig = sns.pairplot(df.select_dtypes(include='number'))
            st.pyplot(pairplot_fig)

    # Additional EDA: Distribution of all numeric columns
    st.subheader("Distribution of All Numeric Columns")
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax, bins=20)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

    # Additional EDA: Categorical columns analysis
    st.subheader("Categorical Columns Analysis")
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        st.write(f"Value counts for {col}:")
        st.write(df[col].value_counts())
        
        # Bar plot for categorical columns
        fig, ax = plt.subplots()
        sns.countplot(y=df[col], ax=ax)
        ax.set_title(f"Countplot of {col}")
        st.pyplot(fig)

    # Show Data Types
    st.subheader("Data Types")
    st.write(df.dtypes)

    # Download cleaned data option
    if st.button("Download Cleaned Data"):
        cleaned_file = df.to_csv(index=False)
        st.download_button(label="Download CSV", data=cleaned_file, file_name='cleaned_data.csv', mime='text/csv')

