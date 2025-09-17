import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("Comprehensive Data Analysis App")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Try reading the CSV with different encodings
    encodings = ['utf-8', 'ISO-8859-1', 'windows-1252']
    df = None

    for encoding in encodings:
        try:
            df = pd.read_csv(uploaded_file, encoding=encoding)
            st.success(f"File loaded successfully with encoding: {encoding}")
            break  # Exit the loop if successful
        except (UnicodeDecodeError, pd.errors.ParserError):
            st.warning(f"Failed to read the file with encoding: {encoding}")

    if df is not None:
        # Proceed with your EDA code here
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

        # Select columns for plotting
        st.subheader("Select Variables to Plot")
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = df.select_dtypes(include='object').columns.tolist()

        # Select X and Y variables for plotting
        x_var = st.selectbox("Select X variable", numeric_cols)
        y_var = st.selectbox("Select Y variable", numeric_cols)

        # Scatter Plot
        if st.button("Generate Scatter Plot"):
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x_var, y=y_var, ax=ax)
            ax.set_title(f"Scatter Plot of {y_var} vs {x_var}")
            st.pyplot(fig)

        # Box Plot
        if st.button("Generate Box Plot"):
            fig, ax = plt.subplots()
            sns.boxplot(x=x_var, y=y_var, data=df, ax=ax)
            ax.set_title(f"Box Plot of {y_var} grouped by {x_var}")
            st.pyplot(fig)

        # Violin Plot
        if st.button("Generate Violin Plot"):
            fig, ax = plt.subplots()
            sns.violinplot(x=x_var, y=y_var, data=df, ax=ax)
            ax.set_title(f"Violin Plot of {y_var} grouped by {x_var}")
            st.pyplot(fig)

        # Histogram for X variable
        if st.button("Generate Histogram for X"):
            fig, ax = plt.subplots()
            sns.histplot(df[x_var], bins=20, kde=True, ax=ax)
            ax.set_title(f"Histogram of {x_var}")
            st.pyplot(fig)

        # Correlation Heatmap (only for numeric columns)
        if st.button("Show Correlation Heatmap"):
            st.subheader("Correlation Heatmap")
            correlation = df.corr()
            fig, ax = plt.subplots()
            sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)

        # Show Data Types
        st.subheader("Data Types")
        st.write(df.dtypes)

        # Download cleaned data option
        if st.button("Download Cleaned Data"):
            cleaned_file = df.to_csv(index=False)
            st.download_button(label="Download CSV", data=cleaned_file, file_name='cleaned_data.csv', mime='text/csv')
