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
        st.subheader("Select Variables for Plots")

        # Scatter Plot
        st.subheader("Scatter Plot")
        x_var_scatter = st.selectbox("Select X variable for Scatter Plot", df.select_dtypes(include='number').columns.tolist())
        y_var_scatter = st.selectbox("Select Y variable for Scatter Plot", df.select_dtypes(include='number').columns.tolist())
        if st.button("Generate Scatter Plot"):
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x_var_scatter, y=y_var_scatter, ax=ax)
            ax.set_title(f"Scatter Plot of {y_var_scatter} vs {x_var_scatter}")
            st.pyplot(fig)

        # Box Plot
        st.subheader("Box Plot")
        x_var_box = st.selectbox("Select X variable for Box Plot", df.select_dtypes(include='number').columns.tolist())
        y_var_box = st.selectbox("Select Y variable for Box Plot", df.select_dtypes(include='number').columns.tolist())
        if st.button("Generate Box Plot"):
            fig, ax = plt.subplots()
            sns.boxplot(x=x_var_box, y=y_var_box, data=df, ax=ax)
            ax.set_title(f"Box Plot of {y_var_box} grouped by {x_var_box}")
            st.pyplot(fig)

        # Violin Plot
        st.subheader("Violin Plot")
        x_var_violin = st.selectbox("Select X variable for Violin Plot", df.select_dtypes(include='number').columns.tolist())
        y_var_violin = st.selectbox("Select Y variable for Violin Plot", df.select_dtypes(include='number').columns.tolist())
        if st.button("Generate Violin Plot"):
            fig, ax = plt.subplots()
            sns.violinplot(x=x_var_violin, y=y_var_violin, data=df, ax=ax)
            ax.set_title(f"Violin Plot of {y_var_violin} grouped by {x_var_violin}")
            st.pyplot(fig)

        # Histogram for X variable
        st.subheader("Histogram")
        x_var_hist = st.selectbox("Select variable for Histogram", df.select_dtypes(include='number').columns.tolist())
        if st.button("Generate Histogram for X"):
            fig, ax = plt.subplots()
            sns.histplot(df[x_var_hist], bins=20, kde=True, ax=ax)
            ax.set_title(f"Histogram of {x_var_hist}")
            st.pyplot(fig)

        # Pair Plot for numeric columns
        if st.button("Generate Pair Plot"):
            st.subheader("Pair Plot of Numeric Variables")
            fig = sns.pairplot(df.select_dtypes(include='number'))
            st.pyplot(fig)

        # Count Plot for categorical variables
        if st.button("Generate Count Plot"):
            if df.select_dtypes(include='object').shape[1] > 0:
                cat_var = st.selectbox("Select Categorical Variable for Count Plot", df.select_dtypes(include='object').columns.tolist())
                fig, ax = plt.subplots()
                sns.countplot(data=df, x=cat_var, ax=ax)
                ax.set_title(f"Count Plot of {cat_var}")
                st.pyplot(fig)
            else:
                st.warning("No categorical columns available for count plot.")

        # Unique Values Summary
        st.subheader("Unique Values in Each Column")
        unique_values = {col: df[col].nunique() for col in df.columns}
        st.write(unique_values)

        # Show Data Types
        st.subheader("Data Types")
        st.write(df.dtypes)

        # Download cleaned data option
        if st.button("Download Cleaned Data"):
            cleaned_file = df.to_csv(index=False)
            st.download_button(label="Download CSV", data=cleaned_file, file_name='cleaned_data.csv', mime='text/csv')
