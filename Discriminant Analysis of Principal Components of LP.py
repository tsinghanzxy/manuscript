import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import StratifiedKFold # For cross-validation
from sklearn.metrics import accuracy_score

def perform_dapc_like_analysis_python(allele_file, metadata_file, output_dir="dapc_results_python"):
    """
    Performs a DAPC-like analysis using Python (PCA followed by LDA).
    Includes:
    1. Cross-validation to optimize the number of PCA components.
    2. Calculation of variable contributions to the LDA discriminant axis.

    Args:
        allele_file (str): Path to the cgMLST allele data file (TSV).
                            First column: strain ID, subsequent columns: allele calls.
        metadata_file (str): Path to the metadata file (TSV).
                            First column: strain ID, Fourth column: Source.
        output_dir (str): Directory to save the output plot and Excel files.
    """
    print("Starting DAPC-like analysis with Python...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data
    print("\nStep 1: Loading data...")
    try:
        allele_df = pd.read_csv(allele_file, sep='\t', low_memory=False)
        metadata_df = pd.read_csv(metadata_file, sep='\t', low_memory=False)
    except FileNotFoundError as e:
        print(f"Error: Input file not found: {e}")
        return
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    print(f"Allele data shape: {allele_df.shape}")
    print(f"Metadata shape: {metadata_df.shape}")

    # 2. Preprocess Data
    print("\nStep 2: Preprocessing data...")

    # Standardize strain ID column names for merging
    allele_strain_id_col = allele_df.columns[0]
    allele_df.rename(columns={allele_strain_id_col: 'Strain_ID'}, inplace=True)

    meta_strain_id_col = metadata_df.columns[0]
    meta_source_col = metadata_df.columns[3]
    metadata_df.rename(columns={meta_strain_id_col: 'Strain_ID',
                                 meta_source_col: 'Source'}, inplace=True)

    metadata_subset = metadata_df[['Strain_ID', 'Source']].drop_duplicates(subset=['Strain_ID'])

    merged_df = pd.merge(allele_df, metadata_subset, on='Strain_ID', how='inner')

    if merged_df.empty:
        print("Error: Merging allele and metadata resulted in an empty DataFrame.")
        print("Please check that strain identifiers match between the files and that column indices are correct.")
        print(f"Sample Strain IDs in allele data (first 5): {allele_df['Strain_ID'].unique()[:5]}")
        print(f"Sample Strain IDs in metadata (first 5): {metadata_subset['Strain_ID'].unique()[:5]}")
        return
    print(f"Merged data shape: {merged_df.shape}")

    merged_df['Source'] = merged_df['Source'].astype(str).str.strip()
    valid_sources = ['Clinical', 'Environmental']
    original_sources_count = merged_df['Source'].value_counts()
    print(f"Original source counts:\n{original_sources_count}")

    merged_df = merged_df[merged_df['Source'].isin(valid_sources)]

    if merged_df.empty:
        print(f"Error: No strains found with 'Clinical' or 'Environmental' source after filtering.")
        print(f"Please ensure the 'Source' column (4th column in metadata) contains these exact terms.")
        return
    print(f"Data shape after filtering for Clinical/Environmental sources: {merged_df.shape}")

    X = merged_df.iloc[:, 1:allele_df.shape[1]]
    y_source = merged_df['Source']
    original_feature_names = X.columns.tolist() # Keep original feature names

    print(f"Shape of feature matrix X: {X.shape}")
    print(f"Number of Clinical samples: {sum(y_source == 'Clinical')}")
    print(f"Number of Environmental samples: {sum(y_source == 'Environmental')}")

    if X.empty or y_source.empty:
        print("Error: Feature matrix X or labels y_source is empty after processing.")
        return
    if X.shape[0] < 2:
        print("Error: Less than 2 samples available for analysis after filtering.")
        return
    if sum(y_source == 'Clinical') == 0 or sum(y_source == 'Environmental') == 0:
        print("Error: At least one sample from each group (Clinical, Environmental) is required for LDA.")
        return


    print("Converting allele data to numeric and checking for missing values...")
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    coerced_nan_count_after = X.isna().sum().sum()
    print(f"Missing values in X (initially or after coercing non-numeric to NaN): {coerced_nan_count_after}")

    if coerced_nan_count_after > 0:
        print("Imputing missing values using the most frequent value for each locus...")
        imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        X_imputed = imputer.fit_transform(X)
        X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
        print(f"Missing values in X after imputation: {X_imputed_df.isna().sum().sum()}")
    else:
        X_imputed_df = X.copy()
        print("No missing values to impute or all were numeric.")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_source)
    print(f"Encoded labels: {le.classes_} -> {np.unique(y_encoded)}")

    print("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed_df)

    n_samples_available = X_scaled.shape[0]
    n_features_available = X_scaled.shape[1]
    n_groups = len(le.classes_)

    # Determine maximum reasonable PCA components
    max_pca_components = min(n_samples_available - 1, n_features_available)
    if max_pca_components < 1:
        print("Error: Not enough data points or features to perform PCA.")
        return

    # 3. Cross-validation to find optimal number of PCA components
    print("\nStep 3: Performing cross-validation to find optimal PCA components (similar to xvalDapc)...")
    # Test a range of PCA components. Ad-hoc range for demonstration.
    # In a real scenario, you might test up to max_pca_components, or a more granular range.
    # Limiting to a maximum of 100 for computational efficiency in many datasets.
    pca_component_range = list(range(1, min(max_pca_components + 1, 101))) # Test up to 100 PCs or max_pca_components

    if not pca_component_range:
        print("Error: PCA component range is empty. Cannot perform cross-validation.")
        return

    cv_accuracies = []
    n_splits = min(5, n_samples_available // n_groups) # Ensure at least 1 sample per group per split
    if n_splits < 2:
        print(f"Warning: Not enough samples for meaningful cross-validation ({n_samples_available} samples, {n_groups} groups). Skipping cross-validation and using default PCA component selection.")
        best_n_pca = min(50, max_pca_components) # Fallback to heuristic
        if best_n_pca < 1: best_n_pca = 1 # Ensure at least 1
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for n_pca in pca_component_range:
            fold_accuracies = []
            if n_pca > n_samples_available - n_splits: # Prevent issues where n_components > samples in a fold
                 print(f"Skipping n_pca={n_pca} as it's too high for some cross-validation folds.")
                 continue

            for train_index, test_index in skf.split(X_scaled, y_encoded):
                X_train, X_test = X_scaled[train_index], X_scaled[test_index]
                y_train, y_test = y_encoded[train_index], y_encoded[test_index]

                if len(np.unique(y_train)) < n_groups or len(np.unique(y_test)) < n_groups:
                    # This can happen if a fold doesn't get all classes, skip
                    continue

                pca_cv = PCA(n_components=n_pca)
                X_pca_train = pca_cv.fit_transform(X_train)
                X_pca_test = pca_cv.transform(X_test)

                # Ensure n_components for LDA is valid (min(n_classes - 1, n_features_after_pca_train))
                n_lda_cv_components = min(n_groups - 1, X_pca_train.shape[1])
                if n_lda_cv_components < 1: # Should not happen if n_pca >= 1
                    continue

                lda_cv = LDA(n_components=n_lda_cv_components)
                lda_cv.fit(X_pca_train, y_train)
                y_pred = lda_cv.predict(X_pca_test)
                fold_accuracies.append(accuracy_score(y_test, y_pred))

            if fold_accuracies:
                cv_accuracies.append((n_pca, np.mean(fold_accuracies)))
            else:
                cv_accuracies.append((n_pca, 0)) # If no valid folds for this n_pca

        if not cv_accuracies:
            print("Cross-validation failed to produce results. Falling back to heuristic PCA component selection.")
            best_n_pca = min(50, max_pca_components)
            if best_n_pca < 1: best_n_pca = 1
        else:
            best_n_pca = max(cv_accuracies, key=lambda item: item[1])[0]
            print(f"Cross-validation results (PCA components vs. accuracy):\n{cv_accuracies}")
            print(f"Optimal number of PCA components chosen by cross-validation: {best_n_pca}")

        # Plot cross-validation results
        if cv_accuracies:
            pca_nums = [item[0] for item in cv_accuracies]
            accuracies = [item[1] for item in cv_accuracies]
            plt.figure(figsize=(10, 6))
            plt.plot(pca_nums, accuracies, marker='o')
            plt.xlabel('Number of PCA Components')
            plt.ylabel('Mean Cross-Validation Accuracy')
            plt.title('Cross-Validation for Optimal PCA Components in DAPC-like Analysis')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.axvline(x=best_n_pca, color='r', linestyle='--', label=f'Optimal PCs: {best_n_pca}')
            plt.legend()
            cv_plot_path = os.path.join(output_dir, "pca_cross_validation_accuracy.pdf")
            plt.savefig(cv_plot_path)
            print(f"Cross-validation plot saved to {cv_plot_path}")
            plt.close()
        else:
             print("Skipping cross-validation plot as no valid results were obtained.")


    # Perform final PCA with the optimal number of components
    print(f"\nStep 4: Performing final PCA with {best_n_pca} components...")
    pca = PCA(n_components=best_n_pca)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Shape of data after PCA: {X_pca.shape}")
    print(f"Cumulative explained variance by {best_n_pca} PCs: {np.sum(pca.explained_variance_ratio_):.4f}")

    # 5. LDA
    print("\nStep 5: Performing LDA on PCA components...")
    n_lda_components = min(n_groups - 1, X_pca.shape[1])
    
    if n_lda_components < 1:
        print(f"Error: Cannot perform LDA. Number of LDA components is {n_lda_components}.")
        print("This can happen if number of PCA components retained is 0, or only one class is present after filtering.")
        return
    
    lda = LDA(n_components=n_lda_components)
    X_lda = lda.fit_transform(X_pca, y_encoded)
    print(f"Shape of data after LDA: {X_lda.shape}")

    # 6. Calculate Variable Contributions (Allele Loci)
    print("\nStep 6: Calculating contributions of original allele loci to LDA discriminant axis...")
    # The contribution is a product of PCA components' loadings and LDA's linear combination coefficients.
    # PCA.components_ has shape (n_components, n_features)
    # LDA.coef_ has shape (n_classes - 1, n_components_pca)
    # We want (n_features, n_classes - 1) contributions
    # So, (n_features, n_components_pca) @ (n_components_pca, n_classes - 1)
    
    # Get the raw allele feature names
    allele_feature_names = original_feature_names
    
    # Calculate the overall transformation matrix from original features to LDA space
    # It's pca.components_.T @ lda.coef_.T
    # This matrix tells how much each original feature contributes to each LDA axis.
    # Shape: (n_original_features, n_lda_components)
    feature_contributions_to_lda = pca.components_.T @ lda.coef_.T

    # For 2 groups, n_lda_components is 1, so we have a single column of contributions
    if n_lda_components == 1:
        contributions_df = pd.DataFrame({
            'Locus': allele_feature_names,
            'Contribution_to_LDA1': feature_contributions_to_lda[:, 0]
        })
        # Sort by absolute contribution to find most influential loci
        contributions_df['Absolute_Contribution'] = contributions_df['Contribution_to_LDA1'].abs()
        contributions_df = contributions_df.sort_values(by='Absolute_Contribution', ascending=False)

        contrib_excel_path = os.path.join(output_dir, "locus_contributions_to_lda.xlsx")
        contributions_df.to_excel(contrib_excel_path, index=False)
        print(f"Locus contributions to LDA saved to {contrib_excel_path}")
        print("\nTop 10 contributing loci to LDA1:")
        print(contributions_df.head(10))
    else:
        print("Skipping locus contributions as LDA resulted in more than 1 component. (Logic needs extension for multiple LDA axes)")
        # If you have more than 2 groups and want contributions to multiple LDA axes,
        # you'd need to iterate through feature_contributions_to_lda[:, i] for each i.

    # 7. Visualization (remains largely the same)
    print("\nStep 7: Visualizing LDA results...")
    if X_lda.shape[1] == 1:
        lda_df = pd.DataFrame({'LDA1': X_lda[:, 0], 'Source': y_source, 'Strain_ID': merged_df['Strain_ID'].values})
        
        plt.figure(figsize=(12, 7))
        sns.histplot(data=lda_df, x='LDA1', hue='Source', kde=True, element="step", stat="density", common_norm=False)
        plt.title('DAPC-like Analysis: LDA of PCA Components', fontsize=16)
        plt.xlabel('Linear Discriminant Axis 1', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.legend(title='Source', labels=le.classes_)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        lda_plot_path = os.path.join(output_dir, "dapc_like_lda_plot.pdf")
        plt.savefig(lda_plot_path)
        print(f"LDA plot saved to {lda_plot_path}")
        plt.close()
    else:
        print(f"LDA resulted in {X_lda.shape[1]} components. Plotting first two if available.")
        if X_lda.shape[1] >= 2:
            lda_df = pd.DataFrame({'LDA1': X_lda[:, 0], 'LDA2': X_lda[:, 1], 'Source': y_source, 'Strain_ID': merged_df['Strain_ID'].values})
            plt.figure(figsize=(12, 8))
            sns.scatterplot(data=lda_df, x='LDA1', y='LDA2', hue='Source', style='Source', s=100, alpha=0.7)
            plt.title('DAPC-like Analysis: LDA of PCA Components (First Two Axes)', fontsize=16)
            plt.xlabel('Linear Discriminant Axis 1', fontsize=14)
            plt.ylabel('Linear Discriminant Axis 2', fontsize=14)
            plt.legend(title='Source')
            plt.grid(True, linestyle='--', alpha=0.7)
            lda_plot_path = os.path.join(output_dir, "dapc_like_lda2D_plot.pdf")
            plt.savefig(lda_plot_path)
            print(f"LDA 2D plot saved to {lda_plot_path}")
            plt.close()

    # 8. Save numerical results to Excel (including Strain_ID for traceability)
    print("\nStep 8: Saving numerical results to Excel...")
    if 'lda_df' in locals():
        excel_output_path = os.path.join(output_dir, "lda_results.xlsx")
        cols_to_order = ['Strain_ID', 'Source'] + [col for col in lda_df.columns if col not in ['Strain_ID', 'Source']]
        lda_df = lda_df[cols_to_order]
        lda_df.to_excel(excel_output_path, index=False)
        print(f"LDA numerical results saved to {excel_output_path}")
    else:
        print("LDA results DataFrame not created, skipping Excel output.")

    # 9. Interpretation and Caveats
    print("\nStep 9: Interpretation and Caveats...")
    print("- The cross-validation plot shows how classification accuracy changes with the number of PCA components.")
    print("- The 'Locus Contributions' Excel file lists allele loci by their importance in discriminating between groups.")
    print("- Positive contributions mean higher allele values for one group (e.g., Clinical), negative for the other (e.g., Environmental) on LDA1.")
    print("- **Remember**: This Python implementation is a DAPC-like analysis. For complex genetic data and full `adegenet` features (like `find.clusters` or more specialized genetic data structures), the R package is still the gold standard.")

    if hasattr(lda, 'explained_variance_ratio_'):
        print(f"Proportion of variance explained by LDA components: {lda.explained_variance_ratio_}")

    print(f"\nAnalysis finished. Results (plots and Excel files) saved in '{output_dir}' directory.")
    return lda_df

if __name__ == '__main__':
    allele_data_file = "cgMLST95_LP_remove_duplicate.txt"
    metadata_info_file = "4147LP_metadata.txt"
    
    if not os.path.exists(allele_data_file) or not os.path.exists(metadata_info_file):
        print(f"Warning: One or both input files ('{allele_data_file}', '{metadata_info_file}') not found in the current directory.")
        print("Please ensure the files are correctly named and placed, or update the file paths in the script.")
        print("\n**For testing, you can uncomment the lines below to create dummy files.**")
        # print("Creating dummy files for a quick test run (replace with your actual data)...")
        # # Make sure you create enough unique alleles and samples for cross-validation to work well
        # dummy_allele_data = {'Strain': [f's{i}' for i in range(50)], # Increased samples
        #                      'locus1': np.random.randint(1,10,50), 'locus2': np.random.randint(1,10,50),
        #                      'locus3': np.random.randint(1,10,50), 'locus4': np.random.randint(1,10,50),
        #                      'locus5': np.random.randint(1,10,50), 'locus6': np.random.randint(1,10,50)}
        # pd.DataFrame(dummy_allele_data).to_csv(allele_data_file, sep='\t', index=False)
        # dummy_meta_data = {'StrainID': [f's{i}' for i in range(50)], 'col2': 'x', 'col3':'y',
        #                    'SourceInfo': np.random.choice(['Clinical', 'Environmental'],50, p=[0.5, 0.5])}
        # pd.DataFrame(dummy_meta_data).to_csv(metadata_info_file, sep='\t', index=False)
        # allele_data_file = "cgMLST95_LP_remove_duplicate2.txt"
        # metadata_info_file = "4147LP_metadata2.txt"
        print("Execution halted. Please provide valid input files or uncomment dummy file creation for testing.")
    else:
        print(f"Using provided files: '{allele_data_file}' and '{metadata_info_file}'")
        results_df = perform_dapc_like_analysis_python(allele_data_file, metadata_info_file, output_dir="dapc_python_output")
        if results_df is not None:
            print("\nFirst 5 rows of LDA results DataFrame:")
            print(results_df.head())