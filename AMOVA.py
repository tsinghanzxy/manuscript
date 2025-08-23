import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
import os
import random # For permutation testing
from tqdm import tqdm # For progress bar

def calculate_allele_differences(row1, row2):
    """
    Calculates the number of differing alleles between two strains.
    Assumes missing values (NaN) are handled prior or are ignored.
    """
    # Ensure both rows are numeric (after imputation)
    diff = np.abs(row1 - row2)
    # Count where differences are non-zero (i.e., alleles are different)
    return np.sum(diff > 1e-9) # Use a small epsilon for floating point comparison

def calculate_amova_components(distance_matrix, y_source):
    """
    Helper function to calculate AMOVA sum of squares and variance percentages.
    Separated for use in permutation testing.
    """
    groups = y_source.unique()
    num_groups = len(groups)
    N = distance_matrix.shape[0]

    if num_groups < 2:
        return 0, 0, 0, 0, 0 # Return zeros if not enough groups

    SST = np.sum(distance_matrix ** 2) / (2 * N)

    SSW = 0
    group_indices_map = {group: y_source[y_source == group].index.tolist() for group in groups}

    # Ensure indices map correctly to the original full distance matrix
    # The current distance_matrix uses the indices of X_imputed_df, which are aligned with merged_df
    # So, we need to map the original DataFrame indices (merged_df.index) to integer positions for slicing distance_matrix
    original_indices_to_pos = {idx: i for i, idx in enumerate(merged_df.index)}

    for group in groups:
        original_group_indices = group_indices_map[group]
        # Convert original DataFrame indices to positions in the distance_matrix
        current_group_positions = [original_indices_to_pos[idx] for idx in original_group_indices]

        if len(current_group_positions) > 1:
            group_dist_matrix = distance_matrix[np.ix_(current_group_positions, current_group_positions)]
            SSW += np.sum(group_dist_matrix ** 2) / (2 * len(current_group_positions))

    SSA = SST - SSW

    percent_variance_among = (SSA / SST) * 100 if SST > 1e-9 else 0
    percent_variance_within = (SSW / SST) * 100 if SST > 1e-9 else 0

    return SST, SSW, SSA, percent_variance_among, percent_variance_within

def perform_amova_analysis(allele_file, metadata_file, output_dir="amova_results", num_permutations=1000):
    """
    Performs a simplified AMOVA-like analysis on cgMLST allele data with
    permutation testing for statistical significance.

    Args:
        allele_file (str): Path to the cgMLST allele data file (TSV).
                            First column: strain ID, subsequent columns: allele calls.
        metadata_file (str): Path to the metadata file (TSV).
                            First column: strain ID, Fourth column: Source.
        output_dir (str): Directory to save the output results.
        num_permutations (int): Number of permutations for p-value calculation.
    """
    print("Starting AMOVA-like analysis with Python...")

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

    # 2. Preprocess Data
    print("\nStep 2: Preprocessing data...")

    # Standardize strain ID column names for merging
    allele_strain_id_col = allele_df.columns[0]
    allele_df.rename(columns={allele_strain_id_col: 'Strain_ID'}, inplace=True)

    meta_strain_id_col = metadata_df.columns[0]
    meta_source_col = metadata_df.columns[3] # Fourth column
    metadata_df.rename(columns={meta_strain_id_col: 'Strain_ID',
                                 meta_source_col: 'Source'}, inplace=True)

    metadata_subset = metadata_df[['Strain_ID', 'Source']].drop_duplicates(subset=['Strain_ID'])

    global merged_df # Declare merged_df as global to be accessible in calculate_amova_components for index mapping
    merged_df = pd.merge(allele_df, metadata_subset, on='Strain_ID', how='inner')

    if merged_df.empty:
        print("Error: Merging allele and metadata resulted in an empty DataFrame.")
        print("Please check that strain identifiers match between the files and that column indices are correct.")
        return
    print(f"Merged data shape: {merged_df.shape}")

    # Filter by source: 'Clinical' or 'Environmental'
    merged_df['Source'] = merged_df['Source'].astype(str).str.strip()
    valid_sources = ['Clinical', 'Environmental']
    merged_df = merged_df[merged_df['Source'].isin(valid_sources)]

    if merged_df.empty:
        print(f"Error: No strains found with 'Clinical' or 'Environmental' source after filtering.")
        print(f"Please ensure the 'Source' column (4th column in metadata) contains these exact terms.")
        return
    if len(merged_df['Source'].unique()) < 2:
        print("Error: After filtering, less than two unique source groups remain. AMOVA requires at least two groups.")
        return

    print(f"Data shape after filtering for Clinical/Environmental sources: {merged_df.shape}")

    X = merged_df.iloc[:, 1:allele_df.shape[1]] # Allele data
    y_source_original = merged_df['Source'] # Original group labels

    # Handle non-numeric allele data and missing values
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

    X_imputed_df = X_imputed_df.astype(int)

    # 3. Calculate Genetic Distances (once for the original data)
    print("\nStep 3: Calculating pairwise genetic distances (number of allele differences)...")
    distances = pdist(X_imputed_df.values, metric=calculate_allele_differences)
    distance_matrix = squareform(distances)
    print(f"Distance matrix shape: {distance_matrix.shape}")

    # 4. Perform AMOVA-like Calculation for Observed Data
    print("\nStep 4: Performing AMOVA-like calculation for observed data...")
    SST_obs, SSW_obs, SSA_obs, percent_among_obs, percent_within_obs = \
        calculate_amova_components(distance_matrix, y_source_original)

    print(f"Observed Percentage of Variation Among Groups ('Source'): {percent_among_obs:.2f}%")

    # 5. Perform Permutation Test for P-value
    print(f"\nStep 5: Performing Permutation Test ({num_permutations} permutations) for p-value...")
    
    perm_percent_among = []
    # Make a copy of the original labels to shuffle
    shuffled_labels = y_source_original.copy().tolist()

    # Using tqdm for a progress bar
    for _ in tqdm(range(num_permutations), desc="Permutations"):
        random.shuffle(shuffled_labels)
        # Create a Series from shuffled_labels with the original index
        y_source_perm = pd.Series(shuffled_labels, index=y_source_original.index)

        # Calculate AMOVA components for the permuted data
        _, _, _, perm_percent_among_val, _ = \
            calculate_amova_components(distance_matrix, y_source_perm)
        perm_percent_among.append(perm_percent_among_val)

    # Calculate p-value
    # p-value = (number of permutations where perm_percent_among >= percent_among_obs + 1) / (num_permutations + 1)
    # The '+1' in numerator and denominator is a common practice to avoid p-value of 0,
    # and to account for the observed value itself as one of the permutations.
    p_value_count = np.sum(np.array(perm_percent_among) >= percent_among_obs)
    p_value = (p_value_count + 1) / (num_permutations + 1)

    print(f"\nObserved Percentage of Variation Among Groups: {percent_among_obs:.2f}%")
    print(f"P-value (based on {num_permutations} permutations): {p_value:.4f}")

    # 6. Output Results
    print("\nStep 6: AMOVA Results Summary")
    
    results_data = {
        'Component': ['Among Groups (Source)', 'Within Groups', 'Total'],
        'Sum of Squares': [SSA_obs, SSW_obs, SST_obs],
        'Percentage of Variance': [percent_among_obs, percent_within_obs, 100.0]
    }
    results_df = pd.DataFrame(results_data)
    
    results_df.loc[len(results_df)] = ['P-value', '', p_value] # Add p-value row
    
    output_excel_path = os.path.join(output_dir, "amova_results_with_pvalue.xlsx")
    results_df.to_excel(output_excel_path, index=False)
    print(f"AMOVA results with p-value saved to {output_excel_path}")

    print("\nInterpretation:")
    print(f"The P-value ({p_value:.4f}) indicates the probability of observing a 'Percentage of Variation Among Groups'")
    print(f"as extreme as {percent_among_obs:.2f}% by chance, if there were no actual genetic differentiation between sources.")
    print("Typically, if p-value < 0.05, the observed genetic differentiation is considered statistically significant.")
    print("This means the 'Source' (Clinical vs Environmental) significantly explains a portion of the genetic variation.")

    return results_df

if __name__ == '__main__':
    allele_data_file = "cgMLST95_LP_remove_duplicate2.txt"
    metadata_info_file = "metadata_4147LP_new2.txt"
    
    if not os.path.exists(allele_data_file) or not os.path.exists(metadata_info_file):
        print(f"Warning: One or both input files ('{allele_data_file}', '{metadata_info_file}') not found in the current directory.")
        print("Please ensure the files are correctly named and placed, or update the file paths in the script.")
        print("\n**For testing, you can uncomment the lines below to create dummy files.**")
        # print("Creating dummy files for a quick test run (replace with your actual data)...")
        # num_samples = 50
        # num_loci = 100
        # dummy_allele_data = {'Strain': [f's{i}' for i in range(num_samples)]}
        # for i in range(num_loci):
        #     if i < 50:
        #         alleles = np.random.randint(1, 5, num_samples)
        #         if i % 2 == 0:
        #             alleles[:num_samples//2] = np.random.randint(5, 10, num_samples//2)
        #         else:
        #             alleles[num_samples//2:] = np.random.randint(10, 15, num_samples - num_samples//2)
        #     else:
        #         alleles = np.random.randint(1, 20, num_samples)
        #     dummy_allele_data[f'locus{i+1}'] = alleles
        # pd.DataFrame(dummy_allele_data).to_csv(allele_data_file, sep='\t', index=False)

        # dummy_meta_data = {'StrainID': [f's{i}' for i in range(num_samples)],
        #                    'col2': 'x', 'col3':'y',
        #                    'SourceInfo': ['Clinical'] * (num_samples//2) + ['Environmental'] * (num_samples - num_samples//2)}
        # pd.DataFrame(dummy_meta_data).to_csv(metadata_info_file, sep='\t', index=False)
        # print("Dummy files created. Please run the script again.")
        print("Execution halted. Please provide valid input files or uncomment dummy file creation for testing.")
    else:
        print(f"Using provided files: '{allele_data_file}' and '{metadata_info_file}'")
        # You can adjust num_permutations, e.g., 9999 for higher precision in p-value
        amova_results_df = perform_amova_analysis(allele_data_file, metadata_info_file, num_permutations=999)
        if amova_results_df is not None:
            print("\nFinal AMOVA Results DataFrame (including P-value):")
            print(amova_results_df)