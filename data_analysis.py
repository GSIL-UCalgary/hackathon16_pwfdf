import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# --- Configuration ---
FILE_PATH = "./data/ofr20161106_appx-1.xlsx" 
TARGET_COLUMN = 'Response'
OUTPUT_IMAGE_FEATURE_EDA = './output/feature_distributions_eda.png'
OUTPUT_IMAGE_MDA_HEATMAP = './output/missing_data_and_zeros_heatmap.png'
OUTPUT_IMAGE_MDA_SPLIT = './output/missing_data_split_analysis.png'
OUTPUT_IMAGE_DUPLICATES_ANALYSIS = './output/duplicate_analysis.png' # Output file for duplicate analysis
# Generic prefix for dynamically generated unit plots
OUTPUT_IMAGE_RANGES_PREFIX = './output/feature_ranges_' 

def load_raw_data(file_path, sheet_name="Appendix1_ModelData"):
    """Loads the Excel file and returns the raw DataFrame."""
    print(f"Loading raw data from {file_path} sheet: {sheet_name}...")
    try:
        # Requires the 'openpyxl' library to be installed.
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please check the path.")
        return None
    except ValueError as e:
        print(f"Error reading Excel sheet: {e}. Make sure the sheet name is correct and 'openpyxl' is installed.")
        return None
    except Exception as e:
        print(f"An error occurred during loading: {e}")
        return None

def clean_data(df_raw):
    """Performs imputation and cleaning for feature EDA."""
    df = df_raw.copy()
    
    # Drop rows where the target 'Response' is missing
    df.dropna(subset=[TARGET_COLUMN], inplace=True)

    # Impute missing numerical values with the median for robust analysis
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numerical_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
            # print(f"Filled NaN in '{col}' with median.") # Suppress internal logs
    
    # Convert 'Response' to integer
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    return df

def perform_mda(df_raw):
    """Performs Missing Data Analysis (MDA) and generates visualizations."""
    
    print("\n" + "="*70)
    print("--- 1. Missing Data Analysis (MDA) ---")
    print("="*70)
    
    # Drop columns that are entirely object/datetime and not useful for quantitative missingness
    # 'Database' is now KEPT to be visualized and specially colored.
    df_analysis = df_raw.drop(columns=['Fire Name', 'Year', 'Fire_ID', 'Fire_SegID', 
                                       'StormDate', 'StormStart', 'StormEnd', 'UTM_X', 'UTM_Y'])

    # --- 1a. Overall Missing Value Counts and Heatmap (UPDATED to include 0s and Database coloring) ---
    
    # 1. Create a visualization matrix for Missing/Zero/Present data
    # Numerical Codes:
    # 0: Present & Non-zero (Light Grey)
    # 1: Present & Zero (Orange)
    # 2: Missing/NaN (Red) - Also used for non-Training/Test categorical values
    # 3: Database = 'Training' (Blue)
    # 4: Database = 'Test' (Green)
    
    # Initialize matrix with 0 (Present & Non-zero)
    df_viz = pd.DataFrame(0, index=df_analysis.index, columns=df_analysis.columns, dtype=np.int8)
    
    # Identify numeric columns for checking zeros
    numeric_cols = df_analysis.select_dtypes(include=np.number).columns.tolist()
    
    # --- Step 1: Mark Zeros (Value 1) for Numerical Columns ---
    # Set 1 for zero values where the original data is NOT NaN
    for col in numeric_cols:
        is_zero_and_present = (df_analysis[col] == 0) & (~df_analysis[col].isnull())
        df_viz.loc[is_zero_and_present, col] = 1

    # --- Step 2: Mark NaNs (Value 2) for ALL Columns ---
    # Set 2 for missing values
    for col in df_analysis.columns:
        is_nan = df_analysis[col].isnull()
        df_viz.loc[is_nan, col] = 2
    
    # --- Step 3: Special Coloring for 'Database' Column (Values 3 and 4) ---
    if 'Database' in df_analysis.columns:
        # Mark 'Training' cells with code 3 (Blue)
        is_training = df_analysis['Database'] == 'Training'
        df_viz.loc[is_training, 'Database'] = 3
        
        # Mark 'Test' cells with code 4 (Green)
        is_test = df_analysis['Database'] == 'Test'
        df_viz.loc[is_test, 'Database'] = 4
        
        # If 'Database' had any non-Training/non-Test string value, it would default to 0 (Present) 
        # unless it was NaN (set to 2 in Step 2). We don't worry about non-standard strings here, 
        # assuming 'Training' and 'Test' cover all valid non-NaN data.
    
    # Transpose the dataframe for the heatmap visualization
    df_viz_final = df_viz.T 
    vmax_val = 4 # Max value for the color scale

    # Define custom colormap: 
    # Index 0: Light Grey (Present & Non-Zero)
    # Index 1: Orange (Present & Zero)
    # Index 2: Red (Missing/NaN)
    # Index 3: Blue (Database: Training)
    # Index 4: Green (Database: Test)
    cmap = ListedColormap(['#E0E0E0', '#F7B538', '#DC3912', '#3498DB', '#2ECC71']) 

    # Visualization 1: Missing Data and Zero Value Heatmap
    plt.figure(figsize=(15, 7))
    
    # Create the heatmap
    ax = sns.heatmap(
        df_viz_final, 
        cbar=False, 
        cmap=cmap, 
        linewidths=0.0, 
        linecolor='#333333', 
        xticklabels=False, # No need for observation index labels
        vmin=0, 
        vmax=vmax_val
    )
    
    # Add title and labels
    plt.title('Missing (NaN), Zero Value, and Database Distribution Across Observations', fontsize=16)
    plt.xlabel('Observation Index')
    plt.ylabel('Feature')
    
    # Create manual legend to explain the colors
    legend_elements = [
        # Feature-specific elements
        Patch(facecolor='#DC3912', edgecolor='#333333', label='Missing (NaN)'),
        Patch(facecolor='#F7B538', edgecolor='#333333', label='Zero Value (0)'),
        Patch(facecolor='#E0E0E0', edgecolor='#333333', label='Present & Non-zero'),
    ]
    
    if 'Database' in df_analysis.columns:
        # Add Database-specific elements
        legend_elements.extend([
            Patch(facecolor='#3498DB', edgecolor='#333333', label='Database: Training Set'),
            Patch(facecolor='#2ECC71', edgecolor='#333333', label='Database: Test Set')
        ])

    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_MDA_HEATMAP, dpi=300)
    plt.close()
    print(f"\nSaved Missing Data & Zero Value Heatmap to: {OUTPUT_IMAGE_MDA_HEATMAP}")
    
    # --- 1b. Missingness Split by Database (Training vs. Test) and Response (0 vs. 1) ---
    
    if 'Database' in df_raw.columns and TARGET_COLUMN in df_raw.columns:
        
        # --- Missingness Split by Database ---
        missing_by_db = df_raw.groupby('Database').apply(lambda x: x.isnull().sum() / len(x) * 100).T
        missing_by_db_pct = missing_by_db[missing_by_db.sum(axis=1) > 0].sort_values(by='Training', ascending=False)

        print("\n--- Missing Percentage Split by Database (Training vs. Test) ---")
        print(missing_by_db_pct.round(2))

        # --- Missingness Split by Response ---
        # Ensure Response is clean for grouping
        df_temp = df_raw.dropna(subset=[TARGET_COLUMN])
        df_temp[TARGET_COLUMN] = df_temp[TARGET_COLUMN].astype(int)

        missing_by_response = df_temp.groupby(TARGET_COLUMN).apply(lambda x: x.isnull().sum() / len(x) * 100).T
        missing_by_response = missing_by_response[missing_by_response.sum(axis=1) > 0].sort_values(by=1, ascending=False)
        
        print("\n--- Missing Percentage Split by Response (0=No Flow, 1=Flow) ---")
        print(missing_by_response.round(2))

        # Visualization 2: Bar Plots for Split Analysis
        
        # Select features that are missing in at least one group
        common_missing_features = missing_by_db_pct.index.union(missing_by_response.index)
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # Plot 1: Split by Database (Absolute Counts)
        missing_by_db_plot = df_raw[common_missing_features.tolist()].isnull().groupby(df_raw['Database']).sum().T
        missing_by_db_plot.plot(kind='bar', ax=axes[0], color=['#5A8D8B', '#F7B538'])
        axes[0].set_title('Missing Count by Database Type', fontsize=14)
        axes[0].set_ylabel('Count of Missing Values')
        axes[0].set_xlabel('Feature')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].legend(title='Database', loc='upper right')

        # Plot 2: Split by Response (Absolute Counts)
        missing_by_response_plot = df_temp[common_missing_features.tolist()].isnull().groupby(df_temp[TARGET_COLUMN]).sum().T
        missing_by_response_plot.plot(kind='bar', ax=axes[1], color=['#7A9A84', '#DC7543'])
        axes[1].set_title('Missing Count by Debris Flow Response', fontsize=14)
        axes[1].set_ylabel('Count of Missing Values')
        axes[1].set_xlabel('Feature')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].legend(title='Response', labels=['No Flow (0)', 'Flow (1)'], loc='upper right')

        plt.suptitle("Analysis of Missing Data Patterns (Absolute Counts)", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(OUTPUT_IMAGE_MDA_SPLIT, dpi=300)
        plt.close(fig)
        print(f"Saved Missing Data Split Analysis to: {OUTPUT_IMAGE_MDA_SPLIT}")
    else:
        print("Cannot perform split analysis: 'Database' or 'Response' column not found.")

def visualize_feature_ranges(df_raw):
    """
    Analyzes and visualizes the minimum and maximum ranges of quantitative features,
    split by their unit of measurement for meaningful comparison.
    """
    print("\n" + "="*70)
    print("--- 2. Feature Range Visualization (Split by Unit) ---")
    print("="*70)

    # --- Feature Groupings by Unit ---
    # Based on the description file
    unit_groups = {
        'Distance (m)': [
            'GaugeDist_m'
        ],
        'Area (km^2)': [
            'ContributingArea_km2'
        ],
        'Duration (H)': [
            'StormDur_H'
        ],
        'Accumulation (mm)': [
            'StormAccum_mm', 'Acc015_mm', 'Acc030_mm', 'Acc060_mm'
        ],
        'Intensity (mm/h)': [
            'StormAvgI_mm/h', 'Peak_I15_mm/h', 'Peak_I30_mm/h', 'Peak_I60_mm/h'
        ],
        'Indices (Proportion/Unitless)': [
            'PropHM23', 'dNBR/1000', 'KF'
        ]
    }
    
    # Identify all columns present in the DataFrame
    present_cols = set(df_raw.select_dtypes(include=np.number).columns)
    
    # Filter groups to only include features present in the raw data
    filtered_groups = {}
    for group_name, features in unit_groups.items():
        # Only keep features that are present in the DataFrame
        present_features = [f for f in features if f in present_cols]
        if present_features:
            filtered_groups[group_name] = present_features

    def create_range_plot(features, group_name):
        """Helper function to create and save a single range plot."""
        if not features: return

        range_data = df_raw[features].agg(['min', 'max', 'mean']).T
        # Sort features by mean value for a visually ordered plot
        range_data.sort_values(by='mean', ascending=False, inplace=True)
        
        print(f"\nFeature Descriptive Statistics ({group_name}):")
        print(range_data[['min', 'max', 'mean']].to_string(float_format="%.4f"))

        # Define scaling based on unit group
        if group_name in ['Coordinates (m)', 'Distance (m)', 'Area (km^2)']:
            use_log_scale = False # Linear scale for large, absolute units
            scale_type = 'Linear Scale'
        elif group_name in ['Indices (Proportion/Unitless)', 'Accumulation (mm)', 'Intensity (mm/h)']:
            use_log_scale = False # Log scale for often smaller, rate/proportional units
            scale_type = 'Linear Scale'
        else:
            use_log_scale = False
            scale_type = 'Linear Scale'


        # Dynamic height based on the number of features in the plot
        fig_height = max(5, 0.5 * len(features) + 1.5) 
        fig, ax = plt.subplots(figsize=(10, fig_height))
        
        y_pos = np.arange(len(range_data.index))
        
        # Plotting the range lines (Min to Max)
        ax.hlines(y=y_pos, 
                  xmin=range_data['min'], 
                  xmax=range_data['max'], 
                  color='#5A8D8B', 
                  linewidth=4,
                  label='Data Range (Min to Max)')
        
        # Plotting the mean markers
        ax.plot(range_data['mean'], y_pos, 
                'o', 
                color='#DC7543', 
                markersize=7, 
                label='Mean Value',
                markeredgecolor='white',
                markeredgewidth=1)
        
        # Set labels and title
        ax.set_yticks(y_pos)
        ax.set_yticklabels(range_data.index)
        ax.set_title(f'Quantitative Feature Ranges - Grouped by {group_name}', fontsize=16)
        
        # Apply scaling
        if use_log_scale and (range_data['min'] >= 0).all():
            ax.set_xscale('log')
            # Set left limit slightly below min value for non-zero data
            min_val = range_data['min'].replace(0, np.nan).min()
            # If all min values are zero or NaN, set a default small starting point
            if pd.isna(min_val) or min_val <= 0:
                 ax.set_xlim(left=1e-4, right=range_data['max'].max() * 1.5)
            else:
                 ax.set_xlim(left=min_val * 0.5, right=range_data['max'].max() * 1.5)

            ax.set_xlabel(f'Value ({scale_type}) - Units: {group_name}')
        else:
             ax.set_xlabel(f'Value ({scale_type}) - Units: {group_name}')
        
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        ax.legend(loc='lower right')
        plt.tight_layout()
        
        # Save file with a sanitized name
        safe_group_name = group_name.replace(' ', '_').replace('/', '-').replace('(', '').replace(')', '').replace('^', '')
        file_path = f"{OUTPUT_IMAGE_RANGES_PREFIX}{safe_group_name}.png"
        
        plt.savefig(file_path, dpi=300)
        plt.close()
        print(f"Saved Feature Range Visualization for '{group_name}' to: {file_path}")

    # Generate plots for each unit group
    for group_name, features in filtered_groups.items():
        create_range_plot(features, group_name)

def find_duplicate_events(df_raw):
    """
    Identifies and returns all rows that are duplicates based on the critical
    Event definition: Location (UTM_X, UTM_Y) + Storm Date (StormDate).
    """
    event_cols = ['UTM_X', 'UTM_Y', 'StormDate', 'StormStart', 'StormEnd']
    
    # 1. Identify which rows are duplicates
    # keep=False marks ALL occurrences of a duplicate set as True (including the first one)
    duplicate_mask = df_raw.duplicated(subset=event_cols, keep=False)
    
    # 2. Filter the original DataFrame using the mask
    duplicate_events_df = df_raw[duplicate_mask].sort_values(by=event_cols)
    
    return duplicate_events_df

def analyze_duplicates(df_raw):
    """
    Analyzes and visualizes the presence of duplicate and highly similar records, 
    focusing on location and storm event identifiers.
    """
    print("\n" + "="*70)
    print("--- 3. Duplicate and Repeat Value Analysis ---")
    print("="*70)

    total_rows = len(df_raw)
    
    # 1. Strict Uniqueness (All Columns)
    unique_rows_all_cols = len(df_raw.drop_duplicates(keep='first'))
    
    # 2. Location Uniqueness
    location_cols = ['UTM_X', 'UTM_Y']
    # Drop NaNs in ID columns before counting unique, as NaN is treated as unique unless explicitly dropped
    df_loc_subset = df_raw[location_cols]
    unique_locations = len(df_loc_subset.drop_duplicates(keep='first'))
    
    # 3. Storm Uniqueness (Date and Time stamps)
    storm_cols = ['StormDate', 'StormStart', 'StormEnd']
    df_storm_subset = df_raw[storm_cols]
    unique_storms = len(df_storm_subset.drop_duplicates(keep='first'))
    
    # 4. Event Uniqueness (Location + Storm) - The most critical check
    event_cols = location_cols + storm_cols # StormDate is usually sufficient to separate events
    df_event_subset = df_raw[event_cols]
    unique_events = len(df_event_subset.drop_duplicates(keep='first'))
    
    print(f"Total Records: {total_rows}")
    print(f"Unique Records (All Columns): {unique_rows_all_cols}")
    print(f"Unique Locations (UTM_X, UTM_Y): {unique_locations}")
    print(f"Unique Storms (Date, Start, End): {unique_storms}")
    print(f"Unique Events (Location + Date): {unique_events}")
    
    # Example Usage:
    duplicate_records = find_duplicate_events(df_raw)

    print("\n" + "="*70)
    print("--- Non-Unique (Duplicate) Event Records ---")
    print("="*70)
    if not duplicate_records.empty:
        print(f"Total non-unique event records found: {len(duplicate_records)}")
        # Display the first few duplicate groups for inspection
        print(duplicate_records.head(10)) 
    else:
        print("No non-unique event records found based on Location + Date.")

    # Visualization: Bar Chart
    
    # Create the data for the plot
    plot_data = pd.DataFrame({
        'Category': [
            'Total Records', 
            'Unique Observations (All Cols)', 
            'Unique Locations (UTM)', 
            'Unique Storms (Time)',
            'Unique Events (Loc + Date)'
        ],
        'Count': [
            total_rows, 
            unique_rows_all_cols, 
            unique_locations, 
            unique_storms,
            unique_events
        ]
    })
    
    plt.figure(figsize=(11, 7))
    # Use a sequential color palette to show the decreasing uniqueness
    ax = sns.barplot(x='Category', y='Count', data=plot_data, 
                     palette=['#5A8D8B', '#7A9A84', '#A0AF99', '#C3C4C3', '#E0E0E0'])
    
    # Add labels on bars
    for p in ax.patches:
        ax.annotate(format(p.get_height(), ',.0f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontsize=10)

    plt.title('Analysis of Unique Records by Scope (Location and Storm)', fontsize=16)
    plt.ylabel('Number of Records')
    plt.xlabel('')
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_DUPLICATES_ANALYSIS, dpi=300)
    plt.close()
    
    print(f"Saved Duplicate Analysis Bar Chart to: {OUTPUT_IMAGE_DUPLICATES_ANALYSIS}")


def perform_feature_eda(df):
    """Performs exploratory data analysis focusing on feature influence on Response."""
    
    print("\n" + "="*70)
    print(f"--- 4. Feature Analysis (EDA) on Cleaned Data ---")
    print("="*70)
    
    print("\nNote: Analysis performed on data imputed with median for missing values.")

    response_counts = df[TARGET_COLUMN].value_counts(normalize=True) * 100
    print("\nTarget Variable Distribution:")
    print(f"No Debris Flow (0): {response_counts.get(0, 0):.2f}%")
    print(f"Debris Flow (1): {response_counts.get(1, 0):.2f}%")

    # Features to analyze: Rainfall, Watershed, and Terrain characteristics
    rainfall_features = [
        'StormAccum_mm', 'StormAvgI_mm/h', 'Peak_I15_mm/h', 
        'Peak_I30_mm/h', 'Peak_I60_mm/h'
    ]
    watershed_features = [
        'ContributingArea_km2', 'PropHM23', 'dNBR/1000', 'KF', 
        'GaugeDist_m'
    ]
    
    # --- Univariate Analysis (Mean Difference) ---
    print("\n--- Univariate Analysis: Mean Difference for Key Features ---")
    
    all_features = rainfall_features + watershed_features
    analysis_results = []

    for feature in all_features:
        if feature not in df.columns: continue
            
        mean_no_df = df[df[TARGET_COLUMN] == 0][feature].mean()
        mean_with_df = df[df[TARGET_COLUMN] == 1][feature].mean()
        
        mean_diff = mean_with_df - mean_no_df
        try:
            # Added a small epsilon to denominator to prevent division by zero in edge cases
            factor = mean_with_df / (mean_no_df + 1e-9)
        except (ZeroDivisionError, RuntimeWarning):
            factor = np.inf

        analysis_results.append({
            'Feature': feature,
            'Mean_No_DF': f"{mean_no_df:.3f}",
            'Mean_With_DF': f"{mean_with_df:.3f}",
            'Mean_Difference': f"{mean_diff:.3f}",
            'Factor_Increase': f"{factor:.2f}x" if factor != np.inf else 'Inf'
        })
        
    results_df = pd.DataFrame(analysis_results).set_index('Feature')
    # Sort by factor increase to highlight the most influential features
    print(results_df.sort_values(by='Factor_Increase', key=lambda x: x.str.replace('x', '').replace('Inf', '99999').astype(float, errors='ignore'), ascending=False))
    
    # --- Visual Analysis (Distribution Plots) ---
    print("\n--- Visual Analysis: Distribution Plots (Saved as image) ---")
    
    num_features = len(all_features)
    cols = 3
    rows = int(np.ceil(num_features / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for i, feature in enumerate(all_features):
        if feature in df.columns:
            sns.boxplot(
                x=TARGET_COLUMN, 
                y=feature, 
                data=df, 
                ax=axes[i],
                palette=["#7A9A84", "#DC7543"]
            )
            axes[i].set_title(f'Response vs. {feature}', fontsize=12)
            axes[i].set_xlabel("Debris Flow Response (0=No, 1=Yes)")
            axes[i].set_ylabel(feature)

    for j in range(num_features, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Feature Distributions Split by Debris Flow Response (Cleaned Data)", fontsize=16, y=0.98)
    
    plt.savefig(OUTPUT_IMAGE_FEATURE_EDA, dpi=300)
    print(f"Successfully saved feature distribution visualization to: {OUTPUT_IMAGE_FEATURE_EDA}")
    plt.close(fig)


if __name__ == '__main__':
    raw_data = load_raw_data(FILE_PATH)
    
    if raw_data is not None:
        # 1. Perform Missing Data Analysis (MDA) on the raw data
        perform_mda(raw_data)
        
        # 2. Visualize Feature Ranges (Now split by Unit)
        visualize_feature_ranges(raw_data)
        
        # 3. Perform Duplicate Analysis on the raw data
        analyze_duplicates(raw_data)
        
        # 4. Clean the data for the main Feature EDA
        cleaned_data = clean_data(raw_data)
        
        # 5. Perform the original Feature EDA on the cleaned data
        if cleaned_data is not None and not cleaned_data.empty:
            perform_feature_eda(cleaned_data)

    print("\nAnalysis script finished.")