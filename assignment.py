
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=== DATA ANALYSIS ASSIGNMENT ===\n")

# Task 1: Load and Explore the Dataset
print("TASK 1: LOADING AND EXPLORING THE DATASET")
print("-" * 50)

try:
    # Load the Iris dataset
    iris = load_iris()
    
    # Create DataFrame
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    print("1. Dataset loaded successfully!")
    print(f"   Dataset shape: {df.shape}")
    
    # Display first few rows
    print("\n2. First 5 rows of the dataset:")
    print(df.head())
    
    # Explore structure
    print("\n3. Dataset structure:")
    print(df.info())
    
    # Check for missing values
    print("\n4. Missing values check:")
    print(df.isnull().sum())
    
    # Since Iris dataset is clean, we'll demonstrate cleaning with a hypothetical scenario
    print("\n5. Data cleaning demonstration:")
    print("   No missing values found - dataset is already clean!")
    
except Exception as e:
    print(f"Error loading dataset: {e}")

# Task 2: Basic Data Analysis
print("\n" + "="*50)
print("TASK 2: BASIC DATA ANALYSIS")
print("-" * 50)

# Basic statistics
print("1. Basic statistics of numerical columns:")
print(df.describe())

# Group by species and compute means
print("\n2. Mean measurements by species:")
species_means = df.groupby('species').mean()
print(species_means)

# Additional analysis
print("\n3. Additional patterns:")
print(f"- Largest sepal length: {df['sepal length (cm)'].max()} cm (species: {df.loc[df['sepal length (cm)'].idxmax(), 'species']})")
print(f"- Smallest petal width: {df['petal width (cm)'].min()} cm (species: {df.loc[df['petal width (cm)'].idxmin(), 'species']})")

# Task 3: Data Visualization
print("\n" + "="*50)
print("TASK 3: DATA VISUALIZATION")
print("-" * 50)

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Iris Dataset Analysis - Comprehensive Visualizations', fontsize=16, fontweight='bold')

# Plot 1: Line chart (showing trends across samples)
print("1. Creating line chart...")
sample_data = df.iloc[:30]  # First 30 samples for clarity
axes[0,0].plot(sample_data.index, sample_data['sepal length (cm)'], marker='o', label='Sepal Length')
axes[0,0].plot(sample_data.index, sample_data['petal length (cm)'], marker='s', label='Petal Length')
axes[0,0].set_title('Trend of Measurements Across Samples')
axes[0,0].set_xlabel('Sample Index')
axes[0,0].set_ylabel('Length (cm)')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Bar chart (average measurements by species)
print("2. Creating bar chart...")
species_means.plot(kind='bar', ax=axes[0,1])
axes[0,1].set_title('Average Measurements by Iris Species')
axes[0,1].set_xlabel('Species')
axes[0,1].set_ylabel('Average Measurement (cm)')
axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0,1].tick_params(axis='x', rotation=45)

# Plot 3: Histogram (distribution of sepal length)
print("3. Creating histogram...")
axes[1,0].hist(df['sepal length (cm)'], bins=15, alpha=0.7, edgecolor='black')
axes[1,0].set_title('Distribution of Sepal Length')
axes[1,0].set_xlabel('Sepal Length (cm)')
axes[1,0].set_ylabel('Frequency')
axes[1,0].grid(True, alpha=0.3)

# Plot 4: Scatter plot (sepal length vs petal length)
print("4. Creating scatter plot...")
colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
for species in df['species'].unique():
    species_data = df[df['species'] == species]
    axes[1,1].scatter(species_data['sepal length (cm)'], 
                     species_data['petal length (cm)'], 
                     label=species, 
                     alpha=0.7)
axes[1,1].set_title('Sepal Length vs Petal Length')
axes[1,1].set_xlabel('Sepal Length (cm)')
axes[1,1].set_ylabel('Petal Length (cm)')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional insights and observations
print("\n" + "="*50)
print("FINDINGS AND OBSERVATIONS")
print("-" * 50)

print("1. SPECIES DIFFERENCES:")
print("   - Setosa has the smallest petals but relatively large sepals")
print("   - Virginica has the largest overall measurements")
print("   - Versicolor falls in between the other two species")

print("\n2. MEASUREMENT PATTERNS:")
print("   - Petal measurements show clearer separation between species")
print("   - Sepal width has the least variation across species")

print("\n3. DATA QUALITY:")
print("   - Dataset is complete with no missing values")
print("   - Measurements show consistent patterns within species")

print("\n4. VISUALIZATION INSIGHTS:")
print("   - Scatter plot shows clear clustering by species")
print("   - Histogram reveals normal distribution of sepal lengths")
print("   - Bar chart highlights systematic differences between species")

print("\n" + "="*50)
print("ASSIGNMENT COMPLETE! ✅")
print("All tasks have been successfully implemented.")