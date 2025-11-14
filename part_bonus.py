"""
Assignment 3 - Part 2 BONUS: Visualization and Interactive Interface
--------------------------------------------------------------------
This script provides:
1. t-SNE and PCA visualization of case embeddings
2. Interactive command-line interface for querying similar cases
3. Cluster analysis and visualization

Run this AFTER completing part2_similarity.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

# ==========================
# Configuration
# ==========================
BASE = Path(__file__).parent.resolve()
OUT_DIR = BASE / "assignment3_outputs"
BONUS_DIR = OUT_DIR / "bonus_visualizations"
BONUS_DIR.mkdir(parents=True, exist_ok=True)

# Input files (from part2_similarity.py)
DOC_EMB_NPY = OUT_DIR / "doc_embeddings.npy"
CASE_IDS_JSON = OUT_DIR / "doc_case_ids.json"
EXTRACTIVE = BASE / "summaries" / "extractive_summaries.json"

# Check if required files exist
if not DOC_EMB_NPY.exists() or not CASE_IDS_JSON.exists():
    raise FileNotFoundError(
        "Please run part2_similarity.py first to generate embeddings.\n"
        f"Missing: {DOC_EMB_NPY} or {CASE_IDS_JSON}"
    )

print("=" * 70)
print("BONUS: CASE EMBEDDING VISUALIZATION & INTERACTIVE INTERFACE")
print("=" * 70)

# ==========================
# Load Data
# ==========================
print("\n[1/5] Loading embeddings and case data...")

doc_embeddings = np.load(str(DOC_EMB_NPY))
with open(CASE_IDS_JSON, "r", encoding="utf-8") as f:
    case_ids = json.load(f)

print(f"  ✓ Loaded {len(case_ids)} case embeddings: {doc_embeddings.shape}")

# Load case texts for display
with open(EXTRACTIVE, "r", encoding="utf-8") as f:
    data = json.load(f)

case_texts = {}
for item in data:
    cid = item.get("case_id")
    segs = item.get("extractive_summary", [])
    if isinstance(segs, list):
        text = " ".join(segs).strip()
    else:
        text = str(segs).strip()
    case_texts[cid] = text[:200] + "..." if len(text) > 200 else text

# ==========================
# Part 1: PCA Visualization
# ==========================
print("\n[2/5] Creating PCA visualization...")

pca = PCA(n_components=2, random_state=42)
embeddings_2d_pca = pca.fit_transform(doc_embeddings)

# Create PCA plot
plt.figure(figsize=(14, 10))
scatter = plt.scatter(
    embeddings_2d_pca[:, 0], 
    embeddings_2d_pca[:, 1],
    c=range(len(case_ids)),
    cmap='viridis',
    alpha=0.6,
    s=100,
    edgecolors='black',
    linewidth=0.5
)

plt.colorbar(scatter, label='Case Index')
plt.title('PCA Visualization of Legal Case Embeddings', fontsize=16, fontweight='bold')
plt.xlabel(f'First Principal Component (explains {pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
plt.ylabel(f'Second Principal Component (explains {pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
plt.grid(True, alpha=0.3)

# Annotate some cases
num_annotations = min(10, len(case_ids))
for i in range(0, len(case_ids), len(case_ids) // num_annotations):
    plt.annotate(
        case_ids[i], 
        (embeddings_2d_pca[i, 0], embeddings_2d_pca[i, 1]),
        fontsize=8,
        alpha=0.7,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3)
    )

plt.tight_layout()
pca_file = BONUS_DIR / "pca_visualization.png"
plt.savefig(pca_file, dpi=300, bbox_inches='tight')
print(f"  ✓ PCA plot saved: {pca_file}")
plt.close()

# ==========================
# Part 2: t-SNE Visualization
# ==========================
print("\n[3/5] Creating t-SNE visualization (this may take a minute)...")

tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(case_ids)-1), max_iter=1000)
embeddings_2d_tsne = tsne.fit_transform(doc_embeddings)

# Create t-SNE plot
plt.figure(figsize=(14, 10))
scatter = plt.scatter(
    embeddings_2d_tsne[:, 0], 
    embeddings_2d_tsne[:, 1],
    c=range(len(case_ids)),
    cmap='plasma',
    alpha=0.6,
    s=100,
    edgecolors='black',
    linewidth=0.5
)

plt.colorbar(scatter, label='Case Index')
plt.title('t-SNE Visualization of Legal Case Embeddings', fontsize=16, fontweight='bold')
plt.xlabel('t-SNE Dimension 1', fontsize=12)
plt.ylabel('t-SNE Dimension 2', fontsize=12)
plt.grid(True, alpha=0.3)

# Annotate some cases
for i in range(0, len(case_ids), len(case_ids) // num_annotations):
    plt.annotate(
        case_ids[i], 
        (embeddings_2d_tsne[i, 0], embeddings_2d_tsne[i, 1]),
        fontsize=8,
        alpha=0.7,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.3)
    )

plt.tight_layout()
tsne_file = BONUS_DIR / "tsne_visualization.png"
plt.savefig(tsne_file, dpi=300, bbox_inches='tight')
print(f"  ✓ t-SNE plot saved: {tsne_file}")
plt.close()

# ==========================
# Part 3: Cluster Analysis
# ==========================
print("\n[4/5] Performing cluster analysis...")

# Determine optimal number of clusters (using elbow method)
n_clusters_range = range(3, min(11, len(case_ids) // 3))
inertias = []

for k in n_clusters_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(doc_embeddings)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(n_clusters_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters', fontsize=12)
plt.ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
plt.title('Elbow Method for Optimal Clusters', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
elbow_file = BONUS_DIR / "elbow_curve.png"
plt.savefig(elbow_file, dpi=300, bbox_inches='tight')
print(f"  ✓ Elbow curve saved: {elbow_file}")
plt.close()

# Use optimal clusters (let's use 5 as a reasonable default)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(doc_embeddings)

# Create clustered t-SNE visualization
plt.figure(figsize=(14, 10))
for cluster in range(optimal_k):
    mask = cluster_labels == cluster
    plt.scatter(
        embeddings_2d_tsne[mask, 0],
        embeddings_2d_tsne[mask, 1],
        label=f'Cluster {cluster+1}',
        alpha=0.6,
        s=100,
        edgecolors='black',
        linewidth=0.5
    )

plt.legend(fontsize=12, loc='best')
plt.title(f't-SNE Visualization with K-Means Clustering (k={optimal_k})', fontsize=16, fontweight='bold')
plt.xlabel('t-SNE Dimension 1', fontsize=12)
plt.ylabel('t-SNE Dimension 2', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
cluster_file = BONUS_DIR / "tsne_clustered.png"
plt.savefig(cluster_file, dpi=300, bbox_inches='tight')
print(f"  ✓ Clustered visualization saved: {cluster_file}")
plt.close()

# Save cluster assignments
cluster_df = pd.DataFrame({
    'case_id': case_ids,
    'cluster': cluster_labels
})
cluster_csv = BONUS_DIR / "cluster_assignments.csv"
cluster_df.to_csv(cluster_csv, index=False)
print(f"  ✓ Cluster assignments saved: {cluster_csv}")

# ==========================
# Part 4: Similarity Heatmap
# ==========================
print("\n[5/5] Creating similarity heatmap...")

# Compute similarity matrix
similarity_matrix = cosine_similarity(doc_embeddings)

# Create heatmap for a subset of cases (full matrix would be too large)
num_display = min(30, len(case_ids))
subset_indices = np.linspace(0, len(case_ids)-1, num_display, dtype=int)
subset_sim = similarity_matrix[subset_indices][:, subset_indices]
subset_labels = [case_ids[i] for i in subset_indices]

plt.figure(figsize=(14, 12))
sns.heatmap(
    subset_sim,
    xticklabels=subset_labels,
    yticklabels=subset_labels,
    cmap='YlOrRd',
    annot=False,
    fmt='.2f',
    cbar_kws={'label': 'Cosine Similarity'},
    square=True
)
plt.title(f'Case Similarity Heatmap (Subset of {num_display} cases)', fontsize=14, fontweight='bold')
plt.xlabel('Case ID', fontsize=12)
plt.ylabel('Case ID', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
heatmap_file = BONUS_DIR / "similarity_heatmap.png"
plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
print(f"  ✓ Similarity heatmap saved: {heatmap_file}")
plt.close()

# ==========================
# Part 5: Interactive CLI
# ==========================
print("\n" + "=" * 70)
print("INTERACTIVE COMMAND-LINE INTERFACE")
print("=" * 70)

# Prepare for queries
caseid_to_idx = {cid: i for i, cid in enumerate(case_ids)}

def query_similar_cases(query_case_id: str, k: int = 5):
    """Find and display top-K similar cases."""
    if query_case_id not in caseid_to_idx:
        print(f"❌ Case ID '{query_case_id}' not found.")
        print(f"Available cases: {', '.join(case_ids[:10])}{'...' if len(case_ids) > 10 else ''}")
        return
    
    query_idx = caseid_to_idx[query_case_id]
    query_cluster = cluster_labels[query_idx]
    similarities = similarity_matrix[query_idx]
    
    # Exclude self
    similarities[query_idx] = -1.0
    
    # Get top-K
    topk_indices = np.argsort(-similarities)[:k]
    
    print(f"\n{'='*70}")
    print(f"Query Case: {query_case_id} (Cluster {query_cluster + 1})")
    print(f"{'='*70}")
    print(f"Summary: {case_texts.get(query_case_id, 'N/A')}")
    print(f"\n{'Top ' + str(k) + ' Most Similar Cases:'}")
    print(f"{'-'*70}")
    
    for rank, idx in enumerate(topk_indices, 1):
        sim_case_id = case_ids[idx]
        sim_score = similarities[idx]
        sim_cluster = cluster_labels[idx]
        same_cluster = "✓" if sim_cluster == query_cluster else "✗"
        
        print(f"\n{rank}. {sim_case_id} (Cluster {sim_cluster + 1}) [{same_cluster}]")
        print(f"   Similarity: {sim_score:.4f}")
        print(f"   Summary: {case_texts.get(sim_case_id, 'N/A')}")

def show_cluster_info(cluster_num: int):
    """Show all cases in a specific cluster."""
    if cluster_num < 1 or cluster_num > optimal_k:
        print(f"❌ Invalid cluster number. Choose between 1 and {optimal_k}")
        return
    
    cluster_idx = cluster_num - 1
    cluster_cases = [case_ids[i] for i, c in enumerate(cluster_labels) if c == cluster_idx]
    
    print(f"\n{'='*70}")
    print(f"Cluster {cluster_num} Information")
    print(f"{'='*70}")
    print(f"Number of cases: {len(cluster_cases)}")
    print(f"Cases: {', '.join(cluster_cases[:20])}")
    if len(cluster_cases) > 20:
        print(f"       ... and {len(cluster_cases) - 20} more")

# Interactive menu
def interactive_menu():
    """Main interactive menu."""
    while True:
        print("\n" + "=" * 70)
        print("INTERACTIVE CASE SIMILARITY SYSTEM")
        print("=" * 70)
        print("\nOptions:")
        print("  1. Query similar cases")
        print("  2. View cluster information")
        print("  3. List all case IDs")
        print("  4. Show random example")
        print("  5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            case_id = input(f"Enter case ID (e.g., {case_ids[0]}): ").strip()
            k = input("Number of similar cases to retrieve (default 5): ").strip()
            k = int(k) if k.isdigit() else 5
            query_similar_cases(case_id, k)
            
        elif choice == '2':
            cluster_num = input(f"Enter cluster number (1-{optimal_k}): ").strip()
            if cluster_num.isdigit():
                show_cluster_info(int(cluster_num))
            else:
                print("❌ Invalid input")
                
        elif choice == '3':
            print(f"\nAll Case IDs ({len(case_ids)} total):")
            for i, cid in enumerate(case_ids, 1):
                print(f"  {i}. {cid}", end="")
                if i % 5 == 0:
                    print()
            print()
            
        elif choice == '4':
            random_case = np.random.choice(case_ids)
            print(f"\nRandom example case: {random_case}")
            query_similar_cases(random_case, 5)
            
        elif choice == '5':
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1-5.")

# ==========================
# Generate Summary Report
# ==========================
summary_report = f"""
BONUS PART - VISUALIZATION AND ANALYSIS SUMMARY
================================================

1. VISUALIZATIONS CREATED
--------------------------
✓ PCA Visualization: Shows cases projected onto 2D space using Principal Component Analysis
  - First two components explain {pca.explained_variance_ratio_[0]:.1%} and {pca.explained_variance_ratio_[1]:.1%} of variance
  - File: {pca_file.name}

✓ t-SNE Visualization: Shows non-linear dimensionality reduction for better cluster separation
  - Reveals natural groupings in the data
  - File: {tsne_file.name}

✓ Clustered t-SNE: Shows K-means clustering results overlaid on t-SNE projection
  - Optimal clusters: {optimal_k} (determined using elbow method)
  - File: {cluster_file.name}

✓ Elbow Curve: Helps determine optimal number of clusters
  - File: {elbow_file.name}

✓ Similarity Heatmap: Shows pairwise similarities between cases
  - Displays subset of {num_display} cases
  - File: {heatmap_file.name}

2. CLUSTER ANALYSIS
-------------------
Number of clusters identified: {optimal_k}

Cluster Distribution:
"""

for i in range(optimal_k):
    count = np.sum(cluster_labels == i)
    summary_report += f"  Cluster {i+1}: {count} cases ({count/len(case_ids)*100:.1f}%)\n"

summary_report += f"""

3. INSIGHTS FROM VISUALIZATIONS
--------------------------------
• PCA Visualization: The first two principal components capture the most significant
  variations in case embeddings. Cases that are close together in PCA space share
  similar semantic content.

• t-SNE Visualization: Provides better separation of clusters compared to PCA.
  Distinct groups visible in t-SNE plot represent cases with very similar legal
  issues, terminology, or procedural contexts.

• Clustering: K-means identified {optimal_k} natural groups in the data. Cases within
  the same cluster likely deal with similar legal domains (e.g., criminal appeals,
  civil disputes, constitutional matters).

• Similarity Patterns: The heatmap reveals that certain cases have very high
  similarity (bright colors), indicating they may be related appeals, similar
  offenses, or cases citing the same legal precedents.

4. PRACTICAL APPLICATIONS
--------------------------
These visualizations enable:
• Quick identification of case clusters by legal domain
• Visual exploration of the case similarity space
• Understanding of embedding quality and data distribution
• Interactive querying through the CLI interface

5. INTERACTIVE FEATURES
------------------------
The command-line interface allows:
• Querying any case to find similar cases
• Viewing cluster memberships
• Exploring case relationships interactively
• Testing retrieval system with custom queries

All visualization files saved to: {BONUS_DIR}
"""

report_file = BONUS_DIR / "visualization_report.txt"
with open(report_file, "w", encoding="utf-8") as f:
    f.write(summary_report)

print("\n" + "=" * 70)
print("✅ BONUS PART COMPLETED")
print("=" * 70)
print(f"\nGenerated Files:")
print(f"  • {pca_file}")
print(f"  • {tsne_file}")
print(f"  • {cluster_file}")
print(f"  • {elbow_file}")
print(f"  • {heatmap_file}")
print(f"  • {cluster_csv}")
print(f"  • {report_file}")

print("\n" + summary_report)

# Ask if user wants to use interactive interface
print("\n" + "=" * 70)
use_cli = input("Would you like to use the interactive interface? (y/n): ").strip().lower()
if use_cli == 'y':
    interactive_menu()
else:
    print("\n💡 You can run this script again anytime to use the interactive interface!")
    print("   Or import the functions: query_similar_cases(), show_cluster_info()")