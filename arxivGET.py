# First, install the arxiv library:
# pip install arxiv
# or for Poetry projects: poetry add arxiv

import arxiv
import pandas as pd
from pathlib import Path

# Create directory for downloaded papers (LaTeX source)
DOWNLOAD_DIR = Path("papers_latex")
DOWNLOAD_DIR.mkdir(exist_ok=True)

# Search for algebraic geometry papers (math.AG is the arXiv subject code)
search = arxiv.Search(
    query="cat:math.AG",
    max_results=20,  # Adjust the number of results as needed
    sort_by=arxiv.SortCriterion.SubmittedDate
)

print(f"Downloading LaTeX source files to: {DOWNLOAD_DIR.absolute()}")
print("(Source files are downloaded as .tar.gz archives)")
print("-" * 60)

all_data = []
downloaded = 0
failed = 0

for i, result in enumerate(search.results(), 1):
    # Extract arxiv ID from entry_id (e.g., http://arxiv.org/abs/2511.04625v1 -> 2511.04625v1)
    arxiv_id = result.entry_id.split('/')[-1]
    
    # Save metadata
    all_data.append([
        result.title,
        result.published,
        result.entry_id,
        result.summary,
        result.pdf_url,
        arxiv_id
    ])
    
    # Download the LaTeX source
    try:
        print(f"[{i}] Downloading: {result.title[:60]}...")
        source_path = DOWNLOAD_DIR / f"{arxiv_id}.tar.gz"
        
        # Check if already downloaded
        if source_path.exists():
            print(f"    ✓ Already exists: {source_path.name}")
            downloaded += 1
        else:
            # Download the LaTeX source
            result.download_source(dirpath=str(DOWNLOAD_DIR), filename=f"{arxiv_id}.tar.gz")
            print(f"    ✓ Downloaded: {source_path.name}")
            downloaded += 1
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        failed += 1

print("\n" + "=" * 60)
print(f"Download Summary:")
print(f"  Total papers: {len(all_data)}")
print(f"  Successfully downloaded: {downloaded}")
print(f"  Failed: {failed}")
print(f"  Location: {DOWNLOAD_DIR.absolute()}")
print("=" * 60)

# Save metadata to CSV for reference
column_names = ["Title", "Date", "Id", "Summary", "URL", "ArXiv_ID"]
df = pd.DataFrame(all_data, columns=column_names)
csv_path = DOWNLOAD_DIR / "latex_sources_metadata.csv"
df.to_csv(csv_path, index=False)
print(f"\nMetadata saved to: {csv_path}")
