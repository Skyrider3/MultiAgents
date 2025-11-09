#!/usr/bin/env python3
"""
Download LaTeX source files from arXiv for multiple math categories (using list)
and organize them into category-specific folders.
"""

import arxiv
import pandas as pd
from pathlib import Path
import tarfile
import shutil
import time

# Configuration
TOPIC_TABLE_PATH = Path("/Users/CP/Documents/MathMind/papers/topicTable.txt")
BASE_PAPERS_DIR = Path("/Users/CP/Documents/MathMind/papers")
MAX_RESULTS_PER_CATEGORY = 20  # Adjust as needed
DELAY_BETWEEN_CATEGORIES = 2  # Seconds to wait between categories to be polite to arXiv

def read_categories(topic_file):
    """Read category codes and names from topicTable.txt"""
    categories = []
    with open(topic_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',', 1)
                if len(parts) == 2:
                    code = parts[0].strip()
                    name = parts[1].strip()
                    categories.append((code, name))
    return categories

def ensure_directory(path):
    """Create directory if it doesn't exist"""
    path.mkdir(parents=True, exist_ok=True)
    return path

def extract_tarball(tar_path, extract_to):
    """Extract a tar.gz file to a specific directory"""
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=extract_to)
        return True
    except Exception as e:
        print(f"    ✗ Failed to extract: {e}")
        return False

def download_category(category_code, category_name, output_dir, max_results=20):
    """Download papers for a specific category"""
    
    # Create output directory
    latex_dir = output_dir / "latex_sources"
    ensure_directory(latex_dir)
    
    # Construct arXiv query
    arxiv_category = f"cat:math.{category_code}"
    
    print(f"\n{'='*70}")
    print(f"Category: {category_name} (math.{category_code})")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")
    
    # Search arXiv
    search = arxiv.Search(
        query=arxiv_category,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    all_data = []
    downloaded = 0
    failed = 0
    extracted = 0
    
    try:
        for i, result in enumerate(search.results(), 1):
            # Extract arxiv ID
            arxiv_id = result.entry_id.split('/')[-1]
            
            # Save metadata
            all_data.append({
                "Title": result.title,
                "Date": result.published,
                "Id": result.entry_id,
                "Summary": result.summary,
                "URL": result.pdf_url,
                "ArXiv_ID": arxiv_id,
                "Category": category_code
            })
            
            # Download the LaTeX source
            try:
                print(f"[{i:2d}] {result.title[:55]}...")
                tar_path = latex_dir / f"{arxiv_id}.tar.gz"
                extract_dir = latex_dir / arxiv_id
                
                # Check if already downloaded
                if tar_path.exists():
                    print(f"     ✓ Archive exists: {tar_path.name}")
                    downloaded += 1
                else:
                    # Download the source
                    result.download_source(dirpath=str(latex_dir), filename=f"{arxiv_id}.tar.gz")
                    print(f"     ✓ Downloaded: {tar_path.name}")
                    downloaded += 1
                    time.sleep(0.5)  # Be polite to arXiv servers
                
                # Extract if not already extracted
                if not extract_dir.exists():
                    ensure_directory(extract_dir)
                    if extract_tarball(tar_path, extract_dir):
                        extracted += 1
                else:
                    extracted += 1
                    
            except Exception as e:
                print(f"     ✗ Failed: {e}")
                failed += 1
                
    except Exception as e:
        print(f"\n✗ Error processing category {category_code}: {e}")
    
    # Save metadata
    if all_data:
        df = pd.DataFrame(all_data)
        csv_path = output_dir / f"{category_code}_metadata.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n📊 Metadata saved: {csv_path.name}")
    
    # Summary
    print(f"\n📈 Summary for {category_code}:")
    print(f"   Papers found: {len(all_data)}")
    print(f"   Downloaded: {downloaded}")
    print(f"   Extracted: {extracted}")
    print(f"   Failed: {failed}")
    
    return len(all_data), downloaded, failed

def main():
    """Main function to process all categories"""
    
    print("="*70)
    print("arXiv Multi-Category LaTeX Downloader")
    print("="*70)
    
    # Read categories
    print(f"\n📖 Reading categories from: {TOPIC_TABLE_PATH}")
    categories = read_categories(TOPIC_TABLE_PATH)
    print(f"   Found {len(categories)} categories")
    
    # Ensure base directory exists
    ensure_directory(BASE_PAPERS_DIR)
    
    # Process each category
    total_papers = 0
    total_downloaded = 0
    total_failed = 0
    
    for i, (code, name) in enumerate(categories, 1):
        # Create category directory
        category_dir = BASE_PAPERS_DIR / name
        ensure_directory(category_dir)
        
        # Download papers
        papers, downloaded, failed = download_category(
            code, name, category_dir, MAX_RESULTS_PER_CATEGORY
        )
        
        total_papers += papers
        total_downloaded += downloaded
        total_failed += failed
        
        # Wait between categories (except after last one)
        if i < len(categories):
            print(f"\n⏳ Waiting {DELAY_BETWEEN_CATEGORIES}s before next category...")
            time.sleep(DELAY_BETWEEN_CATEGORIES)
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 ALL CATEGORIES COMPLETED!")
    print("="*70)
    print(f"Categories processed: {len(categories)}")
    print(f"Total papers: {total_papers}")
    print(f"Successfully downloaded: {total_downloaded}")
    print(f"Failed: {total_failed}")
    print(f"Base directory: {BASE_PAPERS_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()

