"""
ArXiv Paper Fetcher - Download and metadata extraction from arXiv
"""

import asyncio
import aiohttp
import arxiv
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger
import xml.etree.ElementTree as ET
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings, get_ingestion_logger
from src.knowledge.graph.schema import PaperNode, AuthorNode


class ArxivFetcher:
    """
    Fetches papers from arXiv API with metadata extraction
    """

    def __init__(self, download_dir: Path = None):
        self.download_dir = download_dir or settings.papers_dir
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_ingestion_logger()
        self.client = arxiv.Client()

        # Rate limiting
        self.rate_limit_delay = 3.0  # seconds between requests
        self.last_request_time = 0.0

    async def search_papers(
        self,
        query: str,
        max_results: int = None,
        sort_by: arxiv.SortCriterion = None,
        sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending,
        categories: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for papers on arXiv

        Args:
            query: Search query (supports arXiv query syntax)
            max_results: Maximum number of results
            sort_by: Sorting criterion
            sort_order: Sort order
            categories: Filter by arXiv categories (e.g., ['math.NT', 'math.AG'])

        Returns:
            List of paper metadata dictionaries
        """
        max_results = max_results or settings.external.arxiv_max_results

        # Build query with category filters
        if categories:
            category_filter = " OR ".join([f"cat:{cat}" for cat in categories])
            if query:
                query = f"({query}) AND ({category_filter})"
            else:
                query = category_filter

        self.logger.info(f"Searching arXiv for: {query} (max {max_results} results)")

        # Default sort by relevance
        if not sort_by:
            sort_by = arxiv.SortCriterion.Relevance
            if settings.external.arxiv_sort_by == "submittedDate":
                sort_by = arxiv.SortCriterion.SubmittedDate
            elif settings.external.arxiv_sort_by == "lastUpdatedDate":
                sort_by = arxiv.SortCriterion.LastUpdatedDate

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order
        )

        papers = []
        try:
            results = self.client.results(search)

            for result in results:
                await self._rate_limit()
                paper_data = self._extract_paper_metadata(result)
                papers.append(paper_data)

                self.logger.debug(f"Found paper: {paper_data['title']}")

        except Exception as e:
            self.logger.error(f"Error searching arXiv: {e}")
            raise

        self.logger.info(f"Found {len(papers)} papers")
        return papers

    async def fetch_paper_by_id(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a specific paper by arXiv ID

        Args:
            arxiv_id: arXiv paper ID (e.g., "2301.12345" or "math.NT/0501234")

        Returns:
            Paper metadata dictionary or None if not found
        """
        self.logger.info(f"Fetching paper: {arxiv_id}")

        try:
            search = arxiv.Search(id_list=[arxiv_id])
            results = list(self.client.results(search))

            if results:
                paper_data = self._extract_paper_metadata(results[0])
                return paper_data
            else:
                self.logger.warning(f"Paper not found: {arxiv_id}")
                return None

        except Exception as e:
            self.logger.error(f"Error fetching paper {arxiv_id}: {e}")
            raise

    async def download_paper(
        self,
        paper_data: Dict[str, Any],
        download_pdf: bool = True,
        download_source: bool = False
    ) -> Dict[str, Path]:
        """
        Download paper PDF and/or source files

        Args:
            paper_data: Paper metadata dictionary
            download_pdf: Whether to download PDF
            download_source: Whether to download LaTeX source

        Returns:
            Dictionary with paths to downloaded files
        """
        arxiv_id = paper_data['arxiv_id']
        clean_id = arxiv_id.replace('/', '_')

        paths = {}

        # Download PDF
        if download_pdf:
            pdf_path = self.download_dir / f"{clean_id}.pdf"
            if not pdf_path.exists():
                self.logger.info(f"Downloading PDF: {arxiv_id}")

                try:
                    pdf_url = paper_data['pdf_url']
                    await self._download_file(pdf_url, pdf_path)
                    paths['pdf'] = pdf_path
                    self.logger.info(f"Downloaded PDF to: {pdf_path}")

                except Exception as e:
                    self.logger.error(f"Failed to download PDF for {arxiv_id}: {e}")
            else:
                self.logger.debug(f"PDF already exists: {pdf_path}")
                paths['pdf'] = pdf_path

        # Download source
        if download_source and paper_data.get('source_url'):
            source_path = self.download_dir / f"{clean_id}_source.tar.gz"
            if not source_path.exists():
                self.logger.info(f"Downloading source: {arxiv_id}")

                try:
                    source_url = paper_data['source_url']
                    await self._download_file(source_url, source_path)
                    paths['source'] = source_path
                    self.logger.info(f"Downloaded source to: {source_path}")

                except Exception as e:
                    self.logger.error(f"Failed to download source for {arxiv_id}: {e}")
            else:
                self.logger.debug(f"Source already exists: {source_path}")
                paths['source'] = source_path

        # Save metadata
        metadata_path = self.download_dir / f"{clean_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(paper_data, f, indent=2, default=str)
        paths['metadata'] = metadata_path

        return paths

    async def fetch_mathematical_papers(
        self,
        topics: List[str] = None,
        max_papers: int = 20,
        recent_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Fetch papers specifically for mathematical conjecture discovery

        Args:
            topics: List of mathematical topics
            max_papers: Maximum number of papers
            recent_only: Only fetch papers from last 2 years

        Returns:
            List of paper metadata with downloaded PDFs
        """
        if not topics:
            topics = [
                "Riemann Hypothesis",
                "Twin Prime Conjecture",
                "Goldbach Conjecture",
                "P vs NP",
                "Hodge Conjecture",
                "Birch Swinnerton-Dyer",
                "Navier-Stokes",
                "Yang-Mills"
            ]

        math_categories = [
            'math.NT',  # Number Theory
            'math.AG',  # Algebraic Geometry
            'math.CO',  # Combinatorics
            'math.GT',  # Geometric Topology
            'math.AT',  # Algebraic Topology
            'math.FA',  # Functional Analysis
            'math.CA',  # Classical Analysis
            'math.DG',  # Differential Geometry
        ]

        all_papers = []
        papers_per_topic = max(1, max_papers // len(topics))

        for topic in topics:
            self.logger.info(f"Fetching papers on: {topic}")

            # Build query
            query = f"ti:{topic} OR abs:{topic}"
            if recent_only:
                # Add date filter for papers from last 2 years
                query += f" AND submittedDate:[{datetime.now().year - 2}0101 TO {datetime.now().strftime('%Y%m%d')}]"

            # Search papers
            papers = await self.search_papers(
                query=query,
                max_results=papers_per_topic,
                categories=math_categories,
                sort_by=arxiv.SortCriterion.Relevance
            )

            # Download PDFs
            for paper in papers:
                try:
                    paths = await self.download_paper(paper, download_pdf=True)
                    paper['local_paths'] = paths
                    all_papers.append(paper)

                except Exception as e:
                    self.logger.error(f"Failed to process paper {paper['arxiv_id']}: {e}")

            # Rate limiting between topics
            await asyncio.sleep(self.rate_limit_delay)

        self.logger.info(f"Fetched {len(all_papers)} mathematical papers total")
        return all_papers

    async def fetch_citations(self, arxiv_id: str) -> List[str]:
        """
        Fetch citations for a paper (limited - arXiv doesn't provide full citations)

        Args:
            arxiv_id: arXiv paper ID

        Returns:
            List of cited paper IDs (if extractable)
        """
        # Note: arXiv API doesn't provide citations directly
        # This would need to be extracted from the paper content
        # or use external services like Semantic Scholar

        citations = []

        # Try to extract from paper references section
        paper = await self.fetch_paper_by_id(arxiv_id)
        if paper and paper.get('comment'):
            # Simple heuristic to find arXiv citations in comments
            import re
            arxiv_pattern = r'arXiv:(\d{4}\.\d{4,5})'
            matches = re.findall(arxiv_pattern, paper['comment'])
            citations.extend(matches)

        return citations

    def _extract_paper_metadata(self, result: arxiv.Result) -> Dict[str, Any]:
        """Extract metadata from arXiv result"""

        # Extract authors
        authors = []
        for author in result.authors:
            authors.append({
                'name': author.name,
                'affiliation': None  # arXiv doesn't provide affiliations in API
            })

        # Extract categories
        categories = [cat for cat in result.categories]

        # Determine primary domain
        primary_domain = 'mathematics'
        if categories:
            if categories[0].startswith('math.'):
                domain_map = {
                    'math.NT': 'number_theory',
                    'math.AG': 'algebraic_geometry',
                    'math.CO': 'combinatorics',
                    'math.GT': 'geometric_topology',
                    'math.AT': 'algebraic_topology',
                    'math.FA': 'functional_analysis',
                    'math.CA': 'classical_analysis',
                    'math.DG': 'differential_geometry',
                }
                primary_domain = domain_map.get(categories[0], 'mathematics')

        metadata = {
            'arxiv_id': result.entry_id.split('/')[-1],  # Extract ID from URL
            'title': result.title,
            'abstract': result.summary,
            'authors': authors,
            'published_date': result.published,
            'updated_date': result.updated,
            'categories': categories,
            'primary_category': result.primary_category,
            'domain': primary_domain,
            'comment': result.comment,
            'journal_ref': result.journal_ref,
            'doi': result.doi,
            'pdf_url': result.pdf_url,
            'source_url': result.entry_id.replace('/abs/', '/e-print/'),
            'arxiv_url': result.entry_id,
        }

        return metadata

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _download_file(self, url: str, path: Path):
        """Download file from URL with retry logic"""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()

                content = await response.read()
                with open(path, 'wb') as f:
                    f.write(content)

    async def _rate_limit(self):
        """Implement rate limiting to be respectful to arXiv"""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)

        self.last_request_time = asyncio.get_event_loop().time()

    async def get_paper_statistics(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about fetched papers

        Args:
            papers: List of paper metadata

        Returns:
            Statistics dictionary
        """
        stats = {
            'total_papers': len(papers),
            'domains': {},
            'years': {},
            'authors': set(),
            'categories': {},
            'avg_abstract_length': 0
        }

        total_abstract_length = 0

        for paper in papers:
            # Count by domain
            domain = paper.get('domain', 'unknown')
            stats['domains'][domain] = stats['domains'].get(domain, 0) + 1

            # Count by year
            if paper.get('published_date'):
                year = paper['published_date'].year
                stats['years'][year] = stats['years'].get(year, 0) + 1

            # Collect unique authors
            for author in paper.get('authors', []):
                stats['authors'].add(author['name'])

            # Count categories
            for category in paper.get('categories', []):
                stats['categories'][category] = stats['categories'].get(category, 0) + 1

            # Abstract length
            if paper.get('abstract'):
                total_abstract_length += len(paper['abstract'])

        stats['unique_authors'] = len(stats['authors'])
        stats['authors'] = list(stats['authors'])[:20]  # Keep only top 20 for display

        if papers:
            stats['avg_abstract_length'] = total_abstract_length / len(papers)

        return stats


# Specialized fetchers for different mathematical domains
class NumberTheoryFetcher(ArxivFetcher):
    """Specialized fetcher for number theory papers"""

    async def fetch_prime_papers(self, max_papers: int = 10) -> List[Dict[str, Any]]:
        """Fetch papers about prime numbers"""
        prime_topics = [
            "prime numbers",
            "Riemann Hypothesis",
            "prime distribution",
            "twin primes",
            "Goldbach conjecture",
            "prime gaps",
            "Dirichlet theorem"
        ]

        papers = []
        for topic in prime_topics:
            results = await self.search_papers(
                query=f"ti:\"{topic}\" OR abs:\"{topic}\"",
                max_results=max(1, max_papers // len(prime_topics)),
                categories=['math.NT']
            )
            papers.extend(results)

        return papers


class AlgebraicGeometryFetcher(ArxivFetcher):
    """Specialized fetcher for algebraic geometry papers"""

    async def fetch_hodge_papers(self, max_papers: int = 10) -> List[Dict[str, Any]]:
        """Fetch papers about Hodge theory and related conjectures"""
        hodge_topics = [
            "Hodge conjecture",
            "Hodge theory",
            "algebraic cycles",
            "Kahler manifolds",
            "period integrals"
        ]

        papers = []
        for topic in hodge_topics:
            results = await self.search_papers(
                query=f"ti:\"{topic}\" OR abs:\"{topic}\"",
                max_results=max(1, max_papers // len(hodge_topics)),
                categories=['math.AG', 'math.DG']
            )
            papers.extend(results)

        return papers


# Example usage
if __name__ == "__main__":
    async def main():
        fetcher = ArxivFetcher()

        # Fetch mathematical papers
        papers = await fetcher.fetch_mathematical_papers(
            topics=["Riemann Hypothesis", "P vs NP"],
            max_papers=5
        )

        # Print results
        for paper in papers:
            print(f"\nTitle: {paper['title']}")
            print(f"Authors: {', '.join([a['name'] for a in paper['authors']])}")
            print(f"Categories: {', '.join(paper['categories'])}")
            print(f"PDF: {paper['local_paths'].get('pdf', 'Not downloaded')}")

        # Get statistics
        stats = await fetcher.get_paper_statistics(papers)
        print(f"\nStatistics: {json.dumps(stats, indent=2, default=str)}")

    asyncio.run(main())