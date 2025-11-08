"""
Document Ingestion Pipeline - Coordinates fetching, parsing, and storage
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from loguru import logger

from src.config import settings, get_ingestion_logger, log_timing
from src.knowledge.ingestion.arxiv_fetcher import ArxivFetcher
from src.knowledge.ingestion.pdf_parser import PDFParser, ExtractedContent
from src.knowledge.ingestion.formula_extractor import FormulaExtractor
from src.knowledge.graph.neo4j_manager import Neo4jManager
from src.knowledge.graph.schema import (
    NodeType,
    RelationshipType,
    PaperNode,
    AuthorNode,
    ConceptNode,
    TheoremNode,
    ConjectureNode,
    FormulaNode
)
from src.knowledge.vector.qdrant_manager import QdrantManager
from src.agents import ResearcherAgent


class IngestionPipeline:
    """
    Main pipeline for ingesting mathematical papers into the knowledge system
    """

    def __init__(
        self,
        neo4j_manager: Neo4jManager = None,
        qdrant_manager: 'QdrantManager' = None,
        researcher_agent: ResearcherAgent = None
    ):
        self.logger = get_ingestion_logger()
        self.arxiv_fetcher = ArxivFetcher()
        self.pdf_parser = PDFParser()
        self.formula_extractor = FormulaExtractor()

        # Knowledge storage
        self.neo4j = neo4j_manager
        self.qdrant = qdrant_manager

        # Agent for analysis
        self.researcher = researcher_agent

        # Statistics
        self.stats = {
            'papers_fetched': 0,
            'papers_parsed': 0,
            'formulas_extracted': 0,
            'entities_created': 0,
            'relationships_created': 0,
            'errors': []
        }

    async def ingest_from_arxiv(
        self,
        query: str = None,
        topics: List[str] = None,
        max_papers: int = 10,
        analyze_with_agent: bool = True
    ) -> Dict[str, Any]:
        """
        Complete ingestion pipeline from arXiv

        Args:
            query: Search query for arXiv
            topics: List of mathematical topics
            max_papers: Maximum number of papers to ingest
            analyze_with_agent: Whether to use Researcher agent for analysis

        Returns:
            Pipeline results and statistics
        """
        self.logger.info(f"Starting ingestion pipeline for {max_papers} papers")

        with log_timing("complete_ingestion_pipeline"):
            # Step 1: Fetch papers from arXiv
            papers = await self._fetch_papers(query, topics, max_papers)

            # Step 2: Parse PDFs and extract content
            parsed_papers = await self._parse_papers(papers)

            # Step 3: Extract and process formulas
            processed_papers = await self._process_formulas(parsed_papers)

            # Step 4: Store in knowledge graph
            if self.neo4j:
                await self._store_in_graph(processed_papers)

            # Step 5: Create vector embeddings
            if self.qdrant:
                await self._create_embeddings(processed_papers)

            # Step 6: Analyze with Researcher agent
            if analyze_with_agent and self.researcher:
                await self._analyze_with_agent(processed_papers)

            # Generate report
            report = self._generate_report()

        return report

    async def _fetch_papers(
        self,
        query: str,
        topics: List[str],
        max_papers: int
    ) -> List[Dict[str, Any]]:
        """Fetch papers from arXiv"""
        self.logger.info("Step 1: Fetching papers from arXiv")

        with log_timing("arxiv_fetch"):
            if topics:
                papers = await self.arxiv_fetcher.fetch_mathematical_papers(
                    topics=topics,
                    max_papers=max_papers,
                    recent_only=True
                )
            elif query:
                papers = await self.arxiv_fetcher.search_papers(
                    query=query,
                    max_results=max_papers
                )
            else:
                # Default topics for mathematical conjecture discovery
                papers = await self.arxiv_fetcher.fetch_mathematical_papers(
                    max_papers=max_papers
                )

            self.stats['papers_fetched'] = len(papers)

            # Download PDFs
            for paper in papers:
                if 'local_paths' not in paper:
                    paths = await self.arxiv_fetcher.download_paper(paper)
                    paper['local_paths'] = paths

        self.logger.info(f"Fetched {len(papers)} papers")
        return papers

    async def _parse_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse PDF papers"""
        self.logger.info("Step 2: Parsing PDFs")

        parsed_papers = []

        for paper in papers:
            try:
                with log_timing(f"parse_pdf_{paper['arxiv_id']}"):
                    pdf_path = paper.get('local_paths', {}).get('pdf')

                    if pdf_path and Path(pdf_path).exists():
                        # Parse PDF
                        content = self.pdf_parser.parse_pdf(Path(pdf_path))

                        # Extract mathematical structures
                        structures = self.pdf_parser.extract_mathematical_structures(content.text)
                        proofs = self.pdf_parser.extract_proofs(content.text)

                        paper['extracted_content'] = content
                        paper['mathematical_structures'] = structures
                        paper['proofs'] = proofs

                        parsed_papers.append(paper)
                        self.stats['papers_parsed'] += 1

                        self.logger.debug(f"Parsed paper: {paper['title']}")

            except Exception as e:
                self.logger.error(f"Error parsing paper {paper.get('arxiv_id')}: {e}")
                self.stats['errors'].append({
                    'paper': paper.get('arxiv_id'),
                    'error': str(e),
                    'stage': 'parsing'
                })

        return parsed_papers

    async def _process_formulas(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and extract formulas"""
        self.logger.info("Step 3: Extracting formulas")

        for paper in papers:
            try:
                content = paper.get('extracted_content')
                if content:
                    # Extract formulas from text
                    formulas = self.formula_extractor.extract_formulas_from_text(content.text)

                    # Classify and analyze formulas
                    for formula in formulas:
                        formula.classification = self.formula_extractor.classify_formula(formula)
                        formula.components = self.formula_extractor.extract_formula_components(formula)

                    paper['formulas'] = formulas
                    self.stats['formulas_extracted'] += len(formulas)

                    self.logger.debug(f"Extracted {len(formulas)} formulas from {paper['title']}")

            except Exception as e:
                self.logger.error(f"Error extracting formulas: {e}")
                self.stats['errors'].append({
                    'paper': paper.get('arxiv_id'),
                    'error': str(e),
                    'stage': 'formula_extraction'
                })

        return papers

    async def _store_in_graph(self, papers: List[Dict[str, Any]]):
        """Store extracted knowledge in Neo4j graph"""
        self.logger.info("Step 4: Storing in knowledge graph")

        await self.neo4j.connect()

        for paper in papers:
            try:
                with log_timing(f"store_graph_{paper['arxiv_id']}"):
                    # Create paper node
                    paper_props = {
                        'node_id': f"paper_{paper['arxiv_id']}",
                        'title': paper['title'],
                        'abstract': paper.get('abstract', ''),
                        'arxiv_id': paper['arxiv_id'],
                        'publication_date': paper.get('published_date'),
                        'domain': paper.get('domain', 'mathematics'),
                        'keywords': paper.get('categories', [])
                    }

                    paper_node_id = await self.neo4j.create_node(
                        NodeType.PAPER,
                        paper_props
                    )
                    self.stats['entities_created'] += 1

                    # Create author nodes and relationships
                    for author_data in paper.get('authors', []):
                        author_props = {
                            'node_id': f"author_{author_data['name'].replace(' ', '_').lower()}",
                            'name': author_data['name']
                        }

                        # Check if author exists
                        existing = await self.neo4j.find_node(
                            NodeType.AUTHOR,
                            {'name': author_data['name']}
                        )

                        if existing:
                            author_node_id = existing['_id']
                        else:
                            author_node_id = await self.neo4j.create_node(
                                NodeType.AUTHOR,
                                author_props
                            )
                            self.stats['entities_created'] += 1

                        # Create authorship relationship
                        await self.neo4j.create_relationship(
                            paper_node_id,
                            author_node_id,
                            RelationshipType.AUTHORED_BY
                        )
                        self.stats['relationships_created'] += 1

                    # Store theorems
                    for theorem in paper.get('mathematical_structures', {}).get('theorems', []):
                        theorem_props = {
                            'node_id': f"theorem_{paper['arxiv_id']}_{theorem['number']}",
                            'name': f"Theorem {theorem['number']}",
                            'statement': theorem['content'],
                            'domain': paper.get('domain', 'mathematics')
                        }

                        theorem_node_id = await self.neo4j.create_node(
                            NodeType.THEOREM,
                            theorem_props
                        )
                        self.stats['entities_created'] += 1

                        # Link theorem to paper
                        await self.neo4j.create_relationship(
                            paper_node_id,
                            theorem_node_id,
                            RelationshipType.DEFINES
                        )
                        self.stats['relationships_created'] += 1

                    # Store conjectures
                    for conjecture_text in self._extract_conjectures(paper):
                        conjecture_props = {
                            'node_id': f"conjecture_{paper['arxiv_id']}_{hash(conjecture_text)}",
                            'name': conjecture_text[:100],
                            'statement': conjecture_text,
                            'status': 'open',
                            'domain': paper.get('domain', 'mathematics')
                        }

                        conjecture_node_id = await self.neo4j.create_node(
                            NodeType.CONJECTURE,
                            conjecture_props
                        )
                        self.stats['entities_created'] += 1

                        # Link to paper
                        await self.neo4j.create_relationship(
                            paper_node_id,
                            conjecture_node_id,
                            RelationshipType.DEFINES
                        )
                        self.stats['relationships_created'] += 1

                    # Store formulas
                    for formula in paper.get('formulas', [])[:20]:  # Limit formulas per paper
                        formula_props = {
                            'node_id': f"formula_{paper['arxiv_id']}_{hash(formula.latex)}",
                            'formula_id': f"f_{hash(formula.latex)}",
                            'latex': formula.latex,
                            'description': formula.context if formula.context else '',
                            'domain': paper.get('domain', 'mathematics'),
                            'formula_type': formula.formula_type
                        }

                        formula_node_id = await self.neo4j.create_node(
                            NodeType.FORMULA,
                            formula_props
                        )
                        self.stats['entities_created'] += 1

                        # Link to paper
                        await self.neo4j.create_relationship(
                            paper_node_id,
                            formula_node_id,
                            RelationshipType.CONTAINS
                        )
                        self.stats['relationships_created'] += 1

            except Exception as e:
                self.logger.error(f"Error storing in graph: {e}")
                self.stats['errors'].append({
                    'paper': paper.get('arxiv_id'),
                    'error': str(e),
                    'stage': 'graph_storage'
                })

    async def _create_embeddings(self, papers: List[Dict[str, Any]]):
        """Create vector embeddings for semantic search"""
        self.logger.info("Step 5: Creating vector embeddings")

        # This would integrate with Qdrant for vector storage
        # Placeholder for vector embedding creation
        pass

    async def _analyze_with_agent(self, papers: List[Dict[str, Any]]):
        """Use Researcher agent to analyze papers"""
        self.logger.info("Step 6: Analyzing with Researcher agent")

        for paper in papers[:5]:  # Limit for performance
            try:
                # Prepare paper content for agent
                paper_content = {
                    'title': paper['title'],
                    'abstract': paper.get('abstract', ''),
                    'content': paper.get('extracted_content', {}).get('text', '')[:50000],  # Limit content
                    'formulas': [f.latex for f in paper.get('formulas', [])][:20],
                    'theorems': paper.get('mathematical_structures', {}).get('theorems', [])
                }

                # Analyze with researcher
                analysis = await self.researcher.analyze_paper(
                    paper_content=paper_content['content'],
                    paper_metadata=paper
                )

                paper['agent_analysis'] = analysis.dict()

                self.logger.info(f"Agent analyzed: {paper['title']}")

            except Exception as e:
                self.logger.error(f"Error in agent analysis: {e}")
                self.stats['errors'].append({
                    'paper': paper.get('arxiv_id'),
                    'error': str(e),
                    'stage': 'agent_analysis'
                })

    def _extract_conjectures(self, paper: Dict[str, Any]) -> List[str]:
        """Extract conjectures from paper content"""
        conjectures = []

        content = paper.get('extracted_content')
        if not content:
            return conjectures

        # Pattern matching for conjectures
        conjecture_patterns = [
            r'[Cc]onjecture[:\s]+([^.]+\.)',
            r'[Ww]e conjecture that ([^.]+\.)',
            r'[Oo]pen [Pp]roblem[:\s]+([^.]+\.)',
            r'[Ii]t remains open whether ([^.]+\.)'
        ]

        import re
        for pattern in conjecture_patterns:
            matches = re.finditer(pattern, content.text)
            for match in matches:
                conjectures.append(match.group(1))

        return conjectures

    def _generate_report(self) -> Dict[str, Any]:
        """Generate ingestion report"""
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'statistics': self.stats,
            'success_rate': (self.stats['papers_parsed'] / max(1, self.stats['papers_fetched'])) * 100,
            'entities_per_paper': self.stats['entities_created'] / max(1, self.stats['papers_parsed']),
            'formulas_per_paper': self.stats['formulas_extracted'] / max(1, self.stats['papers_parsed']),
            'error_count': len(self.stats['errors'])
        }

        self.logger.info(f"Ingestion complete: {report}")
        return report

    async def ingest_single_pdf(self, pdf_path: Path, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Ingest a single PDF file

        Args:
            pdf_path: Path to PDF file
            metadata: Optional metadata about the paper

        Returns:
            Ingestion results
        """
        self.logger.info(f"Ingesting single PDF: {pdf_path}")

        # Parse PDF
        content = self.pdf_parser.parse_pdf(pdf_path)

        # Extract formulas
        formulas = self.formula_extractor.extract_formulas_from_text(content.text)

        # Extract structures
        structures = self.pdf_parser.extract_mathematical_structures(content.text)
        proofs = self.pdf_parser.extract_proofs(content.text)

        result = {
            'file': str(pdf_path),
            'metadata': metadata or {},
            'content_length': len(content.text),
            'sections': len(content.sections),
            'formulas': len(formulas),
            'theorems': len(structures.get('theorems', [])),
            'definitions': len(structures.get('definitions', [])),
            'proofs': len(proofs),
            'extracted_content': content,
            'mathematical_structures': structures
        }

        # Store in graph if available
        if self.neo4j and metadata:
            await self._store_single_paper(result)

        return result

    async def _store_single_paper(self, paper_data: Dict[str, Any]):
        """Store single paper in graph"""
        # Implementation similar to _store_in_graph but for single paper
        pass


# Batch processing utilities
class BatchIngestionManager:
    """
    Manage batch ingestion of multiple papers
    """

    def __init__(self, pipeline: IngestionPipeline):
        self.pipeline = pipeline
        self.logger = get_ingestion_logger()

    async def batch_ingest(
        self,
        batch_size: int = 5,
        total_papers: int = 20,
        topics: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Ingest papers in batches

        Args:
            batch_size: Papers per batch
            total_papers: Total papers to ingest
            topics: Mathematical topics

        Returns:
            List of batch reports
        """
        reports = []
        papers_ingested = 0

        while papers_ingested < total_papers:
            current_batch = min(batch_size, total_papers - papers_ingested)

            self.logger.info(f"Processing batch: {papers_ingested}-{papers_ingested + current_batch}")

            report = await self.pipeline.ingest_from_arxiv(
                topics=topics,
                max_papers=current_batch
            )

            reports.append(report)
            papers_ingested += current_batch

            # Rate limiting between batches
            await asyncio.sleep(5)

        return reports


# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize components
        neo4j = Neo4jManager()
        await neo4j.connect()

        # Create pipeline
        pipeline = IngestionPipeline(neo4j_manager=neo4j)

        # Ingest papers
        report = await pipeline.ingest_from_arxiv(
            topics=["Riemann Hypothesis", "P vs NP"],
            max_papers=5,
            analyze_with_agent=False  # Set to True when agent is available
        )

        print(json.dumps(report, indent=2, default=str))

        await neo4j.close()

    asyncio.run(main())