"""
Specialized Graph Queries for Mathematical Knowledge Discovery
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger

from src.knowledge.graph.schema import NodeType, RelationshipType


class MathematicalQueries:
    """Specialized queries for mathematical knowledge graph"""

    @staticmethod
    def find_theorem_dependencies() -> str:
        """Find all theorems and their dependencies"""
        return """
        MATCH (t:Theorem)
        OPTIONAL MATCH (t)-[:DEPENDS_ON]->(dep:Theorem)
        OPTIONAL MATCH (t)-[:USES]->(concept:Concept)
        RETURN t.name as theorem,
               t.domain as domain,
               collect(DISTINCT dep.name) as depends_on,
               collect(DISTINCT concept.name) as uses_concepts
        ORDER BY size(depends_on) DESC
        """

    @staticmethod
    def find_conjecture_progress() -> str:
        """Track progress on conjectures over time"""
        return """
        MATCH (c:Conjecture)
        OPTIONAL MATCH (c)<-[:PROVES|DISPROVES]-(p:Proof)
        OPTIONAL MATCH (c)<-[:EXTENDS]-(t:Theorem)
        RETURN c.name as conjecture,
               c.status as status,
               c.year_proposed as year_proposed,
               count(DISTINCT p) as proof_attempts,
               count(DISTINCT t) as related_theorems,
               c.prize as prize
        ORDER BY c.year_proposed DESC
        """

    @staticmethod
    def find_cross_domain_connections() -> str:
        """Find connections between different mathematical domains"""
        return """
        MATCH (d1:Domain)<-[:BELONGS_TO]-(c1:Concept)-[r:RELATED_TO|SIMILAR_TO]-(c2:Concept)-[:BELONGS_TO]->(d2:Domain)
        WHERE d1.name <> d2.name
        RETURN d1.name as domain1,
               d2.name as domain2,
               count(DISTINCT c1) + count(DISTINCT c2) as shared_concepts,
               collect(DISTINCT type(r))[0..5] as relationship_types
        ORDER BY shared_concepts DESC
        """

    @staticmethod
    def find_proof_techniques() -> str:
        """Analyze proof techniques and their effectiveness"""
        return """
        MATCH (p:Proof)-[:PROVES]->(t:Theorem)
        RETURN p.proof_type as technique,
               count(DISTINCT t) as theorems_proved,
               avg(p.completeness) as avg_completeness,
               collect(DISTINCT t.domain)[0..5] as domains
        ORDER BY theorems_proved DESC
        """

    @staticmethod
    def find_collaboration_patterns() -> str:
        """Find collaboration patterns among authors"""
        return """
        MATCH (a1:Author)<-[:AUTHORED_BY]-(p:Paper)-[:AUTHORED_BY]->(a2:Author)
        WHERE a1.name < a2.name
        WITH a1, a2, count(DISTINCT p) as collaborations
        WHERE collaborations > 1
        MATCH (p:Paper)-[:AUTHORED_BY]->(a1)
        MATCH (p)-[:AUTHORED_BY]->(a2)
        MATCH (p)-[:DEFINES|PROVES]->(discovery)
        RETURN a1.name as author1,
               a2.name as author2,
               collaborations,
               count(DISTINCT discovery) as joint_discoveries
        ORDER BY joint_discoveries DESC
        LIMIT 20
        """

    @staticmethod
    def find_emerging_patterns() -> str:
        """Identify emerging patterns in recent research"""
        return """
        MATCH (p:Paper)
        WHERE p.publication_date > datetime() - duration('P2Y')
        MATCH (p)-[:DEFINES|PROVES|USES]->(entity)
        WITH entity, count(DISTINCT p) as recent_papers
        WHERE recent_papers > 3
        MATCH (entity)-[r]-(related)
        RETURN labels(entity)[0] as entity_type,
               entity.name as entity_name,
               recent_papers,
               count(DISTINCT related) as connections,
               collect(DISTINCT type(r))[0..5] as relationship_types
        ORDER BY recent_papers * connections DESC
        LIMIT 30
        """

    @staticmethod
    def trace_concept_evolution() -> str:
        """Trace the evolution of mathematical concepts"""
        return """
        MATCH path = (original:Concept)-[:EVOLVED_FROM|EXTENDS*1..5]->(descendant:Concept)
        WHERE NOT (original)<-[:EVOLVED_FROM|EXTENDS]-()
        RETURN original.name as original_concept,
               [n IN nodes(path) | n.name] as evolution_path,
               length(path) as evolution_depth,
               descendant.name as current_form
        ORDER BY evolution_depth DESC
        LIMIT 20
        """

    @staticmethod
    def find_unproven_dependencies() -> str:
        """Find theorems depending on unproven conjectures"""
        return """
        MATCH (t:Theorem)-[:DEPENDS_ON|USES]->(c:Conjecture)
        WHERE c.status IN ['open', 'partially_proven']
        RETURN t.name as theorem,
               t.domain as domain,
               collect(DISTINCT c.name) as depends_on_conjectures,
               min(c.confidence) as min_confidence
        ORDER BY size(depends_on_conjectures) DESC
        """

    @staticmethod
    def calculate_impact_score() -> str:
        """Calculate impact scores for papers and theorems"""
        return """
        MATCH (p:Paper)
        OPTIONAL MATCH (p)<-[c:CITES]-()
        OPTIONAL MATCH (p)-[:DEFINES|PROVES]->(discovery)
        OPTIONAL MATCH (discovery)<-[:USES|EXTENDS]-()
        WITH p,
             count(DISTINCT c) as citations,
             count(DISTINCT discovery) as discoveries,
             sum(CASE WHEN discovery IS NOT NULL THEN 1 ELSE 0 END) as usage_count
        RETURN p.title as paper,
               p.arxiv_id as arxiv_id,
               citations,
               discoveries,
               usage_count,
               (citations * 1.0 + discoveries * 5.0 + usage_count * 2.0) as impact_score
        ORDER BY impact_score DESC
        LIMIT 50
        """

    @staticmethod
    def find_research_gaps() -> str:
        """Identify research gaps in the knowledge graph"""
        return """
        MATCH (d:Domain)
        OPTIONAL MATCH (d)<-[:BELONGS_TO]-(concept:Concept)
        OPTIONAL MATCH (d)<-[:BELONGS_TO]-(theorem:Theorem)
        OPTIONAL MATCH (d)<-[:BELONGS_TO]-(conjecture:Conjecture)
        WHERE conjecture.status = 'open'
        WITH d,
             count(DISTINCT concept) as concept_count,
             count(DISTINCT theorem) as theorem_count,
             count(DISTINCT conjecture) as open_conjectures
        WHERE concept_count < 10 OR theorem_count < 5
        RETURN d.name as domain,
               concept_count,
               theorem_count,
               open_conjectures,
               d.open_problems as listed_problems
        ORDER BY concept_count + theorem_count ASC
        """

    @staticmethod
    def find_proof_validation_chains() -> str:
        """Find validation chains for proofs"""
        return """
        MATCH (p:Proof)-[:PROVES]->(t:Theorem)
        OPTIONAL MATCH (p)<-[:VALIDATED_BY]-(v:Proof)
        OPTIONAL MATCH (p)<-[:CHALLENGED_BY]-(c:Proof)
        RETURN p.proof_id as proof,
               t.name as theorem,
               p.verified as formally_verified,
               count(DISTINCT v) as validations,
               count(DISTINCT c) as challenges,
               CASE
                   WHEN p.verified THEN 'verified'
                   WHEN count(v) > count(c) THEN 'likely_valid'
                   WHEN count(c) > count(v) THEN 'disputed'
                   ELSE 'unvalidated'
               END as status
        ORDER BY validations DESC
        """

    @staticmethod
    def find_analogous_structures() -> str:
        """Find analogous mathematical structures across domains"""
        return """
        MATCH (s1:Concept)-[r:ANALOGOUS_TO|ISOMORPHIC_TO]-(s2:Concept)
        MATCH (s1)-[:BELONGS_TO]->(d1:Domain)
        MATCH (s2)-[:BELONGS_TO]->(d2:Domain)
        WHERE d1.name <> d2.name
        RETURN s1.name as structure1,
               d1.name as domain1,
               type(r) as relationship,
               s2.name as structure2,
               d2.name as domain2,
               r.similarity_score as similarity
        ORDER BY r.similarity_score DESC
        LIMIT 20
        """

    @staticmethod
    def predict_breakthrough_areas() -> str:
        """Predict areas likely to have breakthroughs"""
        return """
        MATCH (d:Domain)
        MATCH (d)<-[:BELONGS_TO]-(c:Conjecture {status: 'open'})
        OPTIONAL MATCH (c)<-[:EXTENDS|RELATED_TO]-(recent:Theorem)
        WHERE recent.year_proved > datetime().year - 5
        OPTIONAL MATCH (d)<-[:BELONGS_TO]-(p:Paper)
        WHERE p.publication_date > datetime() - duration('P1Y')
        WITH d,
             count(DISTINCT c) as open_conjectures,
             count(DISTINCT recent) as recent_progress,
             count(DISTINCT p) as recent_papers
        WHERE open_conjectures > 0
        RETURN d.name as domain,
               open_conjectures,
               recent_progress,
               recent_papers,
               (recent_progress * 2.0 + recent_papers * 1.0) / open_conjectures as breakthrough_likelihood
        ORDER BY breakthrough_likelihood DESC
        """

    @staticmethod
    def find_circular_dependencies() -> str:
        """Detect circular dependencies in theorems"""
        return """
        MATCH path = (t:Theorem)-[:DEPENDS_ON*2..10]->(t)
        RETURN t.name as theorem,
               [n IN nodes(path) | n.name] as circular_path,
               length(path) as cycle_length
        ORDER BY cycle_length ASC
        LIMIT 10
        """

    @staticmethod
    def analyze_proof_complexity() -> str:
        """Analyze complexity of proofs"""
        return """
        MATCH (p:Proof)-[:PROVES]->(t:Theorem)
        OPTIONAL MATCH (p)-[:USES]->(dep:Theorem)
        OPTIONAL MATCH (p)-[:USES]->(concept:Concept)
        WITH p, t,
             count(DISTINCT dep) as theorem_dependencies,
             count(DISTINCT concept) as concept_dependencies
        RETURN t.name as theorem,
               p.proof_type as proof_type,
               theorem_dependencies,
               concept_dependencies,
               (theorem_dependencies + concept_dependencies) as total_complexity,
               p.verified as is_verified
        ORDER BY total_complexity DESC
        LIMIT 30
        """

    @staticmethod
    def find_interdisciplinary_opportunities() -> str:
        """Find opportunities for interdisciplinary research"""
        return """
        MATCH (d1:Domain)
        MATCH (d2:Domain)
        WHERE d1.name < d2.name
        OPTIONAL MATCH path = (d1)<-[:BELONGS_TO]-()-[*1..3]-()-[:BELONGS_TO]->(d2)
        WITH d1, d2, count(path) as connection_strength
        WHERE connection_strength > 0 AND connection_strength < 10
        MATCH (d1)<-[:BELONGS_TO]-(c1:Conjecture {status: 'open'})
        MATCH (d2)<-[:BELONGS_TO]-(c2:Conjecture {status: 'open'})
        RETURN d1.name as domain1,
               d2.name as domain2,
               connection_strength,
               count(DISTINCT c1) as domain1_open_problems,
               count(DISTINCT c2) as domain2_open_problems,
               connection_strength * (count(c1) + count(c2)) as opportunity_score
        ORDER BY opportunity_score DESC
        LIMIT 15
        """


class GraphAnalytics:
    """Advanced graph analytics queries"""

    @staticmethod
    def community_detection() -> str:
        """Detect research communities"""
        return """
        CALL gds.louvain.stream({
            nodeProjection: 'Author',
            relationshipProjection: {
                COAUTHOR: {
                    type: 'AUTHORED_BY',
                    orientation: 'UNDIRECTED'
                }
            }
        })
        YIELD nodeId, communityId
        WITH communityId, collect(gds.util.asNode(nodeId)) as members
        WHERE size(members) > 3
        UNWIND members as author
        MATCH (author)-[:AUTHORED_BY]->(p:Paper)
        WITH communityId, members, collect(DISTINCT p.domain) as domains
        RETURN communityId,
               size(members) as community_size,
               [m IN members | m.name][0..10] as sample_members,
               domains[0..5] as research_domains
        ORDER BY community_size DESC
        LIMIT 20
        """

    @staticmethod
    def knowledge_flow_analysis() -> str:
        """Analyze how knowledge flows through citations"""
        return """
        MATCH path = (source:Paper)-[:CITES*1..3]->(target:Paper)
        WHERE source.publication_date > target.publication_date
        WITH source, target, length(path) as distance
        MATCH (source)-[:DEFINES|PROVES]->(s_concept)
        MATCH (target)-[:DEFINES|PROVES]->(t_concept)
        WHERE (s_concept)-[:EXTENDS|GENERALIZES]-(t_concept)
        RETURN source.title as source_paper,
               target.title as target_paper,
               distance,
               collect(DISTINCT s_concept.name)[0..3] as source_concepts,
               collect(DISTINCT t_concept.name)[0..3] as target_concepts
        ORDER BY distance ASC
        LIMIT 20
        """

    @staticmethod
    def temporal_trend_analysis() -> str:
        """Analyze temporal trends in research"""
        return """
        WITH datetime().year as current_year
        MATCH (p:Paper)
        WHERE p.publication_date IS NOT NULL
        WITH current_year, p, p.publication_date.year as pub_year
        WHERE pub_year >= current_year - 10
        MATCH (p)-[:DEFINES|PROVES|USES]->(entity)
        WITH pub_year, labels(entity)[0] as entity_type, count(DISTINCT entity) as count
        RETURN pub_year,
               entity_type,
               count
        ORDER BY pub_year DESC, count DESC
        """

    @staticmethod
    def influence_propagation() -> str:
        """Track how influential ideas propagate"""
        return """
        MATCH (influential:Paper)
        WHERE exists((influential)<-[:CITES]-())
        WITH influential, size((influential)<-[:CITES]-()) as citation_count
        WHERE citation_count > 20
        MATCH path = (influential)<-[:CITES*1..3]-(descendant:Paper)
        WITH influential, descendant, length(path) as generation
        MATCH (influential)-[:DEFINES|PROVES]->(original_idea)
        MATCH (descendant)-[:EXTENDS|GENERALIZES]->(evolved_idea)
        WHERE (original_idea)-[:RELATED_TO]-(evolved_idea)
        RETURN influential.title as influential_paper,
               original_idea.name as original_concept,
               generation,
               count(DISTINCT descendant) as influenced_papers,
               collect(DISTINCT evolved_idea.name)[0..5] as evolved_concepts
        ORDER BY influenced_papers DESC
        LIMIT 15
        """


# Query builder helper class
class QueryBuilder:
    """Helper class to build dynamic Cypher queries"""

    @staticmethod
    def build_search_query(
        node_type: NodeType,
        search_terms: List[str],
        properties: List[str],
        limit: int = 50
    ) -> Tuple[str, Dict[str, Any]]:
        """Build a search query for nodes"""

        # Create search conditions
        conditions = []
        for prop in properties:
            term_conditions = [f"toLower(n.{prop}) CONTAINS toLower($term_{i})"
                             for i in range(len(search_terms))]
            if term_conditions:
                conditions.append(f"({' OR '.join(term_conditions)})")

        where_clause = " AND ".join(conditions) if conditions else "true"

        query = f"""
        MATCH (n:{node_type.value})
        WHERE {where_clause}
        RETURN n, elementId(n) as node_id
        ORDER BY n.created_at DESC
        LIMIT $limit
        """

        parameters = {"limit": limit}
        for i, term in enumerate(search_terms):
            parameters[f"term_{i}"] = term

        return query, parameters

    @staticmethod
    def build_path_query(
        start_node_type: NodeType,
        end_node_type: NodeType,
        relationship_types: List[RelationshipType],
        max_depth: int = 5
    ) -> str:
        """Build a path finding query"""

        rel_pattern = "|".join([rt.value for rt in relationship_types])

        return f"""
        MATCH path = (start:{start_node_type.value})-[:{rel_pattern}*1..{max_depth}]->(end:{end_node_type.value})
        WHERE start.node_id = $start_id AND end.node_id = $end_id
        RETURN path
        LIMIT 10
        """

    @staticmethod
    def build_aggregation_query(
        node_type: NodeType,
        group_by: str,
        aggregations: Dict[str, str]
    ) -> str:
        """Build an aggregation query"""

        agg_expressions = [f"{func}(n.{prop}) as {alias}"
                          for alias, (func, prop) in aggregations.items()]

        return f"""
        MATCH (n:{node_type.value})
        WITH n.{group_by} as group_key, n
        RETURN group_key,
               count(n) as count,
               {', '.join(agg_expressions)}
        ORDER BY count DESC
        """