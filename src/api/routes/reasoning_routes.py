"""
Mathematical Reasoning API Routes
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from loguru import logger
import sympy as sp

from src.reasoning.symbolic import (
    SymbolicReasoner, ProofVerifier, ConjectureValidator,
    PatternAnalyzer, EquationSolver, TheoremProver
)
from src.reasoning.numerical import (
    NumericalVerifier, StatisticalAnalyzer, ComputationalExplorer
)
from src.reasoning.geometric import GeometricReasoner, TopologicalAnalyzer
from src.api.websocket_manager import websocket_manager


router = APIRouter()


class SymbolicExpressionRequest(BaseModel):
    """Request for symbolic expression operations"""
    expression: str = Field(..., description="Mathematical expression")
    operation: str = Field(..., description="Operation to perform (simplify, expand, factor, etc.)")
    variables: Optional[List[str]] = None


class EqualityVerificationRequest(BaseModel):
    """Request for equality verification"""
    lhs: str = Field(..., description="Left-hand side expression")
    rhs: str = Field(..., description="Right-hand side expression")


class PatternFindingRequest(BaseModel):
    """Request for pattern finding"""
    sequence: List[Union[int, float]] = Field(..., description="Number sequence")
    max_degree: int = Field(default=5, description="Maximum polynomial degree")


class ConjectureValidationRequest(BaseModel):
    """Request for conjecture validation"""
    conjecture: str = Field(..., description="Conjecture statement")
    test_range: Optional[List[int]] = None
    test_cases: int = Field(default=1000, ge=1, le=100000)


class ProofRequest(BaseModel):
    """Request for proof generation/verification"""
    theorem: str = Field(..., description="Theorem statement")
    proof_type: str = Field(default="auto", description="Proof type (direct, contradiction, induction)")
    assumptions: List[str] = Field(default_factory=list)
    steps: Optional[List[str]] = None


class EquationSolveRequest(BaseModel):
    """Request for equation solving"""
    equation: str = Field(..., description="Equation to solve")
    variable: Optional[str] = None
    domain: str = Field(default="real", description="Solution domain (real, complex, integer)")


class NumericalVerificationRequest(BaseModel):
    """Request for numerical verification"""
    conjecture_type: str = Field(..., description="Type of conjecture (goldbach, collatz, riemann)")
    max_value: int = Field(default=10000, ge=1, le=1000000)


class GeometricRequest(BaseModel):
    """Request for geometric operations"""
    operation: str = Field(..., description="Geometric operation")
    entities: List[Dict[str, Any]] = Field(..., description="Geometric entities")
    parameters: Dict[str, Any] = Field(default_factory=dict)


# Initialize reasoners
symbolic_reasoner = SymbolicReasoner()
proof_verifier = ProofVerifier()
conjecture_validator = ConjectureValidator()
pattern_analyzer = PatternAnalyzer()
equation_solver = EquationSolver()
theorem_prover = TheoremProver()
numerical_verifier = NumericalVerifier()
statistical_analyzer = StatisticalAnalyzer()
computational_explorer = ComputationalExplorer()
geometric_reasoner = GeometricReasoner()
topological_analyzer = TopologicalAnalyzer()


@router.post("/symbolic/expression")
async def process_symbolic_expression(request: SymbolicExpressionRequest):
    """
    Process a symbolic mathematical expression

    Args:
        request: Expression processing request

    Returns:
        Processed expression result
    """
    try:
        expr = symbolic_reasoner.parse_mathematical_expression(request.expression)

        if expr is None:
            raise HTTPException(status_code=400, detail="Invalid expression")

        result = None

        if request.operation == "simplify":
            result = sp.simplify(expr)
        elif request.operation == "expand":
            result = sp.expand(expr)
        elif request.operation == "factor":
            result = sp.factor(expr)
        elif request.operation == "differentiate":
            if request.variables:
                result = sp.diff(expr, *[sp.Symbol(v) for v in request.variables])
            else:
                result = sp.diff(expr)
        elif request.operation == "integrate":
            if request.variables:
                result = sp.integrate(expr, *[sp.Symbol(v) for v in request.variables])
            else:
                result = sp.integrate(expr)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown operation: {request.operation}")

        return {
            "original": str(expr),
            "operation": request.operation,
            "result": str(result),
            "latex": sp.latex(result)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing expression: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/symbolic/verify_equality")
async def verify_equality(request: EqualityVerificationRequest):
    """
    Verify if two expressions are mathematically equal

    Args:
        request: Equality verification request

    Returns:
        Verification result
    """
    try:
        is_equal = symbolic_reasoner.verify_equality(request.lhs, request.rhs)

        return {
            "lhs": request.lhs,
            "rhs": request.rhs,
            "equal": is_equal,
            "lhs_latex": sp.latex(sp.sympify(request.lhs)),
            "rhs_latex": sp.latex(sp.sympify(request.rhs))
        }

    except Exception as e:
        logger.error(f"Error verifying equality: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/symbolic/find_pattern")
async def find_pattern(request: PatternFindingRequest):
    """
    Find mathematical pattern in a sequence

    Args:
        request: Pattern finding request

    Returns:
        Pattern analysis results
    """
    try:
        # Use symbolic reasoner
        pattern = symbolic_reasoner.find_pattern(request.sequence)

        # Use pattern analyzer for more details
        analysis = pattern_analyzer.analyze_sequence(request.sequence)

        # Find recurrence relation
        recurrence = pattern_analyzer.find_recurrence_relation(request.sequence)

        return {
            "sequence": request.sequence,
            "pattern": str(pattern) if pattern else None,
            "analysis": analysis,
            "recurrence_relation": recurrence,
            "next_values": []  # Could predict next values
        }

    except Exception as e:
        logger.error(f"Error finding pattern: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conjecture/validate")
async def validate_conjecture(
    request: ConjectureValidationRequest,
    background_tasks: BackgroundTasks
):
    """
    Validate a mathematical conjecture

    Args:
        request: Conjecture validation request
        background_tasks: Background tasks

    Returns:
        Validation results
    """
    try:
        # Start validation
        test_range = range(*(request.test_range if request.test_range else [1, 1001]))

        results = conjecture_validator.validate_conjecture(
            request.conjecture,
            test_range,
            request.test_cases
        )

        # Check for counterexamples
        counterexample = symbolic_reasoner.generate_counterexample(
            request.conjecture,
            set(test_range),
            max_attempts=min(request.test_cases, 1000)
        )

        if counterexample:
            results["counterexample_found"] = True
            results["counterexample_details"] = counterexample

        # Broadcast discovery if significant
        if results["status"] == "disproven":
            await websocket_manager.broadcast_discovery(
                "counterexample",
                {
                    "conjecture": request.conjecture,
                    "counterexample": results.get("counterexamples", []),
                    "confidence": 1.0
                }
            )

        return results

    except Exception as e:
        logger.error(f"Error validating conjecture: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/proof/generate")
async def generate_proof(request: ProofRequest):
    """
    Generate a mathematical proof

    Args:
        request: Proof generation request

    Returns:
        Generated proof
    """
    try:
        proof = None

        if request.proof_type == "induction" or "induction" in request.theorem.lower():
            # Try proof by induction
            # This would need more sophisticated parsing
            proof = theorem_prover.prove_by_induction(
                base_case=request.assumptions[0] if request.assumptions else "P(1)",
                inductive_step=request.assumptions[1] if len(request.assumptions) > 1 else "P(k) -> P(k+1)",
                conclusion=request.theorem
            )
        elif request.proof_type == "contradiction":
            proof = theorem_prover.prove_by_contradiction(
                assumption=request.assumptions[0] if request.assumptions else "not " + request.theorem,
                derivation=request.steps or [],
                contradiction="False"
            )
        else:
            # Return a template proof structure
            proof = {
                "theorem": request.theorem,
                "proof_type": request.proof_type,
                "assumptions": request.assumptions,
                "steps": request.steps or ["Step 1", "Step 2", "..."],
                "conclusion": request.theorem,
                "verified": False
            }

        # Convert to LaTeX if it's a proof object
        if hasattr(proof, "to_latex"):
            latex = proof.to_latex()
        else:
            latex = None

        return {
            "proof": proof.__dict__ if hasattr(proof, "__dict__") else proof,
            "latex": latex,
            "verified": proof.verified if hasattr(proof, "verified") else False
        }

    except Exception as e:
        logger.error(f"Error generating proof: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/equation/solve")
async def solve_equation(request: EquationSolveRequest):
    """
    Solve an equation or system of equations

    Args:
        request: Equation solving request

    Returns:
        Solutions
    """
    try:
        result = equation_solver.solve_equation(
            request.equation,
            request.variable,
            request.domain
        )

        return result

    except Exception as e:
        logger.error(f"Error solving equation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/numerical/verify")
async def verify_numerically(
    request: NumericalVerificationRequest,
    background_tasks: BackgroundTasks
):
    """
    Numerically verify a mathematical conjecture

    Args:
        request: Numerical verification request
        background_tasks: Background tasks

    Returns:
        Verification results
    """
    try:
        results = {}

        if request.conjecture_type == "goldbach":
            results = numerical_verifier.verify_goldbach_conjecture(request.max_value)
        elif request.conjecture_type == "collatz":
            results = numerical_verifier.verify_collatz_conjecture(request.max_value)
        elif request.conjecture_type == "riemann":
            results = numerical_verifier.verify_riemann_hypothesis_numerically(
                num_zeros=min(request.max_value, 100)
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown conjecture type: {request.conjecture_type}")

        # Broadcast if significant finding
        if not results.get("verified", True):
            await websocket_manager.broadcast_discovery(
                "numerical_counterexample",
                {
                    "conjecture": request.conjecture_type,
                    "counterexample": results.get("counterexample"),
                    "verified_up_to": results.get("verified_up_to")
                }
            )

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in numerical verification: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/numerical/prime_gaps")
async def analyze_prime_gaps(limit: int = 100000):
    """
    Analyze prime gap distribution

    Args:
        limit: Upper limit for prime generation

    Returns:
        Prime gap analysis
    """
    try:
        analysis = statistical_analyzer.analyze_prime_gaps(limit)
        return analysis

    except Exception as e:
        logger.error(f"Error analyzing prime gaps: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/geometric/operation")
async def perform_geometric_operation(request: GeometricRequest):
    """
    Perform geometric operations

    Args:
        request: Geometric operation request

    Returns:
        Operation result
    """
    try:
        result = {}

        if request.operation == "distance":
            entity1 = geometric_reasoner.create_point(request.entities[0]["coordinates"])
            entity2 = geometric_reasoner.create_point(request.entities[1]["coordinates"])
            result["distance"] = geometric_reasoner.distance_between(entity1, entity2)

        elif request.operation == "intersection":
            # Create entities based on types
            entities = []
            for entity_data in request.entities:
                if entity_data["type"] == "point":
                    entities.append(geometric_reasoner.create_point(entity_data["coordinates"]))
                elif entity_data["type"] == "line":
                    entities.append(geometric_reasoner.create_line(
                        entity_data["point"],
                        entity_data["direction"]
                    ))
                elif entity_data["type"] == "plane":
                    entities.append(geometric_reasoner.create_plane(
                        entity_data["point"],
                        entity_data["normal"]
                    ))

            if len(entities) >= 2:
                intersection = geometric_reasoner.intersection(entities[0], entities[1])
                if intersection:
                    result["intersection"] = {
                        "type": intersection.type.value,
                        "coordinates": intersection.coordinates.tolist()
                    }
                else:
                    result["intersection"] = None

        elif request.operation == "convex_hull":
            points = [e["coordinates"] for e in request.entities]
            result = geometric_reasoner.convex_hull(points)

        elif request.operation == "transform":
            entity = geometric_reasoner.create_point(request.entities[0]["coordinates"])
            transformed = geometric_reasoner.transform(
                entity,
                request.parameters["transformation"],
                request.parameters
            )
            result["transformed"] = transformed.coordinates.tolist()

        else:
            raise HTTPException(status_code=400, detail=f"Unknown operation: {request.operation}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in geometric operation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/topological/euler_characteristic")
async def compute_euler_characteristic(
    vertices: int,
    edges: int,
    faces: int
):
    """
    Compute Euler characteristic

    Args:
        vertices: Number of vertices
        edges: Number of edges
        faces: Number of faces

    Returns:
        Euler characteristic and genus
    """
    try:
        euler_char = topological_analyzer.euler_characteristic(vertices, edges, faces)
        genus = topological_analyzer.genus(euler_char)

        return {
            "vertices": vertices,
            "edges": edges,
            "faces": faces,
            "euler_characteristic": euler_char,
            "genus": genus,
            "surface_type": "sphere" if genus == 0 else f"{genus}-torus"
        }

    except Exception as e:
        logger.error(f"Error computing Euler characteristic: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/topological/knot/{knot_code}")
async def get_knot_invariants(knot_code: str):
    """
    Get knot invariants

    Args:
        knot_code: Knot code (e.g., "3_1" for trefoil)

    Returns:
        Knot invariants
    """
    try:
        invariants = topological_analyzer.knot_invariants(knot_code)
        return invariants

    except Exception as e:
        logger.error(f"Error getting knot invariants: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explore/function")
async def explore_function(
    expression: str,
    domain_min: float = -10,
    domain_max: float = 10,
    num_points: int = 1000
):
    """
    Explore mathematical function behavior

    Args:
        expression: Function expression
        domain_min: Minimum domain value
        domain_max: Maximum domain value
        num_points: Number of points to sample

    Returns:
        Function analysis
    """
    try:
        # Parse expression and create function
        expr = sp.sympify(expression)
        symbols = list(expr.free_symbols)

        if len(symbols) != 1:
            raise HTTPException(status_code=400, detail="Function must have exactly one variable")

        var = symbols[0]
        func = sp.lambdify(var, expr, "numpy")

        # Explore function
        analysis = computational_explorer.explore_function_behavior(
            func,
            (domain_min, domain_max),
            num_points
        )

        analysis["expression"] = str(expr)
        analysis["latex"] = sp.latex(expr)

        return analysis

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exploring function: {e}")
        raise HTTPException(status_code=500, detail=str(e))