"""
Configuration Management using Pydantic Settings
"""

from typing import Any, Dict, List, Optional, Set
from pathlib import Path
from pydantic import Field, field_validator, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic.networks import AnyUrl, PostgresDsn, RedisDsn
from enum import Enum


class Environment(str, Enum):
    """Application environment"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class OptimizationMode(str, Enum):
    """LLM optimization modes"""
    QUALITY = "quality"
    SPEED = "speed"
    COST = "cost"


class AWSSettings(BaseSettings):
    """AWS Configuration"""
    region: str = Field("us-east-1", env="AWS_REGION")
    access_key_id: Optional[str] = Field(None, env="AWS_ACCESS_KEY_ID")
    secret_access_key: Optional[SecretStr] = Field(None, env="AWS_SECRET_ACCESS_KEY")
    session_token: Optional[str] = Field(None, env="AWS_SESSION_TOKEN")

    # Bedrock specific
    bedrock_model: str = Field(
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        env="DEFAULT_MODEL"
    )
    enable_model_routing: bool = Field(True, env="ENABLE_MODEL_ROUTING")
    enable_ensemble: bool = Field(False, env="ENABLE_ENSEMBLE")
    optimize_for: OptimizationMode = Field(OptimizationMode.QUALITY, env="OPTIMIZE_FOR")

    model_config = SettingsConfigDict(case_sensitive=False, extra='allow')


class DatabaseSettings(BaseSettings):
    """Database Configuration"""

    # Neo4j
    neo4j_uri: str = Field("bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field("neo4j", env="NEO4J_USER")
    neo4j_password: SecretStr = Field("multiagents123", env="NEO4J_PASSWORD")

    # PostgreSQL
    postgres_host: str = Field("localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(5433, env="POSTGRES_PORT")
    postgres_db: str = Field("multiagents_db", env="POSTGRES_DB")
    postgres_user: str = Field("multiagents", env="POSTGRES_USER")
    postgres_password: SecretStr = Field("multiagents123", env="POSTGRES_PASSWORD")

    # Qdrant Vector DB
    qdrant_host: str = Field("localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(6333, env="QDRANT_PORT")
    qdrant_api_key: Optional[SecretStr] = Field(None, env="QDRANT_API_KEY")
    qdrant_collection_name: str = Field("research_papers", env="QDRANT_COLLECTION_NAME")

    # Redis
    redis_host: str = Field("localhost", env="REDIS_HOST")
    redis_port: int = Field(6379, env="REDIS_PORT")
    redis_password: Optional[SecretStr] = Field("multiagents123", env="REDIS_PASSWORD")
    redis_db: int = Field(0, env="REDIS_DB")
    cache_ttl: int = Field(3600, env="CACHE_TTL")

    # MinIO
    minio_endpoint: str = Field("localhost:9000", env="MINIO_ENDPOINT")
    minio_access_key: str = Field("multiagents", env="MINIO_ACCESS_KEY")
    minio_secret_key: SecretStr = Field("multiagents123", env="MINIO_SECRET_KEY")
    minio_secure: bool = Field(False, env="MINIO_SECURE")
    minio_bucket_papers: str = Field("papers", env="MINIO_BUCKET_PAPERS")
    minio_bucket_outputs: str = Field("outputs", env="MINIO_BUCKET_OUTPUTS")

    @property
    def postgres_url(self) -> str:
        """Get PostgreSQL connection URL"""
        password = self.postgres_password.get_secret_value()
        return f"postgresql://{self.postgres_user}:{password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def redis_url(self) -> str:
        """Get Redis connection URL"""
        if self.redis_password:
            password = self.redis_password.get_secret_value()
            return f"redis://:{password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    model_config = SettingsConfigDict(case_sensitive=False, extra='allow')


class AgentSettings(BaseSettings):
    """Agent Configuration"""
    communication_port: int = Field(5555, env="AGENT_COMMUNICATION_PORT")
    heartbeat_interval: int = Field(30, env="AGENT_HEARTBEAT_INTERVAL")
    agent_timeout: int = Field(300, env="AGENT_TIMEOUT")
    max_concurrent_agents: int = Field(5, env="MAX_CONCURRENT_AGENTS")
    enable_agent_tracing: bool = Field(True, env="ENABLE_AGENT_TRACING")

    # Agent-specific settings
    researcher_max_papers: int = Field(20, env="RESEARCHER_MAX_PAPERS")
    reviewer_skepticism: float = Field(0.95, env="REVIEWER_SKEPTICISM", ge=0.0, le=1.0)
    synthesizer_creativity: float = Field(0.98, env="SYNTHESIZER_CREATIVITY", ge=0.0, le=1.0)
    challenger_adversarial: float = Field(0.99, env="CHALLENGER_ADVERSARIAL", ge=0.0, le=1.0)
    historian_thoroughness: float = Field(0.95, env="HISTORIAN_THOROUGHNESS", ge=0.0, le=1.0)

    model_config = SettingsConfigDict(case_sensitive=False, extra='allow')


class DocumentSettings(BaseSettings):
    """Document Processing Configuration"""
    max_papers_per_session: int = Field(20, env="MAX_PAPERS_PER_SESSION")
    pdf_extraction_method: str = Field("marker", env="PDF_EXTRACTION_METHOD")
    enable_formula_extraction: bool = Field(True, env="ENABLE_FORMULA_EXTRACTION")
    enable_diagram_extraction: bool = Field(True, env="ENABLE_DIAGRAM_EXTRACTION")
    ocr_confidence_threshold: float = Field(0.8, env="OCR_CONFIDENCE_THRESHOLD", ge=0.0, le=1.0)

    # DeepSeek OCR settings (if available)
    deepseek_ocr_endpoint: Optional[str] = Field(None, env="DEEPSEEK_OCR_ENDPOINT")
    deepseek_ocr_api_key: Optional[SecretStr] = Field(None, env="DEEPSEEK_OCR_API_KEY")

    model_config = SettingsConfigDict(case_sensitive=False, extra='allow')


class MathematicalSettings(BaseSettings):
    """Mathematical Processing Configuration"""
    sympy_timeout: int = Field(60, env="SYMPY_TIMEOUT")
    sagemath_endpoint: Optional[str] = Field("http://localhost:8889", env="SAGEMATH_ENDPOINT")
    enable_symbolic_verification: bool = Field(True, env="ENABLE_SYMBOLIC_VERIFICATION")
    proof_verification_depth: int = Field(5, env="PROOF_VERIFICATION_DEPTH")
    conjecture_confidence_threshold: float = Field(0.7, env="CONJECTURE_CONFIDENCE_THRESHOLD", ge=0.0, le=1.0)

    model_config = SettingsConfigDict(case_sensitive=False, extra='allow')


class APISettings(BaseSettings):
    """API Configuration"""
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_reload: bool = Field(True, env="API_RELOAD")
    api_workers: int = Field(4, env="API_WORKERS")
    cors_origins: List[str] = Field(
        ["http://localhost:3000", "http://localhost:8501"],
        env="CORS_ORIGINS"
    )

    # Security
    jwt_secret_key: SecretStr = Field(
        "your_jwt_secret_key_here_change_in_production",
        env="JWT_SECRET_KEY"
    )
    jwt_algorithm: str = Field("HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    api_key: Optional[SecretStr] = Field(
        "your_api_key_here_change_in_production",
        env="API_KEY"
    )

    model_config = SettingsConfigDict(case_sensitive=False, extra='allow')


class UISettings(BaseSettings):
    """UI Configuration"""
    streamlit_port: int = Field(8501, env="STREAMLIT_PORT")
    streamlit_server_headless: bool = Field(True, env="STREAMLIT_SERVER_HEADLESS")
    streamlit_server_enable_cors: bool = Field(False, env="STREAMLIT_SERVER_ENABLE_CORS")
    streamlit_theme: str = Field("dark", env="STREAMLIT_THEME")

    model_config = SettingsConfigDict(case_sensitive=False, extra='allow')


class MonitoringSettings(BaseSettings):
    """Monitoring Configuration"""
    enable_metrics: bool = Field(True, env="ENABLE_METRICS")
    prometheus_port: int = Field(9090, env="PROMETHEUS_PORT")
    grafana_port: int = Field(3000, env="GRAFANA_PORT")
    log_level: LogLevel = Field(LogLevel.INFO, env="LOG_LEVEL")
    log_format: str = Field("json", env="LOG_FORMAT")
    enable_tracing: bool = Field(True, env="ENABLE_TRACING")

    model_config = SettingsConfigDict(case_sensitive=False, extra='allow')


class ExternalServices(BaseSettings):
    """External Service Configuration"""
    arxiv_max_results: int = Field(50, env="ARXIV_MAX_RESULTS")
    arxiv_sort_by: str = Field("relevance", env="ARXIV_SORT_BY")
    semantic_scholar_api_key: Optional[SecretStr] = Field(None, env="SEMANTIC_SCHOLAR_API_KEY")
    openai_api_key: Optional[SecretStr] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[SecretStr] = Field(None, env="ANTHROPIC_API_KEY")

    model_config = SettingsConfigDict(case_sensitive=False, extra='allow')


class FeatureFlags(BaseSettings):
    """Feature Flags"""
    enable_human_in_loop: bool = Field(False, env="ENABLE_HUMAN_IN_LOOP")
    enable_auto_conjecture_generation: bool = Field(True, env="ENABLE_AUTO_CONJECTURE_GENERATION")
    enable_cross_domain_analysis: bool = Field(True, env="ENABLE_CROSS_DOMAIN_ANALYSIS")
    enable_visual_pattern_recognition: bool = Field(False, env="ENABLE_VISUAL_PATTERN_RECOGNITION")
    enable_consensus_mechanism: bool = Field(True, env="ENABLE_CONSENSUS_MECHANISM")

    model_config = SettingsConfigDict(case_sensitive=False, extra='allow')


class Settings(BaseSettings):
    """Main Settings Class"""

    # Environment
    environment: Environment = Field(Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(False, env="DEBUG")
    testing: bool = Field(False, env="TESTING")
    mock_llm_responses: bool = Field(False, env="MOCK_LLM_RESPONSES")
    seed_random: int = Field(42, env="SEED_RANDOM")

    # Sub-configurations
    aws: AWSSettings = Field(default_factory=AWSSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    agents: AgentSettings = Field(default_factory=AgentSettings)
    documents: DocumentSettings = Field(default_factory=DocumentSettings)
    mathematical: MathematicalSettings = Field(default_factory=MathematicalSettings)
    api: APISettings = Field(default_factory=APISettings)
    ui: UISettings = Field(default_factory=UISettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    external: ExternalServices = Field(default_factory=ExternalServices)
    features: FeatureFlags = Field(default_factory=FeatureFlags)

    # Paths
    @property
    def base_dir(self) -> Path:
        """Get base directory"""
        return Path(__file__).parent.parent.parent

    @property
    def data_dir(self) -> Path:
        """Get data directory"""
        return self.base_dir / "data"

    @property
    def papers_dir(self) -> Path:
        """Get papers directory"""
        return self.data_dir / "papers"

    @property
    def outputs_dir(self) -> Path:
        """Get outputs directory"""
        return self.data_dir / "outputs"

    @property
    def cache_dir(self) -> Path:
        """Get cache directory"""
        return self.data_dir / "cache"

    @property
    def logs_dir(self) -> Path:
        """Get logs directory"""
        return self.base_dir / "logs"

    def ensure_directories(self):
        """Ensure all required directories exist"""
        for directory in [self.data_dir, self.papers_dir, self.outputs_dir, self.cache_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    @field_validator("environment", mode='before')
    @classmethod
    def validate_environment(cls, v):
        """Validate environment setting"""
        if isinstance(v, str):
            v = v.lower()
        return v

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra='allow'
    )


# Global settings instance
settings = Settings()

# Ensure directories exist
settings.ensure_directories()


def get_settings() -> Settings:
    """Get settings instance (useful for dependency injection)"""
    return settings


# Helper functions
def is_production() -> bool:
    """Check if running in production"""
    return settings.environment == Environment.PRODUCTION


def is_development() -> bool:
    """Check if running in development"""
    return settings.environment == Environment.DEVELOPMENT


def is_testing() -> bool:
    """Check if running in test mode"""
    return settings.testing or settings.environment == Environment.TESTING


# Export commonly used settings
AWS_REGION = settings.aws.region
NEO4J_URI = settings.database.neo4j_uri
POSTGRES_URL = settings.database.postgres_url
REDIS_URL = settings.database.redis_url
API_HOST = settings.api.api_host
API_PORT = settings.api.api_port
LOG_LEVEL = settings.monitoring.log_level