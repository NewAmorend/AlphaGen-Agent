from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    WQ_USERNAME: str = ""
    WQ_PASSWORD: str = ""
    WQ_REGION: str = "USA"
    WQ_UNIVERSE: str = "TOP3000"
    WQ_DELAY: int = 1
    WQ_NEUTRALIZATION: str = "INDUSTRY"
    WQ_TRUNCATION: float = 0.08
    WQ_PASTEURIZATION: str = "ON"
    WQ_MAX_CONCURRENT: int = 5

    LLM_PROVIDER: str = "kimi"
    LLM_MODEL: str = ""
    LLM_MAX_TOKENS: int = 32768
    KIMI_API_KEY: str = ""
    KIMI_BASE_URL: str = "https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions"
    KIMI_MODEL: str = "kimi-k2.6"
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1/chat/completions"
    DEEPSEEK_MODEL: str = "deepseek-chat"

    # WQ Brain 官方提交阈值（USA TOP3000 delay=1，截至 2026-05）。
    # 评估器优先使用 WQ 自带的 checks 列表；下面这组只在 checks 缺失时作为 fallback。
    MIN_FITNESS: float = 1.0
    MIN_SHARPE: float = 1.25
    MIN_SUB_UNIVERSE_SHARPE: float = 0.67
    MIN_TURNOVER: float = 0.01
    MAX_TURNOVER: float = 0.70
    MIN_RETURNS: float = 0.05  # WQ 不卡，保留作为可选偏好

    DB_PATH: str = "./wq_agent.db"

    WIKI_DIR: str = "./wiki"
    WIKI_RETRIEVE_TOP_K: int = 5
    WIKI_GREP_WEIGHT: int = 7
    WIKI_VECTOR_WEIGHT: int = 3
    WIKI_AUTO_RECORD: bool = True
    WIKI_SUMMARY_CHARS: int = 200

    # EMBEDDING_PROVIDER: none | local | volcengine | zhipu
    EMBEDDING_PROVIDER: str = "none"
    EMBEDDING_MODEL: str = "doubao-embedding-text-240715"   # for volcengine / zhipu
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: str = "https://ark.cn-beijing.volces.com/api/v3/embeddings"
    EMBEDDING_DIM: int = 0   # 0 = 启动后自动检测（本地模型用）
    # 本地模型（pip install fastembed 后启用）；候选见 fastembed.TextEmbedding.list_supported_models()
    LOCAL_EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
