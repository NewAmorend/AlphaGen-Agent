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

    LLM_PROVIDER: str = "openai_compatible"
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = ""
    LLM_WIRE_API: str = "auto"
    LLM_REASONING_EFFORT: str = ""
    LLM_STORE: bool = False
    LLM_ALLOW_INSECURE_HTTP: bool = False
    LLM_CHAT_TOKEN_PARAM: str = "max_tokens"
    LLM_CHAT_REASONING_EFFORT: bool = False
    LLM_MAX_TOKENS: int = 32768
    # 主生成的采样温度。偏低（0.3）易产出同质表达式；调高增大结构/字段多样性，
    # 是最便宜的"减重复"旋钮。refine 另用更高温度（0.55）做变体探索。
    LLM_GEN_TEMPERATURE: float = 0.5
    ANTHROPIC_VERSION: str = "2023-06-01"
    KIMI_API_KEY: str = ""

    # WQ Brain 官方提交阈值（USA TOP3000 delay=1，截至 2026-05）。
    # 评估器优先使用 WQ 自带的 checks 列表；下面这组只在 checks 缺失时作为 fallback。
    MIN_FITNESS: float = 1.0
    MIN_SHARPE: float = 1.25
    MIN_SUB_UNIVERSE_SHARPE: float = 0.67
    MIN_TURNOVER: float = 0.01
    MAX_TURNOVER: float = 0.70
    MIN_RETURNS: float = 0.05  # WQ 不卡，保留作为可选偏好

    # 生成去重：历史已回测且同骨架最佳 fitness 始终低于该值的结构，从新一批生成里排除，
    # 避免反复重测已知低分结构。设为 0 关闭该过滤。
    DEDUP_FITNESS_FLOOR: float = 0.3

    # PnL 收益相关性防重复（gap #4）
    SELF_CORR_THRESHOLD: float = 0.7        # 相关性 cutoff（对齐 WQ）
    SELF_CORR_SHARPE_MARGIN: float = 0.10   # sharpe 超越豁免线（高 10% 则 WQ 仍收）
    SELF_CORR_MIN_OVERLAP: int = 60         # 两向量重叠不足此天数则跳过（判"未知"）

    DB_PATH: str = "./wq_agent.db"

    WIKI_DIR: str = "./wiki"
    WIKI_RETRIEVE_TOP_K: int = 5
    WIKI_GREP_WEIGHT: int = 7
    WIKI_VECTOR_WEIGHT: int = 3
    WIKI_AUTO_RECORD: bool = True
    WIKI_AUTO_RECORD_DIR: str = "./private_wiki"
    WIKI_SUMMARY_CHARS: int = 200

    # EMBEDDING_PROVIDER: none | local | volcengine | zhipu
    EMBEDDING_PROVIDER: str = "none"
    EMBEDDING_MODEL: str = "doubao-embedding-text-240715"   # for volcengine / zhipu
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: str = "https://ark.cn-beijing.volces.com/api/v3/embeddings"
    EMBEDDING_DIM: int = 0   # 0 = 启动后自动检测（本地模型用）
    # 本地模型（pip install fastembed 后启用）；候选见 fastembed.TextEmbedding.list_supported_models()
    LOCAL_EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8-sig", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
