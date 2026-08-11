"""Simple registries for models/datasets/tasks/metrics."""
from typing import Dict, List, Callable, Optional
from core.config import ModelConfig, DatasetConfig, TaskConfig

# Import model clients
from models.openai_client import OpenAIClient
from models.gemini_client import GeminiClient
from models.deepseek_client import DeepSeekClient
from models.scaledown_client import ScaleDownClient
from models.scaledown_summarize_client import ScaleDownSummarizeClient
from models.base import ModelClient, PricingInfo, DEFAULT_MODEL_PRICING

# Import datasets
from dataset.hotpotqa import HotpotQADataset
from dataset.longbench import LongBenchDataset
from dataset.ethic import ETHICDataset
from dataset.msmarco import MSMARCODataset
from dataset.financebench import FinanceBenchDataset
from dataset.qmsum import QMSumDataset
from dataset.clinc_oos import ClincOosDataset
from dataset.base import Dataset

# Import tasks
from tasks.rag_task import RAGTask
from tasks.retrieval_task import RetrievalTask
from tasks.summarization_task import SummarizationTask
from tasks.classification_task import ClassificationTask
from tasks.base import Task

# Import metrics
from metrics.classification import compute_exact_match, compute_f1
from metrics.generation import compute_bertscore
from metrics.rouge_metrics import (
    compute_rouge1,
    compute_rouge2,
    compute_rougeL,
)
from metrics.set_metrics import (
    compute_set_exact_match,
    compute_set_precision,
    compute_set_recall,
    compute_set_f1,
)
from metrics.retrieval import (
    compute_retrieval_precision,
    compute_retrieval_recall,
    compute_retrieval_f1,
)


# ============================================================================
# REGISTRIES
# ============================================================================

MODEL_REGISTRY: Dict[str, type] = {
    "openai": OpenAIClient,
    "gemini": GeminiClient,
    "deepseek": DeepSeekClient,
}

DATASET_REGISTRY: Dict[str, type] = {
    "hotpotqa": HotpotQADataset,
    "longbench": LongBenchDataset,
    "ethic": ETHICDataset,
    "msmarco": MSMARCODataset,
    "financebench": FinanceBenchDataset,
    "qmsum": QMSumDataset,
    "clinc_oos": ClincOosDataset,
}

TASK_REGISTRY: Dict[str, type] = {
    "rag_qa": RAGTask,
    "retrieval_task": RetrievalTask,
    "summarization": SummarizationTask,
    "classification": ClassificationTask,
}

METRIC_REGISTRY: Dict[str, Callable] = {
    "exact_match": compute_exact_match,
    "f1": compute_f1,
    "bertscore": compute_bertscore,
    "rouge1": compute_rouge1,
    "rouge2": compute_rouge2,
    "rougel": compute_rougeL,
    "set_exact_match": compute_set_exact_match,
    "set_precision": compute_set_precision,
    "set_recall": compute_set_recall,
    "set_f1": compute_set_f1,
    "retrieval_precision": compute_retrieval_precision,
    "retrieval_recall": compute_retrieval_recall,
    "retrieval_f1": compute_retrieval_f1,
}


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def get_model_client(config: ModelConfig) -> ModelClient:
    """Instantiate model client from config.

    Args:
        config: ModelConfig with name and optional scaledown settings.

    Returns:
        ModelClient instance (baseline or wrapped with ScaleDownClient).
    """
    # Handle scaledown_summarize: standalone client, no base LLM needed
    if config.mode == "scaledown_summarize":
        pricing = None
        if config.pricing:
            pricing = PricingInfo(
                input_per_1m_tokens=config.pricing.get(
                    "input_per_1m_tokens"
                ),
                output_per_1m_tokens=config.pricing.get(
                    "output_per_1m_tokens"
                ),
            )
        return ScaleDownSummarizeClient(
            model_name=config.name,
            pricing=pricing,
        )

    # Parse model name (format: "provider:model_name" or just "provider")
    parts = config.name.split(":")
    provider = parts[0].lower()
    model_name = parts[1] if len(parts) > 1 else None

    # Get base client class
    if provider not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model provider: {provider}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    client_class = MODEL_REGISTRY[provider]

    # Determine pricing (priority: YAML config > defaults > None)
    pricing = None
    if config.pricing:
        # User provided custom pricing in YAML
        pricing = PricingInfo(
            input_per_1m_tokens=config.pricing.get("input_per_1m_tokens"),
            output_per_1m_tokens=config.pricing.get("output_per_1m_tokens"),
        )
    elif model_name and model_name in DEFAULT_MODEL_PRICING:
        # Use default pricing for known model
        pricing = DEFAULT_MODEL_PRICING[model_name]

    # Instantiate base client with model name and pricing
    if model_name:
        base_client = client_class(model_name=model_name, pricing=pricing)
    else:
        base_client = client_class(pricing=pricing)

    # Wrap with ScaleDown if mode is 'scaledown'
    if config.mode == "scaledown":
        scaledown_config = config.scaledown or {}
        rate = scaledown_config.get("rate", "auto")
        compression_model = scaledown_config.get("compression_model")
        return ScaleDownClient(
            base_client=base_client,
            compression_rate=rate,
            compression_model=compression_model,
            pricing=pricing,
        )

    return base_client


def get_dataset(config: DatasetConfig) -> Dataset:
    """Instantiate dataset from config.

    Args:
        config: DatasetConfig with name, optional subset, and optional task.

    Returns:
        Dataset instance.
    """
    # Parse dataset name (format: "dataset:subset" or just "dataset")
    parts = config.name.split(":")
    dataset_name = parts[0].lower()
    subset = parts[1] if len(parts) > 1 else None

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    dataset_class = DATASET_REGISTRY[dataset_name]

    # Handle LongBench: use config.task for task parameter
    if dataset_name == "longbench":
        task = config.task or None
        return dataset_class(name=config.name, task=task)

    # Handle ETHIC: use subset from name or default to Recalling
    if dataset_name == "ethic":
        ethic_subset = subset if subset else "Recalling"
        return dataset_class(name=config.name, subset=ethic_subset)

    # Handle other datasets: use subset if provided
    if subset:
        return dataset_class(name=config.name, subset=subset)
    else:
        return dataset_class(name=config.name)


def get_task(config: TaskConfig, dataset_name: Optional[str] = None) -> Task:
    """Instantiate task from config.

    Args:
        config: TaskConfig with name.
        dataset_name: Optional dataset name to load specific prompts.

    Returns:
        Task instance.
    """
    task_name = config.name.lower()

    if task_name not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task: {task_name}. "
            f"Available: {list(TASK_REGISTRY.keys())}"
        )

    task_class = TASK_REGISTRY[task_name]

    # Pass dataset_name if the task class supports it
    import inspect
    sig = inspect.signature(task_class.__init__)
    if "dataset_name" in sig.parameters:
        return task_class(name=config.name, dataset_name=dataset_name)

    return task_class(name=config.name)


def get_metrics(metric_names: List[str]) -> Dict[str, Callable]:
    """Get metric functions from names.

    Args:
        metric_names: List of metric names to load.

    Returns:
        Dictionary mapping metric names to callable functions.
    """
    metrics = {}

    for name in metric_names:
        name_lower = name.lower()
        if name_lower not in METRIC_REGISTRY:
            raise ValueError(
                f"Unknown metric: {name}. "
                f"Available: {list(METRIC_REGISTRY.keys())}"
            )
        metrics[name_lower] = METRIC_REGISTRY[name_lower]

    return metrics
