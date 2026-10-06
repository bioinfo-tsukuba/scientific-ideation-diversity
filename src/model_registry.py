"""Shared model and provider identifiers for diversity workflows."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class ProviderName(str, Enum):
    BEDROCK = "bedrock"
    OPENAI = "openai"
    VERTEX = "vertex"
    GOOGLE_AI_STUDIO = "google_ai_studio"
    SPECTER2 = "specter2"
    ANTHROPIC = "anthropic"


class GenerationProviderName(str, Enum):
    """Providers usable for idea generation.

    A generation-specific enum so CLI choices can be derived directly from
    it, rather than filtering out non-generation providers (e.g., SPECTER2)
    from the shared :class:`ProviderName`.  Values match ProviderName values
    one-to-one so manifests remain comparable across the generation /
    embedding / judge pipelines.
    """

    BEDROCK = "bedrock"
    OPENAI = "openai"
    VERTEX = "vertex"
    GOOGLE_AI_STUDIO = "google_ai_studio"


class PromptStyleName(str, Enum):
    STRUCTURED_BG_IDEA = "structured_bg_idea"
    STRUCTURED_FACET = "structured_facet"
    # Prompt-level diversification sensitivity test (#133):
    STRUCTURED_VS = "structured_vs"
    STRUCTURED_SSOT = "structured_ssot"


class CategoryName(str, Enum):
    """LiveIdeaBench subject categories."""

    ACCOUNTING = "Accounting"
    AGRICULTURAL_SCIENCE = "Agricultural science"
    ANTHROPOLOGY = "Anthropology"
    ASTRONOMY = "Astronomy"
    BIOLOGY = "Biology"
    CHEMISTRY = "Chemistry"
    DATA_ENGINEERING = "Data engineering"
    DATA_SCIENCE = "Data science"
    EARTH_SCIENCE = "Earth science"
    ECONOMICS = "Economics"
    FINANCE = "Finance"
    MATHEMATICS_AND_LOGIC = "Mathematics and Logic"
    MEDICINE = "Medicine"
    PEDAGOGY = "Pedagogy"
    PHARMACY = "Pharmacy"
    PHYSICS = "Physics"
    POLITICAL_SCIENCE = "Political science"
    PSYCHOLOGY = "Psychology"
    SOCIOLOGY = "Sociology"
    STATISTICS = "Statistics"
    SYSTEMS_ENGINEERING = "Systems engineering"
    SYSTEMS_SCIENCE = "Systems science"


class EffortName(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class IdeaModelName(str, Enum):
    """Model identifiers for idea generation.

    Values are the provider-specific model IDs used by each backend.
    Claude Sonnet 4.6 runs via Amazon Bedrock (jp region inference profile).
    GPT-5.4 runs via the OpenAI API.
    Gemini 3.1 Pro runs via Vertex AI.
    """

    CLAUDE_SONNET_4_6 = "jp.anthropic.claude-sonnet-4-6"
    GPT_5_4 = "gpt-5.4"
    GEMINI_3_1_PRO = "gemini-3.1-pro-preview"


# Effort levels supported by each idea model.
# Gemini 3.1 Pro cannot fully disable reasoning; EffortName.NONE is not available.
IDEA_MODEL_SUPPORTED_EFFORTS: dict[IdeaModelName, frozenset[EffortName]] = {
    IdeaModelName.CLAUDE_SONNET_4_6: frozenset(EffortName),
    IdeaModelName.GPT_5_4: frozenset(EffortName),
    IdeaModelName.GEMINI_3_1_PRO: frozenset({EffortName.LOW, EffortName.MEDIUM, EffortName.HIGH}),
}

# Max output tokens per model.  Only models whose API requires max_tokens
# (currently Claude) need an entry here.  Other providers use their API default.
IDEA_MODEL_MAX_TOKENS: dict[IdeaModelName, int] = {
    IdeaModelName.CLAUDE_SONNET_4_6: 64_000,
}


# Anthropic-native model identifier for each Anthropic-family
# :class:`IdeaModelName` entry.  Distinct from the model's primary
# ``value`` because Claude runs on Amazon Bedrock with an inference
# profile id (``jp.anthropic.claude-sonnet-4-6``), while Anthropic's
# native SDK (used for ``messages.count_tokens``) requires the bare
# model id (``claude-sonnet-4-6``).  Single-sourcing this mapping
# avoids drift between the existing
# ``scripts/experiment/data/count_claude_reasoning_tokens.py`` and the
# new reasoning-judge backend (PR #253 review).
IDEA_MODEL_ANTHROPIC_NATIVE_ID: dict[IdeaModelName, str] = {
    IdeaModelName.CLAUDE_SONNET_4_6: "claude-sonnet-4-6",
}


# Default provider that serves each idea model.  Mirrors EMBEDDING_MODEL_PROVIDER /
# JUDGE_MODEL_PROVIDER; this is the canonical single-provider mapping used for
# provenance (e.g., ``scripts/experiment/regenerate_empty_facet_records.py``
# derives the provider string from here when backfilling historical records).
# CLI entry points validate ``--provider`` against :data:`IDEA_MODEL_ALLOWED_PROVIDERS`
# so users can pick any supported provider for a given model (e.g., Gemini
# 3.1 Pro can run on either Vertex AI or Google AI Studio).
IDEA_MODEL_PROVIDER: dict[IdeaModelName, GenerationProviderName] = {
    IdeaModelName.CLAUDE_SONNET_4_6: GenerationProviderName.BEDROCK,
    IdeaModelName.GPT_5_4: GenerationProviderName.OPENAI,
    IdeaModelName.GEMINI_3_1_PRO: GenerationProviderName.VERTEX,
}

# Full set of providers the CLI will accept for a given idea model.  Each set
# MUST include the corresponding entry in :data:`IDEA_MODEL_PROVIDER` (the
# default / canonical provider) plus any alternatives.  Gemini 3.1 Pro can
# run on either Vertex AI or Google AI Studio (ai.google.dev); AI Studio
# exposes published Tier-2 hard limits (1k RPM / 7M TPM / 50k RPD) whereas
# Vertex is subject to Dynamic Shared Quota (~23k req/day observed), so both
# paths are kept available and selectable at runtime.
IDEA_MODEL_ALLOWED_PROVIDERS: dict[IdeaModelName, frozenset[GenerationProviderName]] = {
    IdeaModelName.CLAUDE_SONNET_4_6: frozenset({GenerationProviderName.BEDROCK}),
    IdeaModelName.GPT_5_4: frozenset({GenerationProviderName.OPENAI}),
    IdeaModelName.GEMINI_3_1_PRO: frozenset({GenerationProviderName.VERTEX, GenerationProviderName.GOOGLE_AI_STUDIO}),
}


class EmbeddingModelName(str, Enum):
    AMAZON_TITAN_EMBED_TEXT_V2_0 = "amazon.titan-embed-text-v2:0"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    SPECTER2 = "allenai/specter2"
    SPECTER2_ADHOC_QUERY = "allenai/specter2_adhoc_query"
    SPECTER2_BASE = "allenai/specter2_base"


EMBEDDING_MODEL_PROVIDER: dict[EmbeddingModelName, ProviderName] = {
    EmbeddingModelName.AMAZON_TITAN_EMBED_TEXT_V2_0: ProviderName.BEDROCK,
    EmbeddingModelName.TEXT_EMBEDDING_3_LARGE: ProviderName.OPENAI,
    EmbeddingModelName.SPECTER2: ProviderName.SPECTER2,
    EmbeddingModelName.SPECTER2_ADHOC_QUERY: ProviderName.SPECTER2,
}


class JudgeModelName(str, Enum):
    """Model identifiers for LLM-as-judge evaluation.

    Both pairwise (fluency critic) and quality (critic_prompt) pipelines
    share this enum.  Reasoning-capable judges (added for the
    strong-judge ablation in ``rebuttal/strong_judge_design.md``) are
    members of this same enum so any "judge model" type-narrows
    cleanly to :class:`JudgeModelName`; the reasoning-vs-non-reasoning
    branching is done explicitly via :func:`is_reasoning_judge` (which
    consults :data:`REASONING_JUDGE_MODELS`) so the non-reasoning path
    stays free of would-mask branches that could confuse "no reasoning
    trace because the model never produced one" with "thinking was
    disabled".

    The reasoning-judge values are intentionally identical to the
    corresponding :class:`IdeaModelName` SKU strings (single-source
    of truth for the vendor model identifier).  At call time the
    reasoning-judge backend constructs the underlying generation
    backend with ``IdeaModelName(judge_model.value)``.

    * ``GPT_4_1`` — OpenAI Responses, non-reasoning.
    * ``GEMINI_2_5_FLASH`` — Vertex AI / Google AI Studio
      (``google-genai`` SDK, ``thinking_budget=0``).  The GA id is
      pinned rather than a ``-preview`` SKU to avoid Dynamic Shared
      Quota throttling.  Accepted on either provider via
      ``--provider``.
    * ``CLAUDE_HAIKU_4_5`` — Anthropic native SDK; thinking disabled
      by default when the ``thinking`` parameter is not passed.
    * ``GPT_5_4`` — OpenAI Responses with ``reasoning.effort="medium"``
      (vendor default).
    * ``GEMINI_3_1_PRO`` — Vertex AI with
      ``thinking_config.thinking_level="high"`` (vendor default).
    * ``CLAUDE_SONNET_4_6`` — Bedrock Converse with
      ``thinking={"type": "adaptive"}`` and
      ``output_config.effort="high"``.  Claude 4.x has thinking OFF by
      default; ``adaptive`` lets the model decide per call while
      ``effort`` caps the budget.
    """

    # Non-reasoning judges (paper main pipeline).
    GPT_4_1 = "gpt-4.1"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    CLAUDE_HAIKU_4_5 = "claude-haiku-4-5-20251001"
    # Reasoning judges (strong-judge ablation).  Values match
    # :class:`IdeaModelName` SKUs so the model-id string remains
    # single-sourced (see SSOT_IDEA_MODEL_FOR_JUDGE below for the
    # explicit mapping back to IdeaModelName).
    GPT_5_4 = IdeaModelName.GPT_5_4.value
    GEMINI_3_1_PRO = IdeaModelName.GEMINI_3_1_PRO.value
    CLAUDE_SONNET_4_6 = IdeaModelName.CLAUDE_SONNET_4_6.value


# Reasoning judge subset.  Used by :func:`is_reasoning_judge` and by the
# strong-judge CLI to constrain ``--judge-model`` choices.
REASONING_JUDGE_MODELS: frozenset[JudgeModelName] = frozenset({
    JudgeModelName.GPT_5_4,
    JudgeModelName.GEMINI_3_1_PRO,
    JudgeModelName.CLAUDE_SONNET_4_6,
})


def is_reasoning_judge(model: JudgeModelName) -> bool:
    """Return True iff ``model`` runs with vendor reasoning / thinking enabled."""
    return model in REASONING_JUDGE_MODELS


# Translation table from reasoning :class:`JudgeModelName` members back
# to the :class:`IdeaModelName` SKU they share an identifier with.  The
# reasoning-judge backend uses this when constructing the underlying
# generation backend (which is typed on IdeaModelName, the generation-
# side enum, mirroring the original generation pipeline).
SSOT_IDEA_MODEL_FOR_JUDGE: dict[JudgeModelName, IdeaModelName] = {
    JudgeModelName.GPT_5_4: IdeaModelName.GPT_5_4,
    JudgeModelName.GEMINI_3_1_PRO: IdeaModelName.GEMINI_3_1_PRO,
    JudgeModelName.CLAUDE_SONNET_4_6: IdeaModelName.CLAUDE_SONNET_4_6,
}


# Default reasoning level per reasoning judge.  Independent of the
# generation effort (which is varied across runs in the paper).
# Vendor defaults are used so the reviewer-facing framing is
# "off-the-shelf practitioner deployment" rather than "we tuned the
# judge effort to our setting" (see ``rebuttal/strong_judge_design.md``).
REASONING_JUDGE_DEFAULT_EFFORT: dict[JudgeModelName, EffortName] = {
    JudgeModelName.GPT_5_4: EffortName.MEDIUM,
    JudgeModelName.GEMINI_3_1_PRO: EffortName.HIGH,
    JudgeModelName.CLAUDE_SONNET_4_6: EffortName.HIGH,
}


# Default / canonical provider per judge model.  Mirrors IDEA_MODEL_PROVIDER
# on the generation side.  Used for provenance and as the default when
# a caller does not pass ``--provider`` explicitly.
JUDGE_MODEL_PROVIDER: dict[JudgeModelName, ProviderName] = {
    JudgeModelName.GPT_4_1: ProviderName.OPENAI,
    JudgeModelName.GEMINI_2_5_FLASH: ProviderName.VERTEX,
    JudgeModelName.CLAUDE_HAIKU_4_5: ProviderName.ANTHROPIC,
    # Reasoning judges (strong-judge ablation).  Bedrock is canonical
    # for Claude Sonnet 4.6 because the inference profile id is what
    # the Anthropic SDK / Bedrock Converse accept; the count_tokens
    # endpoint (used to derive reasoning tokens, paper §B.1) is the
    # only auxiliary Anthropic-native call.
    JudgeModelName.GPT_5_4: ProviderName.OPENAI,
    JudgeModelName.GEMINI_3_1_PRO: ProviderName.VERTEX,
    JudgeModelName.CLAUDE_SONNET_4_6: ProviderName.BEDROCK,
}


# Full set of providers each judge model can be served on.  Mirrors
# :data:`IDEA_MODEL_ALLOWED_PROVIDERS` on the generation side so the
# judge CLI can accept ``--provider google_ai_studio`` for Gemini
# without introducing a parallel enum value.  Each set MUST include
# the corresponding :data:`JUDGE_MODEL_PROVIDER` entry plus any
# alternatives.
JUDGE_MODEL_ALLOWED_PROVIDERS: dict[JudgeModelName, frozenset[ProviderName]] = {
    JudgeModelName.GPT_4_1: frozenset({ProviderName.OPENAI}),
    JudgeModelName.GEMINI_2_5_FLASH: frozenset({ProviderName.VERTEX, ProviderName.GOOGLE_AI_STUDIO}),
    JudgeModelName.CLAUDE_HAIKU_4_5: frozenset({ProviderName.ANTHROPIC}),
    # Reasoning judges: same allowed providers as the corresponding
    # generation models (paper §B.1).
    JudgeModelName.GPT_5_4: frozenset({ProviderName.OPENAI}),
    JudgeModelName.GEMINI_3_1_PRO: frozenset({ProviderName.VERTEX, ProviderName.GOOGLE_AI_STUDIO}),
    JudgeModelName.CLAUDE_SONNET_4_6: frozenset({ProviderName.BEDROCK}),
}


class AwsRegion(str, Enum):
    """AWS regions used by this project.

    Add entries here as needed when expanding to other regions.
    """

    AP_NORTHEAST_1 = "ap-northeast-1"


class BedrockConfig(BaseModel, frozen=True):
    """Frozen configuration for Bedrock Converse API connections."""

    region: AwsRegion
    connect_timeout: int = 10
    read_timeout: int = 300
    max_pool_connections: int = 256


class VertexAIConfig(BaseModel, frozen=True):
    """Frozen configuration for Vertex AI connections.

    Authentication uses Application Default Credentials (ADC).
    Set up via ``gcloud auth application-default login``.
    """

    project: str
    location: str = "global"


class GoogleAIStudioConfig(BaseModel, frozen=True):
    """Frozen config for Google AI Studio (ai.google.dev) genai client.

    Authentication is via the ``GEMINI_API_KEY`` environment variable, which
    the ``google-genai`` SDK reads automatically when constructing a
    non-Vertex client.  The key is intentionally NOT stored on the config
    object so it never ends up in logs, manifests, or Langfuse metadata.

    Used as an alternative endpoint for the same Gemini models when the
    account's AI Studio tier gives better throughput than Vertex's
    Dynamic Shared Quota (e.g. Tier-2 exposes published 1k RPM / 7M TPM
    / 50k RPD hard limits).  No project / location needed (unlike
    :class:`VertexAIConfig`).
    """


DEFAULT_EFFORTS = list(EffortName)
EFFORT_ORDER = list(DEFAULT_EFFORTS)

# Human-readable display labels for generation models.
IDEA_MODEL_LABEL: dict[IdeaModelName, str] = {
    IdeaModelName.CLAUDE_SONNET_4_6: "Claude Sonnet 4.6",
    IdeaModelName.GPT_5_4: "GPT-5.4",
    IdeaModelName.GEMINI_3_1_PRO: "Gemini 3.1 Pro",
}

# Short key used in results directories and CSV prefixes (e.g. "claude",
# "gpt54", "gemini31pro").
IDEA_MODEL_SHORT_KEY: dict[IdeaModelName, str] = {
    IdeaModelName.CLAUDE_SONNET_4_6: "claude",
    IdeaModelName.GPT_5_4: "gpt54",
    IdeaModelName.GEMINI_3_1_PRO: "gemini31pro",
}

# Name of the phase-2 full-scale run directory under results/effort_diversity/.
IDEA_MODEL_PHASE2_DIR: dict[IdeaModelName, str] = {
    IdeaModelName.CLAUDE_SONNET_4_6: "phase2_claude_1180kw_30x4_facet_seed42",
    IdeaModelName.GPT_5_4: "phase2_gpt54_1180kw_30x4_facet_seed42",
    IdeaModelName.GEMINI_3_1_PRO: "phase2_gemini31pro_1180kw_30x3_facet_seed42",
}

# Embedding service defaults
DEFAULT_SPECTER2_BASE_MODEL_ID: str = EmbeddingModelName.SPECTER2_BASE.value
DEFAULT_SPECTER2_BATCH_SIZE: int = 16


def effort_sort_key(effort: EffortName) -> int:
    return DEFAULT_EFFORTS.index(effort) if effort in DEFAULT_EFFORTS else len(DEFAULT_EFFORTS)
