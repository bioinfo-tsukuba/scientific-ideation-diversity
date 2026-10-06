"""Display-name and color mappings for LLM identifiers.

Kept outside ``src/model_registry`` because it is a presentation concern
specific to this project's plots, not a system-level identity property.
"""

from __future__ import annotations

from src.model_registry import IdeaModelName

IDEA_MODEL_DISPLAY_NAMES: dict[IdeaModelName, str] = {
    IdeaModelName.CLAUDE_SONNET_4_6: "Claude Sonnet 4.6",
    IdeaModelName.GPT_5_4: "GPT-5.4",
    IdeaModelName.GEMINI_3_1_PRO: "Gemini 3.1 Pro",
}

IDEA_MODEL_COLORS: dict[IdeaModelName, str] = {
    IdeaModelName.CLAUDE_SONNET_4_6: "#0000FF",
    IdeaModelName.GPT_5_4: "#FF0000",
    IdeaModelName.GEMINI_3_1_PRO: "#F9A825",
}
