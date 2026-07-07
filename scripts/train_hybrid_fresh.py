"""
Smart training launcher: analyzes the dataset, recommends the best model
architecture, then trains with optimal hyperparameters.

Uses CNN (EfficientNetV2-B0) + RNN (BiLSTM+TCN) + ANN (Fusion + Heads).
ViT-B/16 is only enabled when the dataset is large enough to support it.
"""

import sys, logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from ai_model.model_selector import print_recommendation, recommend
from ai_model.hybrid_torch.trainer import FreshHybridConfig, train_fresh_hybrid

if __name__ == "__main__":
    recommendation = recommend(ROOT)
    print_recommendation(recommendation)
    print()

    config = FreshHybridConfig(
        project_root=ROOT,
        auto_configure=True,
        archive_old=True,
        tread_sequence_source="labels",
        pretrained_required=True,
    )

    logger.info("Starting training with auto-selected configuration...")
    result = train_fresh_hybrid(config)
    logger.info("Training complete!")
    logger.info(f"Test metrics: {result['metrics']['test']}")
