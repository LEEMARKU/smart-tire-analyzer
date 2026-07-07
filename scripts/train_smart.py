"""
Smart training entry point.

Analyzes the dataset and automatically selects the best model architecture
(CNN + RNN + ANN, with optional ViT depending on dataset size).

Usage:
    python scripts/train_smart.py              # analyze + train with auto-config
    python scripts/train_smart.py --analyze     # analyze only, no training
    python scripts/train_smart.py --no-train    # same as --analyze
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai_model.model_selector import print_recommendation, recommend
from ai_model.hybrid_torch.trainer import FreshHybridConfig, train_fresh_hybrid


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart Tire Analyzer — Smart Training")
    parser.add_argument("--analyze", action="store_true", help="Analyze dataset and recommend config only (no training)")
    parser.add_argument("--no-train", action="store_true", help="Same as --analyze")
    parser.add_argument("--train", action="store_true", help="Force training even with --analyze")
    parser.add_argument("--stage1-epochs", type=int, default=None, help="Override stage 1 epochs")
    parser.add_argument("--stage2-epochs", type=int, default=None, help="Override stage 2 epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--no-vit", action="store_true", help="Force disable ViT")
    parser.add_argument("--force-vit", action="store_true", help="Force enable ViT")
    args = parser.parse_args()

    print()
    rec = recommend(ROOT)
    print_recommendation(rec)
    print()

    analyze_only = args.analyze or args.no_train
    if analyze_only and not args.train:
        print("  [ANALYZE MODE] No training requested. Run without --analyze to train.")
        print()
        return

    config = FreshHybridConfig(
        project_root=ROOT,
        auto_configure=True,
        archive_old=True,
        tread_sequence_source="labels",
    )

    if args.stage1_epochs is not None:
        config.stage1_epochs = args.stage1_epochs
    if args.stage2_epochs is not None:
        config.stage2_epochs = args.stage2_epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.no_vit:
        config.use_vit = False
    if args.force_vit:
        config.use_vit = True

    config.pretrained_required = True
    config.num_workers = 2

    print("  Starting training with selected configuration...")
    print()
    result = train_fresh_hybrid(config)
    print()
    print("  Training complete!")
    print(f"  Test metrics: {result['metrics']['test']}")


if __name__ == "__main__":
    main()
