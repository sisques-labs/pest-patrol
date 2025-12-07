"""Compare trained models and show their performance metrics."""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_comparison_report(report_path: Path) -> Dict:
    """Load comparison report from JSON file.

    Args:
        report_path: Path to comparison report JSON file

    Returns:
        Dictionary containing comparison data
    """
    if not report_path.exists():
        raise FileNotFoundError(f"Comparison report not found: {report_path}")

    with open(report_path, "r") as f:
        return json.load(f)


def compare_models(report_path: Path):
    """Display model comparison from report.

    Args:
        report_path: Path to comparison report JSON file
    """
    report = load_comparison_report(report_path)

    print("\n" + "=" * 80)
    print("MODEL COMPARISON REPORT")
    print("=" * 80)
    print(f"\nTotal models trained: {report['summary']['total_models']}")
    print(f"Best model: {report['summary']['best_model']}")
    print(f"Best validation accuracy: {report['summary']['best_accuracy']*100:.2f}%")

    print("\n" + "-" * 80)
    print(f"{'Rank':<6} {'Model':<25} {'Parameters':<15} {'Val Acc':<12} {'Val Loss':<12} {'Checkpoint':<30}")
    print("-" * 80)

    for i, model in enumerate(report["models"], 1):
        checkpoint_name = Path(model["checkpoint_path"]).name
        print(
            f"{i:<6} "
            f"{model['model_name']:<25} "
            f"{model['num_parameters']:>14,} "
            f"{model['best_val_acc']*100:>10.2f}% "
            f"{model['best_val_loss']:>11.4f} "
            f"{checkpoint_name:<30}"
        )

    print("=" * 80)
    print("\nDetailed Results:")
    print("-" * 80)
    for model in report["models"]:
        print(f"\n{model['model_name']}:")
        print(f"  Parameters: {model['num_parameters']:,}")
        print(f"  Best Val Accuracy: {model['best_val_acc']*100:.2f}%")
        print(f"  Best Val Loss: {model['best_val_loss']:.4f}")
        print(f"  Final Train Accuracy: {model['final_train_acc']*100:.2f}%")
        print(f"  Final Val Accuracy: {model['final_val_acc']*100:.2f}%")
        print(f"  Checkpoint: {model['checkpoint_path']}")
        print(f"  Output Directory: {model['output_dir']}")

    print("\n" + "=" * 80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Compare trained models from comparison report"
    )
    parser.add_argument(
        "--report",
        type=str,
        default="outputs/model_comparison.json",
        help="Path to model comparison report JSON file",
    )

    args = parser.parse_args()
    report_path = Path(args.report)

    try:
        compare_models(report_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you have run train_multi.py first to generate the comparison report.")
    except Exception as e:
        print(f"Error loading report: {e}")


if __name__ == "__main__":
    main()

