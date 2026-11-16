"""Quality diagnostics to evaluate learning vs single-class collapse."""
from __future__ import annotations


def analyze(history: list[dict], result: dict) -> dict:
    flags = []
    zero_recall_words = []

    per_word = result.get("per_word", {})
    cls_report = result.get("classification_report", {})

    for word, stats in per_word.items():
        recall = cls_report.get(word, {}).get("recall", 0.0)
        if recall == 0.0:
            zero_recall_words.append(word)

    if len(zero_recall_words) > 0:
        flags.append("single_class_collapse")

    acc = result.get("word_accuracy", 0.0)
    # Held-out in-domain accuracy alone does not validate a webcam/OOV product.
    deployment_ready = False

    return {
        "status": "recognition_not_established" if flags else "offline_research_only",
        "deployment_ready": deployment_ready,
        "flags": flags,
        "observations": {
            "zero_recall_words": zero_recall_words,
            "overall_accuracy": acc
        }
    }


def main():
    from .common import parser,load_config,run_contract,load_json,atomic_json,sha256
    args=parser(__doc__).parse_args();config=load_config(args.config);run=run_contract(config,args.run)
    result=load_json(run/'evaluation_summary.json');training=load_json(run/'training_summary.json')
    report=analyze(training.get('history',[]),result)
    report.update(checkpoint_sha256=result['checkpoint_sha256'],evaluation_sha256=sha256(run/'evaluation_summary.json'),
                  limitation='Research diagnostics do not establish calibrated confidence, webcam quality, or open-set rejection.')
    atomic_json(run/'diagnostics.json',report)
    print(report)


if __name__=='__main__':main()
