from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from datasets import load_metric


class PerformanceBenchmark:
    def __init__(self, pipeline, dataset, optim_type="BERT baseline"):
        self.pipeline = pipeline
        self.dataset = dataset
        self.optim_type = optim_type
        self.acc_score = load_metric("accuracy")

    def compute_accuracy(self):
        preds = [pred["label"] for pred in self.pipeline(self.dataset["text"])]
        preds = self.dataset.features["intent"].str2int(preds)
        labels = self.dataset["intent"]
        acc = self.acc_score(predictions=preds, references=labels)
        print(f"Accuracy on test set : {acc:.2%}")
        return acc

    def compute_size(self):
        model_path = Path("model.pt")
        torch.save(self.pipeline.model.state_dict(), model_path)
        size_mb = model_path.stat().st_size / (1024 * 1024)
        model_path.unlink()
        print("Model size (MB) : {size_mb:.2f}")
        return {"size_mb": size_mb}

    def pipeline_time(self, query="What is the pin number of my account?"):
        latencies = []
        # Warmup
        _ = [self.pipeline(query) for i in range(10)]

        # Multiple runs to get better estimates
        for _ in range(100):
            start_time = perf_counter()
            _ = self.pipeline(query)
            latencies.append(perf_counter() - start_time)
        time_avg_ms = 1000 * np.mean(latencies)
        time_std_ms = 1000 * np.std(latencies)
        print(
            f"Average latency (ms) : {time_avg_ms:.2f} +/- {time_std_ms:.2f}"
        )
        return {"time_avg_ms": time_avg_ms, "time_std_ms": time_std_ms}

    def run_benchmark(self):
        metrics = {}
        metrics[self.optim_type] = self.compute_accuracy()
        metrics[self.optim_type].update(self.compute_size())
        metrics[self.optim_type].update(self.pipeline_time())
        return metrics
