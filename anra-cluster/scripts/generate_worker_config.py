import json
import os


def generate_worker_configs(
    coordinator_url: str,
    drive_folder_id: str,
    num_workers: int,
    worker_prefix: str = "worker",
    output_dir: str = "."
) -> list[str]:
    configs = []
    for i in range(num_workers):
        worker_id = f"{worker_prefix}_{chr(65 + i)}"
        config = {
            "worker_id": worker_id,
            "coordinator_url": coordinator_url,
            "drive_folder_id": drive_folder_id,
            "account_email": f"your_gmail_{i + 1}@gmail.com",
            "anra_repo_path": "/content/drive/MyDrive/AnRa/v2",
            "checkpoint_path": "/content/drive/MyDrive/AnRa/v2/checkpoints/anra_frontier_500m.pt",
            "training_data": "/content/drive/MyDrive/AnRa/v2/training_data/anra_training.txt",
            "cluster_drive_folder": "/content/drive/MyDrive/AnRa/cluster",
            "batch_size": 1,
            "seq_len": 1024,
        }
        config_path = os.path.join(output_dir, f"{worker_id}_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        configs.append(config_path)
        print(f"Generated {config_path}")
    return configs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate per-worker config files")
    parser.add_argument("--coordinator-url", required=True, help="Coordinator backend URL")
    parser.add_argument("--drive-folder-id", required=True, help="Shared Drive cluster folder ID")
    parser.add_argument("--num-workers", type=int, default=3, help="Number of workers")
    parser.add_argument("--prefix", default="worker", help="Worker ID prefix")
    parser.add_argument("--output", default=".", help="Output directory")
    args = parser.parse_args()
    generate_worker_configs(
        coordinator_url=args.coordinator_url,
        drive_folder_id=args.drive_folder_id,
        num_workers=args.num_workers,
        worker_prefix=args.prefix,
        output_dir=args.output,
    )
