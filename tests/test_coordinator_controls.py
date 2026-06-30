import asyncio
import copy
import shutil

import torch
from sqlmodel import Session

from backend import main
from backend.database import ClusterState
from worker.sparse_grad import save_sparse_gradient, sparsify_gradients


class FakeDrive:
    def __init__(self, state, gradients, root):
        self.state = copy.deepcopy(state)
        self.gradients = gradients
        self.root = root

    def acquire_lock(self, holder_id):
        self.state["lock_holder"] = holder_id
        return True

    def release_lock(self, holder_id):
        if self.state.get("lock_holder") == holder_id:
            self.state["lock_holder"] = None

    def read_coordinator_state(self):
        return copy.deepcopy(self.state)

    def write_coordinator_state(self, state):
        self.state = copy.deepcopy(state)

    def read_gradient_file(self, worker_id, step):
        return self.gradients[(worker_id, step)]

    def write_sparse_delta(self, local_path, version):
        name = f"delta_v{version:06d}.npz"
        shutil.copyfile(local_path, self.root / name)
        return name

    def cleanup_old_gradients(self, keep_last_n_steps=3):
        return None


def create_gradient(path, worker_id, step):
    model = torch.nn.Sequential(torch.nn.Linear(16, 16, bias=False))
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    payload = sparsify_gradients(
        dict(model.named_parameters()),
        token_count=1024,
        worker_id=worker_id,
        step=step,
        model_version=0,
        topk_fraction=0.1,
    )
    save_sparse_gradient(payload, path)


def test_controls_and_aggregation_keep_sql_and_drive_in_sync(isolated_database, tmp_path):
    gradients = {}
    submissions = {}
    for worker_id in ("worker_A", "worker_B"):
        path = tmp_path / f"{worker_id}.npz"
        create_gradient(path, worker_id, 64)
        gradients[(worker_id, 64)] = str(path)
        submissions[worker_id] = {
            "step": 64,
            "minibatch_tokens": 1024,
            "model_version": 0,
            "layer_norms": [2.0, 1.0],
        }

    drive = FakeDrive(
        {
            "global_step": 0,
            "phase": "training",
            "submitted_this_step": ["worker_A", "worker_B"],
            "submission_times": {"worker_A": 1, "worker_B": 2},
            "gradient_submissions": submissions,
            "master_weights_version": 0,
            "checkpoint_version": 0,
            "master_weights_path": "base.pt",
            "delta_history": [],
            "current_lr": 1e-3,
        },
        gradients,
        tmp_path,
    )
    main.drive_sync_instance = drive
    with Session(isolated_database) as session:
        session.add(ClusterState(id=1, phase="training"))
        session.commit()

    assert asyncio.run(main.pause_training()) == {"status": "paused"}
    assert asyncio.run(main.resume_training()) == {"status": "resumed"}
    asyncio.run(main.run_aggregation(64))

    assert drive.state["global_step"] == 64
    assert drive.state["master_weights_version"] == 1
    assert drive.state["phase"] == "training"
    assert drive.state["latest_layer_norms"] == [2.0, 1.0]
    with Session(isolated_database) as session:
        state = session.get(ClusterState, 1)
        assert state.global_step == drive.state["global_step"]
        assert state.master_weights_version == drive.state["master_weights_version"]
        assert state.phase == drive.state["phase"]
