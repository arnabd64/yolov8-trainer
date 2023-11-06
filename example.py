from trainer import YOLOTrainer
from wandb_tracker import ExperimentTracker


# experiment tracking constants
PROJECT_NAME = "Example-Project"
RUN_NAME = "example-run"    # (optional) wandb provides a random name
GROUP = "group-1"           # (optional)
TAGS = ['tag-1','tag-2']    # (optional)
NOTES = "Dataset URL: https://universe.roboflow.com/example/dataset"

# hyperparameters
hyperparameters = dict(batch = 16,
                       epochs = 20,
                       optimizer = "AdamW",
                       cache = "disk",
                       cos_lr = True)

# start experiment
with ExperimentTracker(PROJECT_NAME, hyperparameters, RUN_NAME, TAGS, GROUP, NOTES) as run:
    trainer = YOLOTrainer('yolov8s.pt', run.project, run.name, 'data.yaml', run.config)
    trainer.train()
    model = trainer.best_model