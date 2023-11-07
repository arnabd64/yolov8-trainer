from ultralytics import YOLO
from supervision import Detections
import yaml
import os


class YOLOTrainer:
    
    __slots__ = ('model_name','model','project_name','run_name','hyperparameters')
    
    def __init__(
        self,
        model_name = 'yolov8n.pt',
        project_name = 'yolov8-project',
        run_name = 'nano-model-run-0',
        hyperparameters = dict(batch=16, epochs=1)
    ):
        self.model_name = model_name
        self.model = YOLO(model_name)
        self.project_name = project_name
        self.run_name = run_name
        self.hyperparameters = hyperparameters
        
    
    def train(self):
        self.model.train(
            device = 0,
            data = 'data.yaml',
            project = self.project_name,
            name = self.run_name,
            **self.hyperparameters
        )
        
        
    @property
    def best_model_path(self):
        return os.path.join(os.getcwd(), self.project_name, self.run_name, 'weights', 'best.pt')


    @property
    def best_model(self):
        return YOLO(self.best_model_path)
    
    
    def image_inference(self, cv_image, confidence=0.5, use_best_model=True):
        model = self.best_model if use_best_model else self.model        
        output = model.predict(cv_image, conf=confidence, verbose=False)
        return Detections.from_ultralytics(output[0])

    
    def generate_and_export_dataset_yaml(self, labels:list[str], train_dir="train", val_dir="valid", test_dir="test"):
        data_yml = dict(
            path = ".",
            train = train_dir,
            val = val_dir,
            test = test_dir,
            nc = len(labels),
            names = {idx:label for idx, label in enumerate(labels)}
        )
        
        with open(os.path.join(os.getcwd(), 'data.yaml'), "w") as yml:
            yaml.dump(data_yml, yml, yaml.SafeDumper)
            
        return True