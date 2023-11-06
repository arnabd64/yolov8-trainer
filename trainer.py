from ultralytics import YOLO
from supervision import Detections
import os
import torch


class YOLOTrainer:
    
    __slots__ = ('model_name','model','project_name','run_name','yaml_path','hyperparameters','device')
    
    def __init__(
        self,
        model_name = 'yolov8n.pt',
        project_name = 'yolov8-project',
        run_name = 'nano-model-run-0',
        yaml_path = 'data.yaml',
        hyperparameters = dict(batch=16, epochs=1)
    ):
        self.model_name = model_name
        self.model = YOLO(model_name)
        self.project_name = project_name
        self.run_name = run_name
        self.yaml_path = yaml_path
        self.hyperparameters = hyperparameters
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    
    def train(self):
        self.model.train(
            device = self.device,
            data = self.yaml_path,
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
