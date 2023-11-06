import wandb


class ExperimentTracker:
    
    __slots__ = ('project_name','run_name','config','tags','run','group','notes')
    
    def __init__(self, project_name, config=None, run_name=None, tags=None, group=None, notes=None):
        self.project_name = project_name
        self.run_name = run_name
        self.config = config
        self.tags = tags
        self.group = group
        self.notes = notes
        self.run = None
        
        
    def __enter__(self):
        self.run = wandb.init(project = self.project_name,
                              name = self.run_name,
                              config = self.config,
                              tags = self.tags,
                              group = self.group,
                              notes = self.notes)
        return self.run
    
    
    def __exit__(self, *args, **kwargs):
        self.run.finish()
        