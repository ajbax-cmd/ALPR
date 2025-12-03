from ultralytics import RTDETR
import torch


class RTdeterWrapper():
    def __init__(self, weigths="rtdetr-l.pt", img_size=640):
        self.model = RTDETR(weigths)
        self.img_size=img_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_model(self, data_yaml, batch_size, epochs, project,name):
        train_results = self.model.train(
        data=data_yaml, 
        epochs=epochs,  
        imgsz=self.img_size,  
        batch=batch_size,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        scale=0.0,
        device=self.device,
        project=project,
        name=name)

        return train_results
    
    def evaluate(self, data_yaml,project,name):
        metrics = self.model.val(
            data=data_yaml,
            imgsz=self.img_size,
            device=self.device,
            project=project,
            name=name
        )
        return metrics
    
    def inference(self, x, show=False, save=False):
        results = self.model(
            x,
            imgsz=self.img_size,
            device=self.device,
            show=show,
            save=save,
        )
        return results

    
        
    