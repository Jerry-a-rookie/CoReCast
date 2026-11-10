                                                                    

                                                                           
                                                                            
                                                                          
                                                            
   
from models.ComponentForecastAdapter import ComponentForecastAdapter


class RetrievalBackbone(ComponentForecastAdapter):
    def __init__(self, backbone, name, configs):
        if name not in ('DLinear', 'iTransformer'):
            raise ValueError(name)
        super().__init__(backbone, configs.pred_len, centered_references=True)
        self.name = name
