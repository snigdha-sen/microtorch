import numpy as np
import torch
import signal_models


class ModelMaker:           
    def __call__(self, models, grad, params, f):
                                   
        if len(models) == 1:
            #get the signal model function        
            #modelfunc = getattr(signal_models, models[0])
            
            #S = modelfunc(grad, params) 
            
            S = models[0](grad, params)
                
            
        if len(models) == 2:
            #get the signal model function        
            # modelfunc0 = getattr(signal_models, models[0])
            # modelfunc1 = getattr(signal_models, models[1])
            
            # S = f * modelfunc0(grad, params) \
            #      + (1-f) * modelfunc1(grad, params)   
            
            S = f * models[0](grad, params) \
                  + (1-f) * models[1](grad, params)   
                

                
        if len(models) == 3:
            #get the signal model function        
            modelfunc0 = getattr(signal_models, models[0])
            modelfunc1 = getattr(signal_models, models[1])
            modelfunc2 = getattr(signal_models, models[2])
            
            S = f[0] * modelfunc0(grad, params) \
                 + f[1] * modelfunc1(grad, params) \
                 + (1 - f[0] - f[1]) * modelfunc2(grad, params)
                 
        
        if len(models) == 4:
            #get the signal model function        
            modelfunc0 = getattr(signal_models, models[0])
            modelfunc1 = getattr(signal_models, models[1])
            modelfunc2 = getattr(signal_models, models[2])
            modelfunc3 = getattr(signal_models, models[3])
            
            S = f[0] * modelfunc0(grad, params) \
                 + f[1] * modelfunc1(grad, params) \
                 + (1 - f[0] - f[1]) * modelfunc2(grad, params) \
                + (1 - f[0] - f[1]) * modelfunc2(grad, params)

        return S
    
    

        
      


            