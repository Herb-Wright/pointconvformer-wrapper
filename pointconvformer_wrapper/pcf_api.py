import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../pointconvformer'))

try:
    from pointconvformer.model_architecture import PCF_Backbone
    from pointconvformer.model_architecture import get_default_configs

except ImportError as e:
    from logging import warning

    warning('Failed to import PCF. Cannot use Encoder.')
  
    # if we can't import, mock them
    class PCF_Backbone:
        def __init__(self, *args, **kwargs) -> None:
            raise NotImplementedError()
        
    def get_default_configs(*args, **kwargs):
        raise NotImplementedError

