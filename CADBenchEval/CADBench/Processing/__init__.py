def __getattr__(name):
    if name == 'BasicProcessor':
        from ._basicprocessor import BasicProcessor
        return BasicProcessor
    elif name == 'VLMProcessor':
        from ._vlmprocessor import VLMProcessor
        return VLMProcessor
    elif name == 'Processor':
        from ._base import Processor
        return Processor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['BasicProcessor', 'VLMProcessor', 'Processor']