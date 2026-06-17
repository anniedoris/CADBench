def __getattr__(name):
    if name == "Eval":
        from . import Eval
        return Eval
    elif name == "Inference":
        from . import Inference
        return Inference
    elif name == "Processing":
        from . import Processing
        return Processing
    elif name == "pretty_print_metrics":
        from .Eval._main import print_results_tabulate
        return print_results_tabulate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Eval", "Inference", "Processing", "pretty_print_metrics"]