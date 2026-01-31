from contextlib import contextmanager

@contextmanager
def nvtx_range(name: str):
    pushed = False
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(name)
            pushed = True
        yield
    finally:
        try:
            if pushed:
                import torch
                torch.cuda.nvtx.range_pop()
        except Exception:
            pass
