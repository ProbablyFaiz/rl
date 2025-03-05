import rl.utils.io

_DEBUG_MODE = rl.utils.io.getenv("RL_DEBUG", "0") == "1"
_RICH_TRACEBACK = rl.utils.io.getenv("RL_RICH_TRACEBACK", "0") == "1"


def set_debug_mode(debug_mode: bool):
    global _DEBUG_MODE
    _DEBUG_MODE = debug_mode


def set_rich_traceback(rich_traceback: bool):
    global _RICH_TRACEBACK
    _RICH_TRACEBACK = rich_traceback


def get_debug_mode() -> bool:
    return _DEBUG_MODE


def get_rich_traceback() -> bool:
    return _RICH_TRACEBACK
