"""Small ctypes binding for the stable C ABI in engine.h."""

from __future__ import annotations

import ctypes
import threading
from pathlib import Path
from typing import Iterable


ERROR_CAPACITY = 512


class EngineError(RuntimeError):
    pass


def _configure(library: ctypes.CDLL) -> None:
    void_p = ctypes.c_void_p
    char_p = ctypes.c_char_p
    size_t = ctypes.c_size_t
    int_p = ctypes.POINTER(ctypes.c_int)

    library.q35_engine_create.argtypes = [char_p, ctypes.POINTER(void_p), char_p, size_t]
    library.q35_engine_create.restype = ctypes.c_int
    library.q35_engine_destroy.argtypes = [void_p]
    library.q35_engine_destroy.restype = None

    library.q35_session_create.argtypes = [void_p, ctypes.c_int, ctypes.POINTER(void_p), char_p, size_t]
    library.q35_session_create.restype = ctypes.c_int
    library.q35_session_destroy.argtypes = [void_p]
    library.q35_session_destroy.restype = None
    library.q35_session_reset.argtypes = [void_p, char_p, size_t]
    library.q35_session_reset.restype = ctypes.c_int
    library.q35_session_sync.argtypes = [void_p, int_p, ctypes.c_int, char_p, size_t]
    library.q35_session_sync.restype = ctypes.c_int
    library.q35_session_eval.argtypes = [void_p, ctypes.c_int, char_p, size_t]
    library.q35_session_eval.restype = ctypes.c_int
    library.q35_session_position.argtypes = [void_p]
    library.q35_session_position.restype = ctypes.c_int
    library.q35_session_argmax.argtypes = [void_p]
    library.q35_session_argmax.restype = ctypes.c_int
    library.q35_session_is_stop_token.argtypes = [ctypes.c_int]
    library.q35_session_is_stop_token.restype = ctypes.c_int
    library.q35_vocab_size.argtypes = []
    library.q35_vocab_size.restype = ctypes.c_int
    library.q35_session_copy_logits.argtypes = [
        void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        char_p,
        size_t,
    ]
    library.q35_session_copy_logits.restype = ctypes.c_int


def _call(function, *arguments) -> None:
    error = ctypes.create_string_buffer(ERROR_CAPACITY)
    result = function(*arguments, error, len(error))
    if result != 0:
        message = error.value.decode("utf-8", errors="replace") or "native engine failed"
        raise EngineError(message)


class Engine:
    """One loaded model. Sessions borrow its mmap'd read-only weights."""

    def __init__(self, library_path: Path | str, weights_path: Path | str):
        library_path = Path(library_path).resolve()
        weights_path = Path(weights_path).resolve()
        if not library_path.is_file():
            raise EngineError(f"engine library not found: {library_path}")
        if not weights_path.is_file():
            raise EngineError(f"packed weights not found: {weights_path}")

        self._library = ctypes.CDLL(str(library_path))
        _configure(self._library)
        self._handle = ctypes.c_void_p()
        _call(
            self._library.q35_engine_create,
            str(weights_path).encode(),
            ctypes.byref(self._handle),
        )
        self._sessions: set[Session] = set()
        self._lock = threading.Lock()

    @property
    def vocab_size(self) -> int:
        return int(self._library.q35_vocab_size())

    def create_session(self, context_size: int) -> "Session":
        with self._lock:
            if not self._handle:
                raise EngineError("engine is closed")
            handle = ctypes.c_void_p()
            _call(
                self._library.q35_session_create,
                self._handle,
                int(context_size),
                ctypes.byref(handle),
            )
            session = Session(self, handle)
            self._sessions.add(session)
            return session

    def close(self) -> None:
        with self._lock:
            if not self._handle:
                return
            for session in list(self._sessions):
                session._close_native()
            self._sessions.clear()
            self._library.q35_engine_destroy(self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class Session:
    """One mutable token timeline. Calls on one Session must be serialized."""

    def __init__(self, engine: Engine, handle: ctypes.c_void_p):
        self._engine = engine
        self._handle = handle
        self._lock = threading.Lock()

    @property
    def position(self) -> int:
        with self._lock:
            self._require_open()
            return int(self._engine._library.q35_session_position(self._handle))

    def sync(self, tokens: Iterable[int]) -> None:
        values = [int(token) for token in tokens]
        if not values:
            raise EngineError("token sequence is empty")
        native = (ctypes.c_int * len(values))(*values)
        with self._lock:
            self._require_open()
            _call(
                self._engine._library.q35_session_sync,
                self._handle,
                native,
                len(values),
            )

    def eval(self, token: int) -> None:
        with self._lock:
            self._require_open()
            _call(self._engine._library.q35_session_eval, self._handle, int(token))

    def reset(self) -> None:
        with self._lock:
            self._require_open()
            _call(self._engine._library.q35_session_reset, self._handle)

    def argmax(self) -> int:
        with self._lock:
            self._require_open()
            token = int(self._engine._library.q35_session_argmax(self._handle))
            if token < 0:
                raise EngineError("session has no logits; sync or eval tokens first")
            return token

    def is_stop_token(self, token: int) -> bool:
        return bool(self._engine._library.q35_session_is_stop_token(int(token)))

    def copy_logits(self) -> list[float]:
        with self._lock:
            self._require_open()
            count = self._engine.vocab_size
            output = (ctypes.c_float * count)()
            _call(
                self._engine._library.q35_session_copy_logits,
                self._handle,
                output,
                count,
            )
            return list(output)

    def close(self) -> None:
        engine = self._engine
        with engine._lock:
            self._close_native()
            engine._sessions.discard(self)

    def _close_native(self) -> None:
        with self._lock:
            if self._handle:
                self._engine._library.q35_session_destroy(self._handle)
                self._handle = ctypes.c_void_p()

    def _require_open(self) -> None:
        if not self._handle:
            raise EngineError("session is closed")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
