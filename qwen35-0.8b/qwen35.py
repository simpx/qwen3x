"""Thin ctypes binding for the stable C ABI in qwen35.h."""

from __future__ import annotations

import ctypes
import threading
from pathlib import Path
from typing import Callable, Iterable


ERROR_CAPACITY = 512


class EngineError(RuntimeError):
    pass


class SessionBusy(EngineError):
    pass


Q35_OK = 0
Q35_BUSY = -2
Q35_NOT_FOUND = -3

Q35_LOG_DEBUG = 0
Q35_LOG_INFO = 1
Q35_LOG_WARN = 2
Q35_LOG_ERROR = 3

LogCallback = Callable[[int, str, int, str], None]
_NativeLogCallback = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_char_p,
)


class _EngineOptions(ctypes.Structure):
    _fields_ = [
        ("bin_path", ctypes.c_char_p),
    ]


_LOG_CALLBACKS: dict[str, tuple[ctypes.CDLL, _NativeLogCallback]] = {}
_LOG_CALLBACK_LOCK = threading.Lock()


def _configure(library: ctypes.CDLL) -> None:
    void_p = ctypes.c_void_p
    char_p = ctypes.c_char_p
    size_t = ctypes.c_size_t
    int_p = ctypes.POINTER(ctypes.c_int)

    library.q35_log_set_callback.argtypes = [
        _NativeLogCallback, void_p, ctypes.c_int
    ]
    library.q35_log_set_callback.restype = None

    library.q35_engine_create.argtypes = [
        ctypes.POINTER(_EngineOptions), ctypes.POINTER(void_p), char_p, size_t
    ]
    library.q35_engine_create.restype = ctypes.c_int
    library.q35_engine_destroy.argtypes = [void_p]
    library.q35_engine_destroy.restype = None

    library.q35_session_create.argtypes = [void_p, ctypes.c_int, ctypes.POINTER(void_p), char_p, size_t]
    library.q35_session_create.restype = ctypes.c_int
    library.q35_session_destroy.argtypes = [void_p]
    library.q35_session_destroy.restype = None
    library.q35_session_reset.argtypes = [void_p, char_p, size_t]
    library.q35_session_reset.restype = ctypes.c_int
    library.q35_session_sync.argtypes = [
        void_p, int_p, ctypes.c_int, int_p, char_p, size_t
    ]
    library.q35_session_sync.restype = ctypes.c_int
    library.q35_session_eval.argtypes = [void_p, ctypes.c_int, char_p, size_t]
    library.q35_session_eval.restype = ctypes.c_int
    library.q35_session_position.argtypes = [void_p]
    library.q35_session_position.restype = ctypes.c_int
    library.q35_session_argmax.argtypes = [void_p]
    library.q35_session_argmax.restype = ctypes.c_int
    library.q35_session_sample.argtypes = [
        void_p,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.q35_session_sample.restype = ctypes.c_int
    library.q35_token_is_stop.argtypes = [ctypes.c_int]
    library.q35_token_is_stop.restype = ctypes.c_bool
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

    library.q35_session_manager_create.argtypes = [
        void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(void_p), char_p, size_t
    ]
    library.q35_session_manager_create.restype = ctypes.c_int
    library.q35_session_manager_destroy.argtypes = [void_p]
    library.q35_session_manager_destroy.restype = None
    library.q35_session_manager_acquire.argtypes = [
        void_p, char_p, int_p, ctypes.c_int, ctypes.POINTER(void_p), char_p, size_t
    ]
    library.q35_session_manager_acquire.restype = ctypes.c_int
    library.q35_session_manager_release.argtypes = [void_p, void_p, ctypes.c_bool]
    library.q35_session_manager_release.restype = None
    library.q35_session_manager_forget.argtypes = [void_p, char_p, char_p, size_t]
    library.q35_session_manager_forget.restype = ctypes.c_int


def set_log_callback(library_path: Path | str,
                     callback: LogCallback | None,
                     level: int = Q35_LOG_INFO) -> None:
    """Configure the process-wide native logger for this shared library."""
    library_path = Path(library_path).resolve()
    if not library_path.is_file():
        raise EngineError(f"engine library not found: {library_path}")

    library = ctypes.CDLL(str(library_path))
    _configure(library)
    native_callback = _NativeLogCallback()
    if callback is not None:
        def forward_log(_user_data, native_level, file, line, message):
            try:
                decode = lambda value: value.decode("utf-8", errors="replace")
                callback(int(native_level), decode(file), int(line), decode(message))
            except Exception:
                # A logger failure must never terminate native inference.
                pass

        native_callback = _NativeLogCallback(forward_log)

    key = str(library_path)
    with _LOG_CALLBACK_LOCK:
        library.q35_log_set_callback(native_callback, None, int(level))
        if callback is None:
            _LOG_CALLBACKS.pop(key, None)
        else:
            # ctypes callbacks must remain alive while C++ can call them.
            _LOG_CALLBACKS[key] = (library, native_callback)


def _call(function, *arguments) -> None:
    error = ctypes.create_string_buffer(ERROR_CAPACITY)
    result = function(*arguments, error, len(error))
    _check(result, error)


def _check(result: int, error) -> None:
    if result == Q35_OK:
        return
    message = error.value.decode("utf-8", errors="replace") or "native engine failed"
    if result == Q35_BUSY:
        raise SessionBusy(message)
    raise EngineError(message)


class Engine:
    """One loaded model. Sessions borrow its mmap'd read-only weights."""

    def __init__(self, library_path: Path | str, bin_path: Path | str):
        library_path = Path(library_path).resolve()
        bin_path = Path(bin_path).resolve()
        if not library_path.is_file():
            raise EngineError(f"engine library not found: {library_path}")
        if not bin_path.is_file():
            raise EngineError(f"packed weight bin not found: {bin_path}")

        self._library = ctypes.CDLL(str(library_path))
        _configure(self._library)
        self._handle = ctypes.c_void_p()
        self._sessions: set[Session] = set()
        self._managers: set[SessionManager] = set()
        self._lock = threading.Lock()

        options = _EngineOptions(
            bin_path=str(bin_path).encode(),
        )
        _call(
            self._library.q35_engine_create,
            ctypes.byref(options),
            ctypes.byref(self._handle),
        )

    @property
    def vocab_size(self) -> int:
        return int(self._library.q35_vocab_size())

    def token_is_stop(self, token: int) -> bool:
        return bool(self._library.q35_token_is_stop(int(token)))

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
            session = Session(self, handle, owns_native=True)
            self._sessions.add(session)
            return session

    def create_session_manager(self, session_count: int, context_size: int) -> "SessionManager":
        with self._lock:
            if not self._handle:
                raise EngineError("engine is closed")
            manager = SessionManager(self, session_count, context_size)
            self._managers.add(manager)
            return manager

    def close(self) -> None:
        with self._lock:
            if not self._handle:
                return
            for manager in list(self._managers):
                manager._close_native()
            self._managers.clear()
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

    def __init__(self, engine: Engine, handle: ctypes.c_void_p, *, owns_native: bool):
        self._engine = engine
        self._handle = handle
        self._owns_native = owns_native
        self._lock = threading.Lock()

    @property
    def position(self) -> int:
        with self._lock:
            self._require_open()
            return int(self._engine._library.q35_session_position(self._handle))

    def sync(self, tokens: Iterable[int]) -> int:
        values = [int(token) for token in tokens]
        if not values:
            raise EngineError("token sequence is empty")
        native = (ctypes.c_int * len(values))(*values)
        cached_tokens = ctypes.c_int()
        with self._lock:
            self._require_open()
            _call(
                self._engine._library.q35_session_sync,
                self._handle,
                native,
                len(values),
                ctypes.byref(cached_tokens),
            )
            return int(cached_tokens.value)

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

    def sample(self, temperature: float, top_p: float, rng: int,
               *, top_k: int = 0) -> tuple[int, int]:
        if temperature < 0:
            raise EngineError("temperature must be non-negative")
        if not 0 < top_p <= 1:
            raise EngineError("top_p must be in (0, 1]")
        native_rng = ctypes.c_uint64(rng)
        with self._lock:
            self._require_open()
            token = int(self._engine._library.q35_session_sample(
                self._handle,
                float(temperature),
                int(top_k),
                float(top_p),
                ctypes.byref(native_rng),
            ))
            if token < 0:
                raise EngineError("session has no logits or sampling options are invalid")
            return token, int(native_rng.value)

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
        if not self._owns_native:
            raise EngineError("managed sessions must be released through SessionManager")
        engine = self._engine
        with engine._lock:
            self._close_native()
            engine._sessions.discard(self)

    def _close_native(self) -> None:
        with self._lock:
            if self._handle:
                if self._owns_native:
                    self._engine._library.q35_session_destroy(self._handle)
                self._handle = ctypes.c_void_p()

    def _detach(self) -> None:
        with self._lock:
            self._handle = ctypes.c_void_p()

    def _require_open(self) -> None:
        if not self._handle:
            raise EngineError("session is closed")

    def __del__(self):
        try:
            if self._owns_native:
                self.close()
        except Exception:
            pass


class SessionManager:
    """Fixed native pool that owns Session state, ID bindings, prefix reuse and LRU."""

    def __init__(self, engine: Engine, session_count: int, context_size: int):
        self._engine = engine
        self._handle = ctypes.c_void_p()
        self._lock = threading.Lock()
        self._active: set[Session] = set()
        self.session_count = int(session_count)
        _call(
            engine._library.q35_session_manager_create,
            engine._handle,
            self.session_count,
            int(context_size),
            ctypes.byref(self._handle),
        )

    @property
    def engine(self) -> Engine:
        return self._engine

    def acquire(self, session_id: str | None, tokens: Iterable[int]) -> Session:
        values = [int(token) for token in tokens]
        if not values:
            raise EngineError("token sequence is empty")
        native = (ctypes.c_int * len(values))(*values)
        encoded = session_id.encode("utf-8") if session_id is not None else None
        with self._lock:
            self._require_open()
            handle = ctypes.c_void_p()
            error = ctypes.create_string_buffer(ERROR_CAPACITY)
            result = self._engine._library.q35_session_manager_acquire(
                self._handle, encoded, native, len(values), ctypes.byref(handle),
                error, len(error),
            )
            _check(result, error)
            session = Session(self._engine, handle, owns_native=False)
            self._active.add(session)
            return session

    def release(self, session: Session, *, keep: bool) -> None:
        with self._lock:
            self._require_open()
            if session not in self._active:
                raise EngineError("session is not acquired from this manager")
            self._engine._library.q35_session_manager_release(
                self._handle, session._handle, bool(keep)
            )
            self._active.remove(session)
            session._detach()

    def forget(self, session_id: str) -> bool:
        with self._lock:
            self._require_open()
            error = ctypes.create_string_buffer(ERROR_CAPACITY)
            result = self._engine._library.q35_session_manager_forget(
                self._handle, session_id.encode("utf-8"), error, len(error)
            )
            if result == Q35_NOT_FOUND:
                return False
            _check(result, error)
            return True

    def close(self) -> None:
        engine = self._engine
        with engine._lock:
            self._close_native()
            engine._managers.discard(self)

    def _close_native(self) -> None:
        with self._lock:
            if not self._handle:
                return
            if self._active:
                raise EngineError("cannot close SessionManager while a request is running")
            self._engine._library.q35_session_manager_destroy(self._handle)
            self._handle = ctypes.c_void_p()

    def _require_open(self) -> None:
        if not self._handle:
            raise EngineError("session manager is closed")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
