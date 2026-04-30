from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import logging
import os
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Optional, Protocol

from semantic_matcher import build_system_prompt, build_user_prompt

logger = logging.getLogger("site_orcamento_ia.llm")


def _carregar_variaveis_ambiente_locais() -> None:
    raiz = Path(__file__).resolve().parent
    for nome_arquivo in (".env.local", ".env", ".env.example"):
        caminho = raiz / nome_arquivo
        if not caminho.exists():
            continue
        try:
            for linha in caminho.read_text(encoding="utf-8").splitlines():
                texto = linha.strip()
                if not texto or texto.startswith("#") or "=" not in texto:
                    continue
                chave, valor = texto.split("=", 1)
                chave = chave.strip()
                valor = valor.strip().strip('"').strip("'")
                if chave and chave not in os.environ:
                    os.environ[chave] = valor
        except Exception as exc:
            logger.warning("Falha ao carregar vari?veis locais de %s: %s", caminho, exc)


_carregar_variaveis_ambiente_locais()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class LLMDecisionConfig:
    enabled: bool = field(default_factory=lambda: _env_bool("ENABLE_LLM_SEMANTIC_DECISION", True))
    backend_mode: str = field(default_factory=lambda: os.getenv("LLM_BACKEND_MODE", "huggingface").strip().lower())
    model_name: str = field(default_factory=lambda: os.getenv("LLM_MODEL_NAME", "qwen2.5:3b-instruct"))
    model_repo: str = field(default_factory=lambda: os.getenv("LLM_MODEL_REPO", "Qwen/Qwen2.5-3B-Instruct"))
    model_file: str = field(default_factory=lambda: os.getenv("LLM_MODEL_FILE", "qwen2.5-1.5b-instruct-q2_k.gguf"))
    model_path: Optional[str] = field(default_factory=lambda: (os.getenv("LLM_MODEL_PATH") or "").strip() or None)
    model_cache_dir: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL_CACHE_DIR", str(Path.home() / ".cache" / "site-orcamento-ia-llm"))
    )
    hf_api_token: Optional[str] = field(
        default_factory=lambda: (os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or "").strip() or None
    )
    hf_provider: Optional[str] = field(default_factory=lambda: (os.getenv("HF_PROVIDER") or "").strip() or None)
    timeout_seconds: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_SECONDS", 4.0))
    min_confidence: float = field(default_factory=lambda: _env_float("LLM_MIN_CONFIDENCE", 0.62))
    max_candidates: int = field(default_factory=lambda: _env_int("LLM_MAX_CANDIDATES", 5))
    decision_min_top_score: float = field(default_factory=lambda: _env_float("LLM_DECISION_MIN_TOP_SCORE", 0.55))
    decision_max_top_score: float = field(default_factory=lambda: _env_float("LLM_DECISION_MAX_TOP_SCORE", 0.86))
    decision_max_gap: float = field(default_factory=lambda: _env_float("LLM_DECISION_MAX_GAP", 0.06))
    llama_n_ctx: int = field(default_factory=lambda: _env_int("LLM_LLAMA_N_CTX", 2048))
    llama_n_threads: int = field(default_factory=lambda: _env_int("LLM_LLAMA_N_THREADS", 4))
    llama_n_batch: int = field(default_factory=lambda: _env_int("LLM_LLAMA_N_BATCH", 64))


class LLMBackend(Protocol):
    def generate(
        self,
        *,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        timeout_seconds: float,
    ) -> str:
        ...


class BackendUnavailableError(RuntimeError):
    pass


class LlamaCppSemanticDecisionBackend:
    def __init__(
        self,
        model_name: str,
        *,
        model_path: Optional[str],
        model_repo: str,
        model_file: str,
        model_cache_dir: str,
        n_ctx: int,
        n_threads: int,
        n_batch: int,
    ) -> None:
        self.model_name = model_name
        self.model_path = Path(model_path).expanduser() if model_path else None
        self.model_repo = model_repo
        self.model_file = model_file
        self.model_cache_dir = Path(model_cache_dir)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_batch = n_batch
        self._model = None
        self._model_path: Optional[Path] = None
        self._load_error: Optional[str] = None
        self._lock = Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="site-orcamento-ia-llm")

    def _resolve_model_path(self) -> Path:
        if self._model_path is not None:
            return self._model_path

        if self.model_path is not None:
            if not self.model_path.exists():
                raise BackendUnavailableError(f"Caminho local do modelo n?o encontrado: {self.model_path}")
            self._model_path = self.model_path
            return self._model_path

        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        if importlib.util.find_spec("huggingface_hub") is None:
            raise BackendUnavailableError("Depend?ncia huggingface_hub n?o est? dispon?vel no ambiente.")
        from huggingface_hub import hf_hub_download

        logger.info("Baixando modelo GGUF '%s' (%s) para cache local...", self.model_file, self.model_repo)
        downloaded_path = hf_hub_download(
            repo_id=self.model_repo,
            filename=self.model_file,
            cache_dir=str(self.model_cache_dir),
        )
        self._model_path = Path(downloaded_path)
        return self._model_path

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if self._load_error:
            raise BackendUnavailableError(self._load_error)

        with self._lock:
            if self._model is not None:
                return
            if self._load_error:
                raise BackendUnavailableError(self._load_error)
            try:
                if importlib.util.find_spec("llama_cpp") is None:
                    raise BackendUnavailableError("Depend?ncia llama-cpp-python n?o est? dispon?vel no ambiente.")

                from llama_cpp import Llama

                model_path = self._resolve_model_path()
                logger.info(
                    "Carregando modelo GGUF '%s' com n_ctx=%s e n_threads=%s...",
                    model_path,
                    self.n_ctx,
                    self.n_threads,
                )
                self._model = Llama(
                    model_path=str(model_path),
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    n_batch=self.n_batch,
                    use_mmap=True,
                    use_mlock=False,
                    verbose=False,
                    n_gpu_layers=0,
                )
            except BackendUnavailableError as exc:
                self._load_error = str(exc)
                raise
            except Exception as exc:
                self._load_error = f"Falha ao carregar backend llama_cpp: {exc}"
                raise BackendUnavailableError(self._load_error) from exc

    def warm_up(self) -> None:
        self._load_model()

    def _generate_sync(self, system_prompt: str, user_prompt: str) -> str:
        self._load_model()
        assert self._model is not None

        mensagens = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            resultado = self._model.create_chat_completion(
                messages=mensagens,
                temperature=0.0,
                top_p=1.0,
                max_tokens=96,
                stream=False,
            )
        except Exception:
            prompt = f"{system_prompt}\n\n{user_prompt}\n\nResponda somente em JSON v?lido."
            resultado = self._model.create_completion(
                prompt=prompt,
                temperature=0.0,
                top_p=1.0,
                max_tokens=96,
                stream=False,
            )

        if isinstance(resultado, dict):
            choices = resultado.get("choices") or []
            if choices:
                first = choices[0] or {}
                message = first.get("message") or {}
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
                text = first.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
        return str(resultado).strip()

    def generate(
        self,
        *,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        timeout_seconds: float,
    ) -> str:
        del model_name
        future = self._executor.submit(self._generate_sync, system_prompt, user_prompt)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise TimeoutError("Infer?ncia local excedeu o tempo limite.") from exc


class HuggingFaceSemanticDecisionBackend:
    def __init__(
        self,
        model_name: str,
        *,
        model_repo: str,
        api_token: Optional[str],
        provider: Optional[str],
    ) -> None:
        self.model_name = model_name
        self.model_repo = model_repo
        self.api_token = api_token
        self.provider = provider or None
        self._client = None
        self._load_error: Optional[str] = None
        self._lock = Lock()

    def _load_client(self) -> None:
        if self._client is not None:
            return
        if self._load_error:
            raise BackendUnavailableError(self._load_error)

        with self._lock:
            if self._client is not None:
                return
            if self._load_error:
                raise BackendUnavailableError(self._load_error)
            try:
                if importlib.util.find_spec("huggingface_hub") is None:
                    raise BackendUnavailableError("Depend?ncia huggingface_hub n?o est? dispon?vel no ambiente.")
                if not self.api_token:
                    raise BackendUnavailableError("HF_API_TOKEN n?o configurado para o backend Hugging Face.")

                from huggingface_hub import InferenceClient

                kwargs = {"api_key": self.api_token}
                if self.provider:
                    kwargs["provider"] = self.provider
                self._client = InferenceClient(**kwargs)
            except BackendUnavailableError as exc:
                self._load_error = str(exc)
                raise
            except Exception as exc:
                self._load_error = f"Falha ao carregar backend Hugging Face: {exc}"
                raise BackendUnavailableError(self._load_error) from exc

    def warm_up(self) -> None:
        self._load_client()

    def generate(
        self,
        *,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        timeout_seconds: float,
    ) -> str:
        del model_name
        self._load_client()
        assert self._client is not None

        mensagens = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            resposta = self._client.chat_completion(
                model=self.model_repo,
                messages=mensagens,
                temperature=0.0,
                max_tokens=96,
                timeout=timeout_seconds,
            )
        except TypeError:
            resposta = self._client.chat_completion(
                model=self.model_repo,
                messages=mensagens,
                temperature=0.0,
                max_tokens=96,
            )
        except Exception as exc:
            raise BackendUnavailableError(f"Falha na infer?ncia remota do Hugging Face: {exc}") from exc

        if hasattr(resposta, "choices"):
            choices = getattr(resposta, "choices") or []
            if choices:
                primeiro = choices[0]
                message = getattr(primeiro, "message", None)
                if message is not None:
                    content = getattr(message, "content", None)
                    if isinstance(content, str) and content.strip():
                        return content.strip()
                text = getattr(primeiro, "text", None)
                if isinstance(text, str) and text.strip():
                    return text.strip()

        if isinstance(resposta, dict):
            choices = resposta.get("choices") or []
            if choices:
                first = choices[0] or {}
                message = first.get("message") or {}
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
                text = first.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()

        if isinstance(resposta, str):
            return resposta.strip()

        return json.dumps(resposta, ensure_ascii=False)


@lru_cache(maxsize=4)
def _create_backend_cached(
    backend_mode: str,
    model_name: str,
    model_repo: str,
    model_file: str,
    model_path: Optional[str],
    model_cache_dir: str,
    hf_api_token: Optional[str],
    hf_provider: Optional[str],
    llama_n_ctx: int,
    llama_n_threads: int,
    llama_n_batch: int,
) -> Optional[LLMBackend]:
    mode = (backend_mode or "huggingface").strip().lower()
    if mode in {"disabled", "off", "false"}:
        return None
    if mode == "huggingface":
        return HuggingFaceSemanticDecisionBackend(
            model_name=model_name,
            model_repo=model_repo,
            api_token=hf_api_token,
            provider=hf_provider,
        )
    if mode != "llama_cpp":
        raise BackendUnavailableError(f"Modo de backend n?o suportado nesta vers?o: {mode}")
    return LlamaCppSemanticDecisionBackend(
        model_name=model_name,
        model_path=model_path,
        model_repo=model_repo,
        model_file=model_file,
        model_cache_dir=model_cache_dir,
        n_ctx=llama_n_ctx,
        n_threads=llama_n_threads,
        n_batch=llama_n_batch,
    )


def create_default_backend(config: Optional[LLMDecisionConfig] = None) -> Optional[LLMBackend]:
    config = config or LLMDecisionConfig()
    if not config.enabled:
        return None
    try:
        return _create_backend_cached(
            config.backend_mode,
            config.model_name,
            config.model_repo,
            config.model_file,
            config.model_path,
            config.model_cache_dir,
            config.hf_api_token,
            config.hf_provider,
            config.llama_n_ctx,
            config.llama_n_threads,
            config.llama_n_batch,
        )
    except BackendUnavailableError as exc:
        logger.warning("Backend sem?ntico indispon?vel: %s", exc)
        return None


def get_llm_runtime_status(config: Optional[LLMDecisionConfig] = None) -> dict:
    config = config or LLMDecisionConfig()

    if not config.enabled:
        return {
            "enabled": False,
            "available": False,
            "status": "disabled",
            "message": "LLM desativada na configura??o atual.",
        }

    mode = (config.backend_mode or "huggingface").strip().lower()
    if mode in {"disabled", "off", "false"}:
        return {
            "enabled": True,
            "available": False,
            "status": "disabled",
            "message": "LLM desativada pelo modo de backend.",
        }

    if mode == "huggingface":
        if importlib.util.find_spec("huggingface_hub") is None:
            return {
                "enabled": True,
                "available": False,
                "status": "missing_dependency",
                "message": "huggingface_hub indispon?vel. O app vai usar fallback sem LLM.",
            }
        if not config.hf_api_token:
            return {
                "enabled": True,
                "available": False,
                "status": "missing_token",
                "message": "HF_API_TOKEN n?o configurado. O app vai usar fallback sem LLM.",
            }
        provider_texto = f" via provider '{config.hf_provider}'" if config.hf_provider else ""
        return {
            "enabled": True,
            "available": True,
            "status": "remote_ready",
            "message": f"LLM remota dispon?vel{provider_texto} com modelo {config.model_repo}.",
        }

    if mode != "llama_cpp":
        return {
            "enabled": True,
            "available": False,
            "status": "unsupported",
            "message": f"Backend '{mode}' n?o suportado nesta vers?o.",
        }

    if importlib.util.find_spec("llama_cpp") is None:
        return {
            "enabled": True,
            "available": False,
            "status": "missing_dependency",
            "message": "llama-cpp-python indispon?vel no ambiente. O app vai usar fallback sem LLM.",
        }

    if config.model_path:
        model_path = Path(config.model_path).expanduser()
        if model_path.exists():
            return {
                "enabled": True,
                "available": True,
                "status": "ready_on_demand",
                "message": f"LLM dispon?vel sob demanda com modelo local em {model_path}.",
            }
        return {
            "enabled": True,
            "available": False,
            "status": "missing_model",
            "message": f"Modelo local configurado, mas n?o encontrado em {model_path}.",
        }

    if importlib.util.find_spec("huggingface_hub") is None:
        return {
            "enabled": True,
            "available": False,
            "status": "missing_dependency",
            "message": "huggingface_hub indispon?vel. Sem download do GGUF, o app vai usar fallback sem LLM.",
        }

    cache_dir = Path(config.model_cache_dir).expanduser()
    cached_model = cache_dir / config.model_file
    if cached_model.exists():
        return {
            "enabled": True,
            "available": True,
            "status": "cached",
            "message": f"LLM dispon?vel. Modelo GGUF j? encontrado no cache ({cached_model}).",
        }

    return {
        "enabled": True,
        "available": True,
        "status": "download_on_first_use",
        "message": "LLM habilitada. O modelo GGUF ser? carregado apenas no primeiro caso amb?guo.",
    }


def run_llm_decision(
    backend: LLMBackend,
    *,
    input_description: str,
    candidates: list[dict],
    metadata: Optional[dict],
    config: Optional[LLMDecisionConfig] = None,
) -> tuple[str, float]:
    config = config or LLMDecisionConfig()
    system_prompt = build_system_prompt(config.model_name)
    user_prompt = build_user_prompt(
        input_description=input_description,
        candidates=candidates,
        metadata=metadata,
        max_candidates=config.max_candidates,
    )
    start = time.perf_counter()
    response_text = backend.generate(
        model_name=config.model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        timeout_seconds=config.timeout_seconds,
    )
    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
    return response_text, elapsed_ms
