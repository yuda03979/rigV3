from typing import Optional, Sequence, Literal, Union
from pydantic.json_schema import JsonSchemaValue
from globals_dir.utils import CustomBasePydantic


class ConfigOptions(CustomBasePydantic):
    numa: Optional[bool] = None
    num_ctx: Optional[int] = None
    num_batch: Optional[int] = None
    num_gpu: Optional[int] = None
    main_gpu: Optional[int] = None
    low_vram: Optional[bool] = None
    f16_kv: Optional[bool] = None
    logits_all: Optional[bool] = None
    vocab_only: Optional[bool] = None
    use_mmap: Optional[bool] = None
    use_mlock: Optional[bool] = None
    embedding_only: Optional[bool] = None
    num_thread: Optional[int] = None

    # runtime options
    num_keep: Optional[int] = None
    seed: Optional[int] = None
    num_predict: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    tfs_z: Optional[float] = None
    typical_p: Optional[float] = None
    repeat_last_n: Optional[int] = None
    temperature: Optional[float] = None
    repeat_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    mirostat: Optional[int] = None
    mirostat_tau: Optional[float] = None
    mirostat_eta: Optional[float] = None
    penalize_newline: Optional[bool] = None
    stop: Sequence[str] | None = None


class OllamaModelConfig(CustomBasePydantic):
    name: str
    task: Literal["embed", "generate"]

    options: ConfigOptions = ConfigOptions()
    prefix: str = ''  # for embed models
    suffix: str = ''
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[Sequence[int]] = None
    stream: bool = False
    raw: Optional[bool] = None
    format: Literal['', 'json'] | JsonSchemaValue | None = None
    images: Optional[Sequence[Union[str, bytes]]] = None
    keep_alive: Optional[Union[float, str]] = None
    truncate: Optional[bool] = None
