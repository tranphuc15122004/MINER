import logging

logger = logging.getLogger(__name__)

# ======================================================
# Compat layer for old code using transformers.modeling_utils
# ======================================================

try:
    # Transformers <= 4.30 style
    from transformers.utils import (
        WEIGHTS_NAME,
        TF2_WEIGHTS_NAME,
        TF_WEIGHTS_NAME,
    )
    try:
        from transformers.utils import cached_file as cached_path
    except ImportError:
        from transformers.utils.hub import cached_file as cached_path
except ImportError:
    # Fallback
    WEIGHTS_NAME = "pytorch_model.bin"
    TF2_WEIGHTS_NAME = "tf_model.h5"
    TF_WEIGHTS_NAME = "model.ckpt"

    def cached_path(*args, **kwargs):
        raise EnvironmentError("cached_path not available.")


def get_checkpoint_from_transformer_cache(
    pretrained_model_name_or_path,
    cache_dir=None,
    force_download=False,
    proxies=None,
    resume_download=False,
    local_files_only=False,
    use_auth_token=None,
    revision=None,
    subfolder="",
    from_tf=False,
):
    """
    Simplified version for compatibility.
    Repo không dùng TuringNLRv3, chỉ cần load weight PT là đủ.
    """
    filename = TF2_WEIGHTS_NAME if from_tf else WEIGHTS_NAME

    try:
        resolved = cached_path(
            pretrained_model_name_or_path,
            filename,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
            revision=revision,
            subfolder=subfolder,
        )
        return resolved
    except Exception as e:
        logger.error(f"Error in get_checkpoint_from_transformer_cache: {e}")
        raise


def state_dict_convert(state_dict, config=None):
    """
    Stub: code gốc dùng để rename weight checkpoint.
    Repo của bạn không chạy mô hình NLRv3, nên trả nguyên state_dict.
    """
    return state_dict
