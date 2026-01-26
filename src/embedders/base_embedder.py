from abc import ABC, abstractmethod
import numpy as np
from typing import Union, List, Optional, Dict, Any
from enum import Enum
import platform
import asyncio
from concurrent.futures import ThreadPoolExecutor


class DeviceType(Enum):
    """Supported device types for inference."""
    CPU = 'cpu'
    CUDA = 'cuda'
    MPS = 'mps'  # Apple Silicon
    # Mobile-specific devices
    MOBILE_CPU = 'mobile_cpu'
    MOBILE_GPU = 'mobile_gpu'
    NNAPI = 'nnapi'  # Android Neural Networks API
    COREML = 'coreml'  # iOS Core ML
    METAL = 'metal'  # iOS/macOS Metal


class QuantizationType(Enum):
    """Quantization types for model optimization."""
    NONE = 'none'
    FP16 = 'fp16'
    INT8 = 'int8'
    INT4 = 'int4'
    DYNAMIC = 'dynamic'


class ModelFormat(Enum):
    """Supported model formats."""
    PYTORCH = 'pytorch'
    ONNX = 'onnx'
    TFLITE = 'tflite'
    COREML = 'coreml'
    TORCHSCRIPT = 'torchscript'


class BaseEmbedder(ABC):
    """
    Abstract base class for all embedders with mobile support.

    Provides unified interface for embedding generation across different
    modalities with support for mobile deployment, memory management,
    quantization, and async operations.
    """

    def __init__(
        self,
        model_name: str,
        device: str = 'cpu',
        quantization: str = 'none',
        model_format: str = 'pytorch',
        max_batch_size: Optional[int] = None,
        memory_limit_mb: Optional[int] = None,
        enable_async: bool = False
    ):
        """
        Initialize the embedder with mobile-aware configuration.

        Args:
            model_name: Name/path of the model to use
            device: Device to run inference on ('cpu', 'cuda', 'mps', 'mobile_cpu', etc.)
            quantization: Quantization type ('none', 'fp16', 'int8', 'int4', 'dynamic')
            model_format: Model format ('pytorch', 'onnx', 'tflite', 'coreml', 'torchscript')
            max_batch_size: Maximum batch size for mobile memory constraints
            memory_limit_mb: Memory limit in MB for mobile devices
            enable_async: Enable async embedding operations
        """
        self.model_name = model_name
        self.device = device
        self.quantization = QuantizationType(quantization) if isinstance(quantization, str) else quantization
        self.model_format = ModelFormat(model_format) if isinstance(model_format, str) else model_format
        self.max_batch_size = max_batch_size
        self.memory_limit_mb = memory_limit_mb
        self.enable_async = enable_async

        self.model = None
        self.embedding_dim: int = 0
        self._is_loaded: bool = False
        self._executor: Optional[ThreadPoolExecutor] = None

        # Mobile-specific settings
        self._is_mobile = self._detect_mobile_environment()
        self._apply_mobile_defaults()

    def _detect_mobile_environment(self) -> bool:
        """Detect if running on a mobile device or mobile-like environment."""
        system = platform.system().lower()
        machine = platform.machine().lower()

        # Check for mobile indicators
        mobile_indicators = [
            'android' in system,
            'ios' in system,
            'iphone' in machine,
            'ipad' in machine,
            'arm' in machine and system == 'linux',  # Raspberry Pi / ARM devices
            self.device in ('mobile_cpu', 'mobile_gpu', 'nnapi', 'coreml', 'metal'),
        ]

        return any(mobile_indicators)

    def _apply_mobile_defaults(self):
        """Apply sensible defaults for mobile environments."""
        if self._is_mobile:
            # Apply conservative defaults for mobile
            if self.max_batch_size is None:
                self.max_batch_size = 4  # Small batch size for mobile
            if self.memory_limit_mb is None:
                self.memory_limit_mb = 256  # Conservative memory limit
            if self.quantization == QuantizationType.NONE:
                self.quantization = QuantizationType.FP16  # Default to FP16 on mobile

    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._is_loaded and self.model is not None

    @property
    def is_mobile(self) -> bool:
        """Check if running in mobile environment."""
        return self._is_mobile

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the embedding model.

        Should handle:
        - Model loading based on model_format
        - Quantization application
        - Device placement
        - Setting self._is_loaded = True on success
        """
        pass

    def unload_model(self) -> None:
        """
        Unload the model to free memory.

        Critical for mobile devices with limited RAM.
        """
        if self.model is not None:
            # Clear model reference
            del self.model
            self.model = None
            self._is_loaded = False

            # Force garbage collection for mobile
            import gc
            gc.collect()

            # Clear CUDA cache if applicable
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    @abstractmethod
    def embed(self, content: Union[str, object]) -> np.ndarray:
        """
        Generate embedding vector for single input.

        Args:
            content: Input content (text, image path, etc.)

        Returns:
            Embedding vector as numpy array (L2-normalized)
        """
        pass

    @abstractmethod
    def batch_embed(self, contents: List, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Generate embeddings for batch of inputs.

        Args:
            contents: List of inputs
            batch_size: Override batch size (respects max_batch_size limit)

        Returns:
            Matrix of embedding vectors
        """
        pass

    async def async_embed(self, content: Union[str, object]) -> np.ndarray:
        """
        Async version of embed for non-blocking mobile UI.

        Args:
            content: Input content

        Returns:
            Embedding vector as numpy array
        """
        if not self.enable_async:
            return self.embed(content)

        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.embed, content)

    async def async_batch_embed(self, contents: List, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Async version of batch_embed for non-blocking mobile UI.

        Args:
            contents: List of inputs
            batch_size: Override batch size

        Returns:
            Matrix of embedding vectors
        """
        if not self.enable_async:
            return self.batch_embed(contents, batch_size)

        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.batch_embed(contents, batch_size)
        )

    def get_embedding_dim(self) -> int:
        """Get the dimensionality of embeddings."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.embedding_dim

    def get_effective_batch_size(self, requested_batch_size: Optional[int] = None) -> int:
        """
        Get effective batch size respecting mobile constraints.

        Args:
            requested_batch_size: Requested batch size

        Returns:
            Effective batch size within limits
        """
        if requested_batch_size is None:
            requested_batch_size = 32  # Default

        if self.max_batch_size is not None:
            return min(requested_batch_size, self.max_batch_size)

        return requested_batch_size

    def get_config(self) -> Dict[str, Any]:
        """Get current embedder configuration."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'quantization': self.quantization.value,
            'model_format': self.model_format.value,
            'max_batch_size': self.max_batch_size,
            'memory_limit_mb': self.memory_limit_mb,
            'embedding_dim': self.embedding_dim,
            'is_loaded': self._is_loaded,
            'is_mobile': self._is_mobile,
            'enable_async': self.enable_async,
        }

    def get_memory_usage_mb(self) -> float:
        """
        Estimate current memory usage in MB.

        Useful for mobile memory management.
        """
        if not self._is_loaded:
            return 0.0

        try:
            import torch
            if hasattr(self.model, 'parameters'):
                param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
                buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
                return (param_size + buffer_size) / (1024 * 1024)
        except (ImportError, AttributeError):
            pass

        return 0.0

    def check_memory_available(self, required_mb: float) -> bool:
        """
        Check if required memory is available within limits.

        Args:
            required_mb: Required memory in MB

        Returns:
            True if memory is available
        """
        if self.memory_limit_mb is None:
            return True

        current_usage = self.get_memory_usage_mb()
        return (current_usage + required_mb) <= self.memory_limit_mb

    def warmup(self, sample_input: Optional[Union[str, object]] = None) -> None:
        """
        Warm up the model with a sample inference.

        Useful for mobile to ensure model is fully loaded and optimized
        before actual use.

        Args:
            sample_input: Optional sample input for warmup
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Subclasses should implement with appropriate sample input
        pass

    def __enter__(self):
        """Context manager entry - load model."""
        if not self._is_loaded:
            self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - unload model for mobile memory management."""
        if self._is_mobile:
            self.unload_model()
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None
        return False

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name='{self.model_name}', "
            f"device='{self.device}', "
            f"quantization='{self.quantization.value}', "
            f"is_loaded={self._is_loaded}, "
            f"is_mobile={self._is_mobile})"
        )
