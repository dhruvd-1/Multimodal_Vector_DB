from setuptools import setup, find_packages

setup(
    name="multimodal-vector-db",
    version="0.1.0",
    description="Multimodal Vector Database with Cross-Modal Search",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "faiss-cpu>=1.7.4",
        "numpy>=1.24.0",
        "Pillow>=9.5.0",
        "opencv-python>=4.7.0",
        "librosa>=0.10.0",
    ],
    python_requires=">=3.8",
)
