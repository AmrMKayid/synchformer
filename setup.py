from setuptools import setup, find_packages

setup(
    name="synchformer",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.5.0",
        "torchaudio>=0.7.0",
        "torchvision>=0.8.0",
        "omegaconf>=2.0.0",
        "requests>=2.25.0",
    ],
    author="AmrMKayid",
    author_email="amrmkayid@gmail.com",
    description="Synchformer: Audio-Visual Synchronization Model",
    keywords="deep-learning, audio-visual, synchronization",
    url="https://github.com/AmrMKayid/synchformer",
) 