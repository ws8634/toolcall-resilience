from setuptools import setup, find_packages

setup(
    name="toolcall-resilience",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
        ],
    },
    python_requires=">=3.11",
    entry_points={
        "console_scripts": [
            "toolcall-resilience=toolcall_resilience.__main__:main",
        ],
    },
    author="Toolcall Resilience Team",
    description="A resilient tool calling framework with validation, retries, and error handling",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
)
