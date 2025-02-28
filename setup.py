from setuptools import find_packages, setup

setup(
    name="reprs",
    version="0.0.1",
    author="Malcolm Sailor",
    author_email="malcolm.sailor@gmail.com",
    description="TODO",
    long_description="TODO",
    long_description_content_type="text/markdown",
    # TODO add time_shifter to requirements
    install_requires=["numpy", "pandas", "tqdm", "music_df"],
    extras_require={
        "midilike": ["time-shifter"],
        "tests": ["metricker", "pytest"],
    },
    url="TODO",
    project_urls={
        "Bug Tracker": "TODO",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
)
