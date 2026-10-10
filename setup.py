from setuptools import setup, Extension, find_packages
import platform
import sys

# === Platform-specific build flags ===
system = platform.system()
arch = platform.machine()
extra_compile_args = []

if system == 'Darwin':
    if arch == 'arm64':
        extra_compile_args.extend(['-arch', 'arm64'])
    elif arch == 'x86_64':
        extra_compile_args.extend(['-arch', 'x86_64'])

# === Define C Extensions ===
ext_modules = [
    Extension(
        'shortstop.training.negative_set',
        sources=['src/shortstop/training/negative_set.c'],
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        'shortstop.training.feature_extractor',
        sources=['src/shortstop/training/feature_extractor.c'],
        extra_compile_args=extra_compile_args,
    ),
]

# === Install requirements ===
# NOTE: tensorflow_macos is replaced with a conditional version below
tf_dep = 'tensorflow-macos==2.11.0' if system == 'Darwin' else 'tensorflow==2.11.0'

setup(
    name='shortstop',
    version='1.1.0',
    description='ShortStop: A classifier for translated smORFs',
    author='Brendan Miller',
    url='https://github.com/brendan-miller-salk/ShortStop',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    include_package_data=True,
    package_data={
        'shortstop': [
            'demo_data/*',
            'standard_prediction_model/*',
        ],
    },
    install_requires=[
        'Bio==1.6.2',
        'biopython==1.80',
        'eli5==0.13.0',
        'joblib==1.2.0',
        'keras==2.11.0',
        'matplotlib==3.6.3',
        'numpy==1.26.4',
        'pandas==2.2.2',
        'plotly==5.13.0',
        'protlearn==0.0.3',
        'scikit_learn==1.2.2',
        'seaborn==0.13.2',
        tf_dep,
        'umap==0.1.1',
        'umap_learn==0.5.3',
        'xgboost==1.7.3',
        'setuptools<81',
    ],
    python_requires='>=3.7,<3.11',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Salk License',
    ],
    entry_points={
        'console_scripts': [
            'shortstop = shortstop.cli:main',
        ],
    },
    ext_modules=ext_modules,
    zip_safe=False,
)