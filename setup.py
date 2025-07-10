from setuptools import setup, find_packages

setup(
    name='shortstop',
    version='1.0.0',
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
            'training/*.so',  # ✅ Include precompiled binaries
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
        'tensorflow_macos==2.11.0',
        'umap==0.1.1',
        'umap_learn==0.5.3',
        'xgboost==1.7.3',
        'setuptools<81'
    ],
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Salk License',
    ],
    entry_points={
        'console_scripts': [
            'shortstop = shortstop.cli:main',
        ],
    },
    zip_safe=False,
)