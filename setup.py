import setuptools

setuptools.setup(
    name='regional_deep_atrophy',
    version='0.1',
    license='Apache 2.0',
    description='Regional Deep Atrophy: A deep learning approach for longitudinal brain atrophy estimation and heatmap generation',
    url='',
    keywords=['deformation', 'registration', 'imaging', 'cnn', 'mri', 'longitudinal'],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'scikit-image',
        'h5py',
        'numpy',
        'scipy',
        'nibabel',
        'neurite',
    ]
)
