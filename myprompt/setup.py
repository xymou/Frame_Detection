import setuptools



with open('README.md', 'r') as f:
    setuptools.setup(
        name = 'myprompt',
        version = '1.0',
        description = "openprompt framework for my projects",
        author = '',
        author_email = '',
        license="Apache",
        keywords = ['PLM', 'prompt'],
        python_requires=">=3.6.0",
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Intended Audience :: Researchers",
            "Intended Audience :: Students",
            "Intended Audience :: Developers",

        ]
    )