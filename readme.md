# MovieMaster

A machine learning-based movie recommendation system using TensorFlow and Flask.

## Prerequisites

- pyenv (for managing Python versions)
- Python 3.10.4
- pip

## Installation Guide

### 1. Install pyenv

**On macOS:**

```bash
# Install with Homebrew
brew install pyenv

# Add to your shell (for bash)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# For zsh, use ~/.zshrc instead of ~/.bashrc
```

**On Linux:**

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl

# Install pyenv
curl https://pyenv.run | bash

# Add to your shell
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
```

**On Windows:**
We recommend using WSL2 (Windows Subsystem for Linux) and following the Linux instructions.

### 2. Install Python 3.10.4

```bash
# Install Python 3.10.4 with pyenv
pyenv install 3.10.4

# Set it as your global version
pyenv global 3.10.4

# Verify installation
python --version  # Should show 3.10.4
```

### 3. Clone and Setup Project

```bash
# Clone the repository
git clone https://github.com/MadsRunge/MovieMaster.git
cd MovieMaster

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Add Dataset

Place the IMDB dataset file `imdb_top_1000(1).csv` in:

```
MovieMaster/app/data/imdb_top_1000(1).csv
```

(Contact repository owner for access to the dataset)

### 5. Run the Application

```bash
python run.py
```

The server will start at http://127.0.0.1:5000

## Testing the API

You can test the following endpoints:

1. Get all movies:

```
GET http://127.0.0.1:5000/api/movies
```

2. Get movie recommendations:

```
GET http://127.0.0.1:5000/api/recommend?title=Soul
```

3. Get specific movie details:

```
GET http://127.0.0.1:5000/api/movie/Soul
```

## Troubleshooting

If you encounter any issues:

1. Ensure you're using Python 3.10.4:

```bash
python --version
```

2. Verify pyenv is properly installed:

```bash
pyenv --version
```

3. Make sure all dependencies are installed:

```bash
pip list
```

4. Check if the dataset file is in the correct location and has the correct name

## Technologies Used

- Python 3.10.4
- Flask
- TensorFlow
- scikit-learn
- pandas
- NumPy

## License

[MIT License](LICENSE)
