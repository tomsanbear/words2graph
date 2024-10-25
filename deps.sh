#!/bin/bash

# Flag to check if -y option is provided
AUTO_YES=false

# Parse command line arguments
while getopts ":y" opt; do
  case ${opt} in
    y )
      AUTO_YES=true
      ;;
    \? )
      echo "Invalid option: $OPTARG" 1>&2
      exit 1
      ;;
  esac
done

# Function to ask for user confirmation
ask_install() {
    if $AUTO_YES; then
        return 0
    fi
    read -p "Do you want to install $1? (y/n): " choice
    case "$choice" in 
        y|Y ) return 0;;
        n|N ) echo "Exiting as $1 is required but not installed."; exit 1;;
        * ) echo "Invalid input. Please enter y or n."; ask_install "$1";;
    esac
}

# check if platform is osx
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "This script is only supported on macOS at the moment. Please submit a pull request on GitHub if you want to add support for other platforms."
    exit 1
fi

# install brew if not already installed
if ! command -v brew &> /dev/null; then
    ask_install "Homebrew"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# install pyenv if not already installed
if ! command -v pyenv &> /dev/null; then
    ask_install "pyenv"
    brew install pyenv
    if ! grep -q "alias brew='env PATH=\"\${PATH//\$(pyenv root)\/shims:/}\" brew'" ~/.zshrc; then
        echo "alias brew='env PATH=\"\${PATH//\$(pyenv root)\/shims:/}\" brew'" >> ~/.zshrc
    fi
fi

# install huggingface-cli if not already installed
if ! command -v huggingface-cli &> /dev/null; then
    ask_install "huggingface-cli"
    brew install huggingface-cli
fi

# install ollama if not already installed
if ! command -v ollama &> /dev/null; then
    ask_install "ollama"
    brew install ollama
    ollama pull llama3.1:latest
fi

# install and setup python packages
LDFLAGS="-L$(brew --prefix sqlite)/lib" CPPFLAGS="-I$(brew --prefix sqlite)/include" PYTHON_CONFIGURE_OPTS="--enable-loadable-sqlite-extensions" pyenv install -s
pyenv exec python -m venv env
env/bin/pip install --upgrade pip pip-tools
env/bin/pip-compile
env/bin/pip install -r requirements.txt
