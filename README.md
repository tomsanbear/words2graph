# words2graph

words2graph is a tool that processes PDF documents, extracts named entities and relationships, and visualizes them as a knowledge graph.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/tomsanbear/words2graph.git
   cd words2graph
   ```

2. Run the setup script:

   ```bash
   ./deps.sh
   ```

   This script will:

   - Install Homebrew (if not already installed)
   - Install pyenv
   - Install huggingface-cli
   - Install Ollama and pull the llama3.1:latest model
   - Set up a Python virtual environment and install required packages

   Note: The script currently supports macOS. For other platforms, you may need to install dependencies manually.

3. Activate the virtual environment:
   ```bash
   source env/bin/activate
   ```

## Getting Started

1. Ensure you have a PDF file you want to process.

2. Load the PDF into the database:

   ```bash
   python main.py load path/to/your/document.pdf
   ```

   This command will process the PDF, extract text, identify entities and relationships, and store them in the database.

3. Visualize the knowledge graph:

   ```bash
   python main.py visualize
   ```

   This will generate a file named `knowledge_graph.html` in your current directory.

4. Open `knowledge_graph.html` in a web browser to view the interactive knowledge graph.

## Usage

The main script provides two primary commands:

- `load`: Process a PDF and store its content in the database.

  ```bash
  python main.py load <path_to_pdf>
  ```

- `visualize`: Generate a visualization of the knowledge graph.
  ```bash
  python main.py visualize
  ```

## Notes

- The graph visualization shows entities as nodes and relationships as directed edges.
- Edge thickness represents the number of connections between entities.
- Hover over edges to see detailed information about the relationships.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)
