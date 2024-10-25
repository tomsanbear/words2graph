import hashlib
import json
import logging
import os
import re
import sqlite3
import unicodedata
from collections import Counter
from typing import Callable

import click
import groq
import httpx
import magic
import networkx as nx
import ollama
import sqlite_vec
from dotenv import load_dotenv
from flair.data import Sentence
from flair.models import SequenceTagger
from langchain_text_splitters import RecursiveCharacterTextSplitter
from nltk.stem import WordNetLemmatizer
from pypdf import PdfReader
from pyvis.network import Network
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

# disable flair logs
flair_logger = logging.getLogger("flair")
flair_logger.setLevel(logging.ERROR)

ChatCompletionCallable = Callable[[list[dict[str, str]]], str]


def get_ollama_client() -> tuple[ollama.Client, ChatCompletionCallable]:
    # check if remote desktop host is available else fallback to local ollama
    response = httpx.get("http://192.168.2.125:11434")
    if response.status_code == 200:
        client = ollama.Client(host="http://192.168.2.125:11434")
    else:
        client = ollama.Client()

    def chat_completion(messages: list[dict[str, str]]) -> str:
        return client.chat(messages=messages, model="llama3.1:latest", stream=False)[
            "message"
        ]["content"]

    return (client, chat_completion)


def get_groq_client() -> tuple[groq.Client, ChatCompletionCallable]:
    client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))

    models = [
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
    ]

    def chat_completion(messages: list[dict[str, str]]) -> str:
        for model in models:
            try:
                return (
                    client.chat.completions.create(
                        messages=messages, model=model, timeout=5
                    )
                    .choices[0]
                    .message.content
                )
            except Exception:
                pass
        raise Exception("Failed to generate chat completion")

    return (client, chat_completion)


def chat_completion(messages: list[dict[str, str]]) -> str:
    """
    Chat completion that uses a fallback mechanism to try multiple models if the first one fails.
    """
    errors = []

    key = hashlib.sha256(json.dumps(messages).encode()).hexdigest()
    connection = sqlite3.connect("chat_completion_cache.db")
    cursor = connection.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS chat_completion_cache (key TEXT PRIMARY KEY, response TEXT)"
    )
    cursor.execute("SELECT response FROM chat_completion_cache WHERE key = ?", (key,))
    cached_response = cursor.fetchone()
    if cached_response and cached_response[0]:
        return cached_response[0]

    for _, chat_completion in [
        get_groq_client(),
        get_ollama_client(),
    ]:
        try:
            completion = chat_completion(messages)
            cursor.execute(
                "INSERT INTO chat_completion_cache (key, response) VALUES (?, ?)",
                (key, completion),
            )
            connection.commit()
            connection.close()
            return completion
        except Exception as e:
            errors.append(e)

    raise Exception("Failed to generate chat completion", errors)


def chunk_text(text: str):
    """
    Chunks text into smaller chunks, returning the chunk, start index, and end index.
    """

    # NOTE: if you change these settings, you should drop the contents of the document chunks table so you don't duplicate chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )

    offsets = []
    for chunk in splitter.split_text(text):
        start_index = text.index(chunk)
        end_index = start_index + len(chunk)
        offsets.append((chunk, start_index, end_index))

    return [offset for offset in offsets if offset[0]]


def get_db_connection():
    """
    Returns a connection to the sqlite database.
    Enables use of the sqlite_vec extension to perform vector search across embeddings.
    """
    conn = sqlite3.connect("main.db")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def setup_db(clean: bool = False):
    """
    Sets up the database. Provisions tables, indices and anything else that is needed.
    """
    with get_db_connection() as conn:
        if clean:
            print("Cleaning database")
            conn.execute("DROP TABLE IF EXISTS documents")
            conn.execute("DROP TABLE IF EXISTS document_chunks")
            conn.execute("DROP TABLE IF EXISTS graph_nodes")
            conn.execute("DROP TABLE IF EXISTS graph_edges")
            conn.commit()

        # Migration 1
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                file_name TEXT NOT NULL, 
                content_hash TEXT UNIQUE NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                document_id INTEGER NOT NULL, 
                text TEXT NOT NULL, 
                embedding BLOB NOT NULL, 
                start_index INTEGER NOT NULL, 
                end_index INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS unq_document_chunks ON document_chunks (document_id, start_index, end_index)"
        )
        conn.commit()

        # Migration 2
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS graph_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                document_chunk_id INTEGER NOT NULL, 
                label TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS unq_graph_nodes ON graph_nodes (document_chunk_id, label)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS graph_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                source_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                label TEXT NOT NULL,
                FOREIGN KEY (source_id) REFERENCES graph_nodes(id),
                FOREIGN KEY (target_id) REFERENCES graph_nodes(id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS unq_graph_edges ON graph_edges (source_id, target_id, label)"
        )
        conn.commit()


def insert_document(file_name: str, content_hash: str) -> int:
    """
    Inserts a document into the database.
    """
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO documents (file_name, content_hash) VALUES (?, ?) ON CONFLICT DO NOTHING",
        (file_name, content_hash),
    )
    conn.commit()
    return conn.execute(
        "SELECT id FROM documents WHERE content_hash = ?", (content_hash,)
    ).fetchone()[0]


def get_text_embeddings(texts: list[str]):
    """
    Gets the embeddings for a list of texts.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts, convert_to_tensor=True).tolist()


def insert_chunk(
    document_id: int,
    text: str,
    start_index: int,
    end_index: int,
    embedding: list[float],
):
    """
    Inserts a chunk into the database.
    """
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO document_chunks (
            document_id, text, start_index, end_index, embedding) 
        VALUES (?, ?, ?, ?, ?)""",
            (document_id, text, start_index, end_index, json.dumps(embedding)),
        )
        conn.commit()


def get_named_entities(text: str) -> list[tuple[str, int, int]]:
    """
    Gets the named entities from a text.
    """
    english_tagger = SequenceTagger.load("flair/ner-english")
    french_tagger = SequenceTagger.load("flair/ner-french")
    sentence = Sentence(text)
    english_tagger.predict(sentence)
    french_tagger.predict(sentence)
    english_entities = [
        (entity.text, entity.start_position, entity.end_position)
        for entity in sentence.get_spans("ner")
    ]
    french_entities = [
        (entity.text, entity.start_position, entity.end_position)
        for entity in sentence.get_spans("ner")
    ]
    return english_entities + french_entities


def get_document_chunks(document_id: int) -> list[tuple[int, str, int, int]]:
    """
    Gets the chunks for a document.
    """
    with get_db_connection() as conn:
        return conn.execute(
            "SELECT id, text, start_index, end_index FROM document_chunks WHERE document_id = ?",
            (document_id,),
        ).fetchall()


def get_edges(text: str, entities: list[str]) -> list[tuple[str, str, str]]:
    """
    For a given body of text, and a list of entities, returns a list of edges that connect them.

    This process is performed using an LLM so it may miss or incorrectly identify edges.
    """
    if not entities or len(entities) < 2:
        return []

    system_prompt = """You are an expert at identifying relationships between entities in a body of text.

You do this by using the provided list of entities from the user, and the body of text, to first explain your answer, and then outputting a list of edges that connect the entities.

**Important Guidelines:**
- Only relationships that are explicitly stated in the text should be included, do not make assumptions or invent relationships even if they seem obvious.
- You should provide your list of edges as valid JSON in the format described below.
- When writing your label/relationship, be as concise as possible without losing meaning. Try to make the label flow as if the source was prefixed, and the target was suffixed.
- When providing the source and target, you should primarily use the entities provided by the user, however you can also use any other entities that are **DIRECTLY** mentioned in the text.
- ONLY respond with JSON and nothing else.

**JSON Format:**
- The list of edges should be an array of objects, where each object has three properties: `source`, `target`, and `label`.
- `source` and `target` should be the text of the entities that are connected by the edge.
- `label` should be a short description of the relationship between the two entities.
- When parsing the JSON, the following python code will be used, so please ensure your output will be correctly parsed:
```python
start_index = raw_content.find("[")
end_index = raw_content.rfind("]")
json_text = raw_content[start_index : end_index + 1] or "[]"
```

**Example Answer:**
```json
[{"source": "<entity1>", "target": "<entity2>", "label": "<relationship>"}, ...]
```"""

    user_prompt = f"""Given the following body of text,
```plaintext
{text}
```

And the following list of entities,
```json
{json.dumps(entities)}
```

Provide a list of edges that connect the entities. Make sure to follow the instructions when picking relationships and formatting your answer, your list of edges must be in JSON format."""

    additional_messages = []
    for i in range(3):
        raw_content = chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        # find the index of the first occurrence of the '[' character, and the last occurrence of the ']' character
        start_index = raw_content.find("[")
        end_index = raw_content.rfind("]")
        json_text = raw_content[start_index : end_index + 1] or "[]"

        try:
            return json.loads(json_text)
        except json.JSONDecodeError as error:
            print(f"Parsing JSON failed on attempt {i}: {error}\n\n{raw_content}")
            additional_messages.append({"role": "assistant", "content": raw_content})
            additional_messages.append(
                {
                    "role": "user",
                    "content": f"I'm sorry, but when extracting the following json\n```json\n{json_text}```\nI received the following error:\n{error}\n\nPlease reformat your original answer to fix the error.",
                }
            )

    raise Exception("Failed to get valid JSON from LLM")


def insert_graph_node(document_chunk_id: int, label: str) -> int:
    """
    Inserts a graph node into the database.
    """
    with get_db_connection() as conn:
        existing_node = conn.execute(
            "SELECT id FROM graph_nodes WHERE document_chunk_id = ? AND label = ?",
            (document_chunk_id, label),
        ).fetchone()
        if existing_node:
            return existing_node[0]

        conn.execute(
            "INSERT INTO graph_nodes (document_chunk_id, label) VALUES (?, ?)",
            (document_chunk_id, label),
        )
        conn.commit()
        return conn.execute(
            "SELECT id FROM graph_nodes WHERE document_chunk_id = ? AND label = ?",
            (document_chunk_id, label),
        ).fetchone()[0]


def insert_graph_edge(source_id: int, target_id: int, label: str) -> int:
    """
    Inserts a graph edge into the database.
    """
    with get_db_connection() as conn:
        existing_edge = conn.execute(
            "SELECT id FROM graph_edges WHERE source_id = ? AND target_id = ? AND label = ?",
            (source_id, target_id, label),
        ).fetchone()
        if existing_edge:
            return existing_edge[0]

        conn.execute(
            "INSERT INTO graph_edges (source_id, target_id, label) VALUES (?, ?, ?)",
            (source_id, target_id, label),
        )
        conn.commit()
        return conn.execute(
            "SELECT id FROM graph_edges WHERE source_id = ? AND target_id = ? AND label = ?",
            (source_id, target_id, label),
        ).fetchone()[0]


@click.group()
@click.option(
    "--clean", is_flag=True, help="Cleans the database before running the command"
)
def cli(clean):
    # setup the database prior to any operation
    setup_db(clean)


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, readable=True))
def load(path):
    """
    Loads a document into the database. Populates chunks, entities and anything else required for querying.
    """

    if not magic.from_file(path, mime=True).startswith("application/pdf"):
        raise click.UsageError("Input file is not a PDF")

    # insert the document into the database
    with open(path, "rb") as f:
        content_hash = hashlib.md5(f.read()).hexdigest()
    document_id = insert_document(path, content_hash)
    if document_id is None:
        raise click.UsageError("Failed to insert document into database")

    # extract each page of text
    reader = PdfReader(path)
    page_texts = [
        page.extract_text() for page in tqdm(reader.pages, desc="Processing PDF pages")
    ]

    # drop existing chunks
    # TODO: make this configurable instead? needs to be here if we adjust the size of the chunks
    with get_db_connection() as conn:
        conn.execute(
            "DELETE FROM document_chunks WHERE document_id = ?", (document_id,)
        )
        conn.commit()

    # split each page into chunks of text and insert into the database
    for page_text in tqdm(page_texts, desc="Processing PDF pages"):
        chunks = [
            chunk for chunk in chunk_text(page_text) if chunk[0] and chunk[0] != ""
        ]
        chunk_embeddings = get_text_embeddings([chunk[0] for chunk in chunks])
        for chunk, embedding in zip(chunks, chunk_embeddings):
            insert_chunk(document_id, chunk[0], chunk[1], chunk[2], embedding)

    # run NER on each chunk
    chunks = get_document_chunks(document_id)
    for chunk in tqdm(chunks, desc="Processing chunks"):
        entities = get_named_entities(chunk[1])
        edges = get_edges(chunk[1], [entity[0] for entity in entities])
        for edge in edges:
            print(f"Found edge: {edge['source']} -> {edge['target']} ({edge['label']})")
            source_id = insert_graph_node(chunk[0], clean_node(edge["source"]))
            target_id = insert_graph_node(chunk[0], clean_node(edge["target"]))
            insert_graph_edge(source_id, target_id, edge["label"])


def get_graph_edges() -> list[tuple[str, str, str]]:
    """
    Gets all the edges from the database.

    Returns a tuple of (source_text, target_text, label).
    """
    with get_db_connection() as conn:
        return conn.execute(
            """
            SELECT
                source_node.label as source_text,
                target_node.label as target_text,
                graph_edges.label
            FROM graph_edges
            INNER JOIN graph_nodes as source_node ON graph_edges.source_id = source_node.id
            INNER JOIN graph_nodes as target_node ON graph_edges.target_id = target_node.id
            """
        ).fetchall()


def clean_node(node: str) -> str:
    """
    Clean the node by doing the following:
    - Remove case sensitivity
    - Remove punctuation
    - Lemmatize
    - Remove diacritical marks
    """
    # Remove case sensitivity
    node = node.lower()

    # Remove diacritical marks
    node = "".join(
        c for c in unicodedata.normalize("NFD", node) if unicodedata.category(c) != "Mn"
    )

    # Remove punctuation
    node = re.sub(r"[^\w\s]", "", node)

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    node = lemmatizer.lemmatize(node)

    return node


@cli.command()
def visualize():
    """
    Visualizes the graph using the pyvis library with directed edges.
    """
    edges = get_graph_edges()
    nx_graph = nx.DiGraph()  # Change to DiGraph for directed edges

    # Count the number of connections between each pair of nodes and collect labels
    edge_counts = Counter()
    edge_labels = {}
    for source, target, label in edges:
        key = (source, target)  # Keep original direction
        edge_counts[key] += 1
        if key not in edge_labels:
            edge_labels[key] = set()
        edge_labels[key].add(label)

    # Add edges to the graph with width based on connection count
    for (source, target), count in edge_counts.items():
        # Calculate edge width (you can adjust the scaling factor as needed)
        width = 1 + (
            count * 2
        )  # Example: base width of 1, increasing by 2 for each additional connection
        labels = "\n".join(
            f"- {label}" for label in sorted(edge_labels[(source, target)])
        )
        title = f"Connections: {count}\n{labels}"
        nx_graph.add_edge(
            source,
            target,
            width=width,
            title=title,
        )

    network = Network(
        height="750px",
        width="100%",
        bgcolor="#222222",
        font_color="white",
        directed=True,  # Enable directed edges
    )
    network.from_nx(nx_graph)

    # Customize node appearance
    for node in network.nodes:
        node["size"] = 20  # Adjust node size as needed
        node["title"] = node["id"]  # Use node label as hover text

    # Customize edge appearance
    for edge in network.edges:
        edge["color"] = "#FFFFFF"  # Set edge color to white
        edge["arrows"] = "to"  # Add arrows to show direction

    network.show("knowledge_graph.html", notebook=False)


if __name__ == "__main__":
    cli()
    cli()
