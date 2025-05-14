import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
CSV_PATH = "products.csv"

# Load product data
products = pd.read_csv(CSV_PATH)

# LangChain setup
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=OPENAI_API_KEY)
loader = CSVLoader(file_path=CSV_PATH)
documents = loader.load()
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_documents(documents, embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Prompt templates
system_message = SystemMessagePromptTemplate.from_template(
    "You are a helpful assistant for product recommendations. Use the context to answer questions."
)
human_message = HumanMessagePromptTemplate.from_template("{question}")
prompt_template = ChatPromptTemplate.from_messages([system_message, human_message])

class State(TypedDict):
    messages: List[dict]
    user_input: str
    response: str
    product: dict
    recommendations: List[dict]

def chat_node(state: State) -> State:
    user_query = state["user_input"]
    q = user_query.strip().lower()
    # Handle product list queries
    if (
        q == "products" or
        "what are the products" in q or
        "what products are available" in q or
        "list products" in q or
        "show products" in q
    ):
        names = products['title'].tolist()
        return {
            **state,
            "response": f"Here are the available products: {', '.join(names)}",
            "product": {},
            "recommendations": []
        }
    # Try to match a product by title (exact or partial)
    matched = products[products['title'].str.lower().str.contains(user_query.lower())]
    if not matched.empty:
        prod = matched.iloc[0].to_dict()
        recs = products[(products['category'] == prod['category']) & (products['title'] != prod['title'])]
        rec_list = recs[['title', 'image_path']].rename(
            columns={'title': 'name', 'image_path': 'image_url'}
        ).to_dict(orient='records')
        prod_for_frontend = {
            "name": prod["title"],
            "description": prod["description"],
            "price": prod["price"],
            "color": prod["color"],
            "image_url": prod["image_path"]
        }
        return {
            **state,
            "response": "",
            "product": prod_for_frontend,
            "recommendations": rec_list
        }
    # RAG fallback for semantic queries
    docs = retriever.get_relevant_documents(user_query)
    if docs:
        # Try to extract the top relevant product
        top_doc = docs[0].page_content
        # Find the product in the DataFrame
        for _, row in products.iterrows():
            if row['title'].lower() in top_doc.lower():
                prod = row.to_dict()
                recs = products[(products['category'] == prod['category']) & (products['title'] != prod['title'])]
                rec_list = recs[['title', 'image_path']].rename(
                    columns={'title': 'name', 'image_path': 'image_url'}
                ).to_dict(orient='records')
                prod_for_frontend = {
                    "name": prod["title"],
                    "description": prod["description"],
                    "price": prod["price"],
                    "color": prod["color"],
                    "image_url": prod["image_path"]
                }
                return {
                    **state,
                    "response": "",
                    "product": prod_for_frontend,
                    "recommendations": rec_list
                }
    # If nothing found, fallback to LLM
    context = "\n".join([doc.page_content for doc in docs])
    full_prompt = prompt_template.format_prompt(question=user_query + "\nContext:\n" + context)
    response = llm.invoke(full_prompt.to_messages())
    return {
        **state,
        "response": response.content,
        "product": {},
        "recommendations": []
    }

# Build LangGraph workflow
graph = StateGraph(State)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
memory = MemorySaver()
app_graph = graph.compile(checkpointer=memory)

app = Flask(__name__)
CORS(app)

@app.route("/products", methods=["GET"])
def list_products():
    return jsonify(products[['title']].rename(columns={'title': 'name'}).to_dict(orient='records'))

@app.route("/product", methods=["GET", "POST"])
def product_info():
    if request.method == "POST":
        user_query = request.json.get("query", "")
    else:
        user_query = request.args.get("query", "")
    initial_state = {
        "messages": [{"role": "user", "content": user_query}],
        "user_input": user_query,
        "response": "",
        "product": {},
        "recommendations": []
    }
    config = {"configurable": {"thread_id": "user1"}}
    output = app_graph.invoke(initial_state, config)
    if output.get("product") and output["product"].get("name"):
        return jsonify({
            "product": output["product"],
            "recommendations": output.get("recommendations", [])
        })
    else:
        return jsonify({"response": output["response"]})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
