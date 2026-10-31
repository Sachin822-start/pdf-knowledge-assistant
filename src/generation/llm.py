
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:1.5b"


def generate_answer(
    question: str,
    context: str,
    temperature: float = 0.1
) -> str:

    prompt = f"""
You are a technical assistant answering questions using retrieved PDF passages.

STRICT RULES:
1. Answer only using information explicitly supported by the CONTEXT.
2. Do not introduce assumptions or unsupported technical claims.
3. Distinguish routing configuration from security permissions.
4. A public subnet having a route to an Internet Gateway does NOT,
   by itself, mean unsolicited inbound traffic is allowed.
5. If the context does not contain enough information, say:
   "The provided documents do not contain enough information to answer this fully."
6. Keep the answer clear and concise, preferably 3-5 sentences.
7. Do not invent PDF filenames or page numbers.

QUESTION:
{question}

CONTEXT:
{context}

Answer:
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 120
        }
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=180
        )

        response.raise_for_status()

        result = response.json()

        return result.get("response", "").strip()

    except requests.exceptions.RequestException as error:
        return f"Error communicating with Ollama: {error}"


if __name__ == "__main__":

    test_context = """
    SOURCE: sample.pdf
    PAGE: 43

    A NAT Gateway allows instances in a private subnet
    to initiate outbound connections to the internet.
    It prevents the internet from initiating connections
    to those instances.
    """

    question = "What is the purpose of a NAT Gateway?"

    answer = generate_answer(question, test_context)

    print("\nQUESTION:")
    print(question)

    print("\nGENERATED ANSWER:")
    print(answer)
