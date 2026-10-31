from src.generation.llm import generate_answer


def main():
    question = "How does a NAT Gateway work in AWS?"

    context = """
    A NAT Gateway allows instances in a private subnet to initiate
    outbound connections to the internet while preventing unsolicited
    inbound connections from the internet.

    It is deployed in a public subnet and uses an Elastic IP address.
    The private subnet route table sends internet-bound traffic to
    the NAT Gateway. The NAT Gateway then forwards the traffic to
    the internet through the Internet Gateway.
    """

    print("Question:")
    print(question)

    print("\nGenerating answer using Ollama...")
    answer = generate_answer(question, context)

    print("\nGenerated Answer")
    print("=" * 60)
    print(answer)


if __name__ == "__main__":
    main()