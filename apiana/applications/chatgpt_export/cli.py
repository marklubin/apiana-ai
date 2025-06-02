import argparse
import json

from apiana.batch.chatgpt.chatgpt_export_loader import ChatGPTExportLoader
from apiana.batch.chatgpt.chatgpt_export_processor import ChatGPTExportProcessor
from apiana.batch.summary.summary_generator import SummaryGenerator

export_loader = ChatGPTExportLoader()
export_processor = ChatGPTExportProcessor(export_loader)


def parse_conversations_menu():
    input_file = input("Enter input file path: ").strip()
    output_dir = input("Enter output directory: ").strip()
    print(
        f"Processing conversations with input file: {input_file} and output directory: {output_dir}"
    )
    export_processor.extract_convos_with_persist(input_file, output_dir)
    print("Finished processing conversations.")


def parse_conversations(input_file: str, output_dir: str):
    print(
        f"Processing conversations with input file: {input_file} and output directory: {output_dir}"
    )
    export_processor.extract_convos_with_persist(input_file, output_dir)
    print("Finished processing conversations.")


def generate_summaries(input_file: str, output: str):
    print(f"Generating summary for: {input_file}")
    with open(input_file) as f:
        data = json.load(f)
        print(f"JSON data: {data}")
        print("Calling LLM for summary generation...")
        generator = SummaryGenerator()
        result = generator.generate(json.dumps(data))
        print("Got response.")
        print("Response was:" + result)


def enrich_embeddings(input_file: str, output_dir: str):
    print(f"Enriching embeddings for: {input_file}, output to: {output_dir}")
    # TODO: Implement embedding enrichment
    print("Embedding enrichment not yet implemented.")


def interactive_menu():
    # CLI loop to select a tool from a menu
    while True:
        print("\n=== Apiana ChatGPT Export CLI ===")
        print("1. Parse Conversations")
        print("2. Generate Summaries")
        print("3. Enrich and store embeddings")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ").strip()

        if choice == "1":
            parse_conversations_menu()
        elif choice == "2":
            # Export summary
            print("Exporting summary...")
        elif choice == "3":
            # Enrich/Store data
            print("Enriching and storing data...")
        elif choice == "4":
            print("Exiting the CLI loop.")
            break
        else:
            print("Invalid choice. Please try again.")


def main():
    parser = argparse.ArgumentParser(
        description="Apiana ChatGPT Export CLI - Process ChatGPT conversation exports",
        epilog="If no subcommand is provided, the interactive menu will be shown.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Parse subcommand
    parse_parser = subparsers.add_parser("parse", help="Parse ChatGPT conversations")
    parse_parser.add_argument(
        "-i", "--input", required=True, help="Input JSON file path"
    )
    parse_parser.add_argument(
        "-o", "--output", required=True, help="Output directory path"
    )

    # Summary subcommand
    summary_parser = subparsers.add_parser(
        "summary", help="Generate conversation summaries"
    )
    summary_parser.add_argument("-i", "--input", required=True, help="Input file path")
    summary_parser.add_argument(
        "-o", "--output", required=True, help="Output directory path"
    )

    # Enrich subcommand
    enrich_parser = subparsers.add_parser(
        "enrich", help="Enrich conversations with embeddings"
    )
    enrich_parser.add_argument("-i", "--input", required=True, help="Input file path")
    enrich_parser.add_argument(
        "-o", "--output", required=True, help="Output directory path"
    )

    args = parser.parse_args()

    if args.command == "parse":
        parse_conversations(args.input, args.output)
    elif args.command == "summary":
        generate_summaries(args.input, args.output)
    elif args.command == "enrich":
        enrich_embeddings(args.input, args.output)
    else:
        # No subcommand provided, show interactive menu
        interactive_menu()


if __name__ == "__main__":
    main()
