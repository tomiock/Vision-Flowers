import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Convert JSONL descriptions to JSON for Web Viewer")
    parser.add_argument("--input_file", type=str, default="/data-net/storage2/datasets/OxfordF/generated_descriptions_7b.json", help="Input JSONL file")
    parser.add_argument("--output_file", type=str, default="/data-net/storage2/datasets/OxfordF/dataset_viewer_data.json", help="Output JSON file for web app")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found.")
        return

    data = []
    print(f"Reading {args.input_file}...")
    with open(args.input_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                # entry is {"filename.jpg": "description"}
                # We want a list of objects: [{"filename": "...", "description": "..."}]
                for filename, description in entry.items():
                    data.append({
                        "filename": filename,
                        "description": description
                    })
            except json.JSONDecodeError:
                continue
    
    print(f"Found {len(data)} entries.")
    
    print(f"Writing to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(data, f, indent=2)
        
    print("Done. You can now open dataset_viewer.html")

if __name__ == "__main__":
    main()
