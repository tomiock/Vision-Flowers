import http.server
import socketserver
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Serve the Dataset Viewer")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    parser.add_argument("--directory", type=str, default="/data-net/storage2/datasets/OxfordF/", help="Directory to serve (should contain dataset_viewer.html and images/)")
    args = parser.parse_args()

    # Change to the target directory
    os.chdir(args.directory)
    
    # Define the handler to serve files
    Handler = http.server.SimpleHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", args.port), Handler) as httpd:
            print(f"Serving HTTP on 0.0.0.0 port {args.port} (http://0.0.0.0:{args.port}/) ...")
            print(f"Access this from your laptop using: http://<server-ip>:{args.port}/web.html")
            print("Press Ctrl+C to stop.")
            httpd.serve_forever()
    except OSError as e:
        print(f"Error: Could not bind to port {args.port}. It might be in use.")
        print(f"Try a different port using --port <port_number>")

if __name__ == "__main__":
    main()
