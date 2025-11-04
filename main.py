#!/usr/bin/env python3
"""
Main entry point for Feature Selection using Genetic Algorithms
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='Feature Selection using Genetic Algorithms'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Web app command
    web_parser = subparsers.add_parser('web', help='Run web application')
    web_parser.add_argument('--host', default='0.0.0.0', help='Host address')
    web_parser.add_argument('--port', type=int, default=5000, help='Port number')
    web_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # Example command
    example_parser = subparsers.add_parser('example', help='Run example usage')
    
    args = parser.parse_args()
    
    if args.command == 'web':
        from src.web_app.app import app
        print(f"Starting web application on http://{args.host}:{args.port}")
        # Disable reloader to avoid clearing in-memory progress storage during development
        app.run(debug=args.debug, host=args.host, port=args.port, use_reloader=False, threaded=True)
    
    elif args.command == 'example':
        import subprocess
        example_path = Path(__file__).parent / 'examples' / 'example_usage.py'
        subprocess.run([sys.executable, str(example_path)])
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()

