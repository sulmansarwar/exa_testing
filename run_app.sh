#!/bin/bash

echo "🚀 Starting Exa + Claude Comparison App..."
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "Please create a .env file with:"
    echo "  EXA_API_KEY=your-key"
    echo "  ANTHROPIC_API_KEY=your-key"
    exit 1
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

echo "✅ Starting Streamlit app..."
echo "   App will open at: http://localhost:8501"
echo ""

streamlit run app.py
