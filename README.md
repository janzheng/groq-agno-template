# Groq + Agno AI Agent Template

[Agno](agno.com) is a lightweight framework for building multi-modal Agents. Its easy to use, extremely fast and supports multi-modal inputs and outputs.

With Groq & Agno, you can build:

- Agentic RAG: Agents that can search different knowledge stores for RAG or dynamic few-shot learning.
- Image Agents: Agents that can understand images and make tool calls accordingly.
- Reasoning Agents: Agents that can reason using a reasoning model, then generate a result using another model.
- Structured Outputs: Agents that can generate pydantic objects adhering to a schema.



Build intelligent AI agents with Groq's lightning-fast inference and Agno's powerful agent framework. This template demonstrates both simple chat agents and complex multi-tool agents with real-time data processing.

## Live Demo

**Simple Chat Agent**: Run `python main.py` for basic AI conversations  
**Advanced Stock Agent**: Run `python stocks.py` for interactive stock analysis with charts and real-time data

## Overview

This template showcases how to build production-ready AI agents using Groq API and the Agno framework. From simple conversational agents to sophisticated multi-tool systems that can analyze financial data, create visualizations, and maintain conversation context.

**Key Features:**
- 🤖 Simple chat agent setup with Groq integration
- 📊 Advanced multi-tool agent with YFinance integration
- 🎨 Rich terminal UI with charts and visualizations
- 💬 Persistent conversation context and memory
- 📈 Real-time stock analysis and comparison tools
- Sub-second response times, efficient concurrent request handling, and production-grade performance powered by Groq

## Architecture

**Tech Stack:**
- **AI Framework:** Agno for agent orchestration and tool management
- **AI Infrastructure:** Groq API for ultra-fast LLM inference
- **Data Sources:** YFinance for real-time financial data
- **Visualization:** Rich, Plotext, and Matplotlib for charts and tables
- **CLI Interface:** Rich console with beautiful markdown rendering

## Quick Start

### Prerequisites
- Python 3.12+
- Groq API key ([Create a free GroqCloud account and generate an API key here](https://console.groq.com/keys))

### Setup

1. **Clone the repository**
   ```bash
   gh repo clone https://github.com/janzheng/groq-agno-template
   cd groq-agno-template
   ```

2. **Set up Python environment with uv**
   ```bash
   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Initialize the project
   uv init
   
   # Create virtual environment
   uv venv
   
   # Activate virtual environment (optional - uv run handles this automatically)
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   
   **Option A: Using uv (recommended)**
   ```bash
   # Install all dependencies from pyproject.toml
   uv sync
   
   # Or add dependencies individually
   uv add agno>=1.7.0 python-dotenv>=0.9.9 groq>=0.29.0 matplotlib>=3.10.3 mdv>=1.7.5 plotext>=5.3.2 rich>=14.0.0 yfinance>=0.2.64
   ```
   
   **Option B: Using pip**
   ```bash
   # Install from requirements.txt
   pip install -r requirements.txt
   
   # Or install individually
   pip install agno>=1.7.0 python-dotenv>=0.9.9 groq>=0.29.0 matplotlib>=3.10.3 mdv>=1.7.5 plotext>=5.3.2 rich>=14.0.0 yfinance>=0.2.64
   ```

4. **Configure your API key**
   ```bash
   # Create .env file with your Groq API key
   echo "GROQ_API_KEY=gsk_your_actual_api_key_here" > .env
   ```

5. **Run the simple agent**
   ```bash
   uv run python main.py
   ```

6. **Run the advanced stock analysis agent**
   ```bash
   uv run python stocks.py
   ```

## Examples

### Simple Chat Agent (`main.py`)
```python
from agno.agent import Agent
from agno.models.groq import Groq
from dotenv import load_dotenv

load_dotenv()

agent = Agent(
    model=Groq(id="llama-3.1-8b-instant"),
    markdown=True
)

# Print the response in the terminal
agent.print_response("Share a 2 sentence horror story.")
```

### Advanced Multi-Tool Agent (`stocks.py`)
The advanced example includes:
- **Natural Language Processing**: Parse user requests like "analyze apple stock" or "compare tesla vs ford"
- **Real-Time Data**: Fetch live stock prices, analyst recommendations, company news
- **Rich Visualizations**: Terminal charts, comparison tables, and performance metrics
- **Conversation Memory**: Maintain context across multiple interactions
- **Multiple Chart Types**: ASCII, terminal plots, and matplotlib exports

**Key Features:**
- 🗣️ Natural language commands ("analyze apple over 6 months")
- 📊 Multiple visualization options (terminal charts, PNG exports)
- 💬 Persistent chat context and conversation memory
- 📈 Historical performance analysis (1d/5d/1m/6m/ytd/1y)
- ⚖️ Side-by-side stock comparisons
- 🎯 Market sentiment analysis

## Agent Capabilities

### Simple Agent Tools
- Basic conversation and text generation
- Markdown formatting support
- Groq model integration

### Advanced Agent Tools
- **YFinance Integration**: Real-time stock data, company info, analyst recommendations
- **Chart Generation**: Terminal plots, matplotlib charts, ASCII fallbacks
- **Natural Language Parsing**: Convert user requests to structured commands
- **Context Management**: Remember previous analyses and conversations
- **Rich Terminal UI**: Beautiful tables, progress bars, and formatted output

## Usage Examples

### Natural Language Stock Analysis
```bash
# Start the advanced agent
uv run python stocks.py

# Try these natural language commands:
"analyze apple stock over the past year"
"compare tesla vs ford performance"
"how is the market doing today?"
"tell me about nvidia's recent performance"
"what's the best performing tech stock?"
```

### Menu-Driven Interface
The stocks agent also supports traditional menu navigation:
1. Analyze Custom Stocks
2. Compare Stocks
3. Market Sentiment Analysis
4. Show Session Info
5. Exit

## Customization
This template is designed to be a foundation for you to get started with. Key areas for customization:

### Model Configuration
```python
# Change the default model in stocks.py
DEFAULT_MODEL_ID = "qwen/qwen3-32b"  # or "llama-3.1-8b-instant"
```

### Agent Instructions
Modify the agent instructions in `stocks.py` to customize behavior:
- Analysis format and style
- Tool usage patterns
- Response formatting
- Context handling

### Tool Integration
Add new tools to expand agent capabilities:
```python
from agno.tools.web import WebTools
from agno.tools.email import EmailTools

agent = Agent(
    model=Groq(id="llama-3.1-8b-instant"),
    tools=[
        YFinanceTools(...),
        WebTools(),
        EmailTools()
    ]
)
```

### Visualization Options
The template supports multiple visualization libraries:
- **Rich**: Terminal tables and markdown rendering
- **Plotext**: Terminal-based charts and graphs
- **Matplotlib**: High-quality PNG chart exports
- **MDV**: Enhanced markdown rendering

## Advanced Features

### Conversation Context
The advanced agent maintains conversation history and can reference previous analyses:
```python
chat_context = {
    "conversation_history": [],
    "analyzed_stocks": {},
    "session_start": datetime.now(),
    "total_queries": 0
}
```

### Multi-Agent Architecture
The stocks example uses specialized agents:
- **Reasoning Agent**: Comprehensive stock analysis
- **NLP Agent**: Natural language command parsing
- **Chat Agent**: Conversational responses with context

### Real-Time Data Integration
All financial data comes from live sources:
- Current stock prices and volumes
- Analyst recommendations and price targets
- Company news and earnings data
- Historical performance metrics

## Troubleshooting

### Common Issues

1. **Missing API Key**
   ```bash
   # Ensure your .env file contains:
   GROQ_API_KEY=gsk_your_actual_api_key_here
   ```

2. **Dependency Issues**
   ```bash
   # Reinstall dependencies
   uv sync --reinstall
   ```

3. **Chart Display Problems**
   ```bash
   # Install additional visualization libraries
   uv add plotext matplotlib rich mdv
   ```

4. **Permission Errors**
   ```bash
   # Ensure proper virtual environment activation
   source .venv/bin/activate
   uv run python stocks.py
   ```

## Development

### Project Structure
```
groq-agno/
├── main.py              # Simple chat agent example
├── stocks.py            # Advanced multi-tool stock analysis agent
├── pyproject.toml       # Project dependencies and configuration
├── requirements.txt     # Alternative pip-based dependency list
├── .env                 # API keys and environment variables (create from .env.example)
├── LICENSE              # MIT license
└── README.md           # This file
```

### Adding New Features
1. **New Tools**: Add tools to the agent's tool list
2. **Custom Commands**: Extend the natural language parser
3. **Visualization**: Add new chart types or export formats
4. **Data Sources**: Integrate additional APIs or data feeds

## Next Steps

### For Developers
- **Create your free GroqCloud account**: Access official API docs, the playground for experimentation, and more resources via [Groq Console](https://console.groq.com).
- **Build and customize**: Fork this repo and start customizing to build out your own application.
- **Get support**: Connect with other developers building on Groq, chat with our team, and submit feature requests on our [Groq Developer Forum](community.groq.com).

### For Founders and Business Leaders
- **See enterprise capabilities**: This template showcases production-ready AI that can handle realtime business workloads with sophisticated tool integration and data processing.
- **Discuss Your needs**: [Contact our team](https://groq.com/enterprise-access/) to explore how Groq can accelerate your AI initiatives.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Credits
Created with Groq API and Agno framework for building intelligent, production-ready AI agents with real-time data processing capabilities.
