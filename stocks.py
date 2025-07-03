from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.reasoning import ReasoningTools
from agno.tools.yfinance import YFinanceTools

from dotenv import load_dotenv
load_dotenv()

# Model configuration - change here to switch models globally
DEFAULT_MODEL_ID = "qwen/qwen3-32b"

# CLI visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
    from rich.prompt import Prompt, Confirm
    from rich.panel import Panel
    from rich.text import Text
    from rich.live import Live
    from rich.layout import Layout
    from rich.markdown import Markdown
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

try:
    import plotext as plt_terminal
    HAS_PLOTEXT = True
except ImportError:
    HAS_PLOTEXT = False

# Additional markdown rendering libraries
try:
    from rich_cli import Markdown as RichCLIMarkdown
    HAS_RICH_CLI = True
except ImportError:
    HAS_RICH_CLI = False

try:
    import mdv  # Terminal markdown viewer
    HAS_MDV = True
except ImportError:
    HAS_MDV = False

try:
    from textual_markdown import Markdown as TextualMarkdown
    HAS_TEXTUAL_MARKDOWN = True
except ImportError:
    HAS_TEXTUAL_MARKDOWN = False

import time
import re
import json
import warnings
from datetime import datetime, timedelta

# Suppress matplotlib font warnings but keep real errors
warnings.filterwarnings("ignore", message="Glyph.*missing from font.*")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Initialize console for rich output
console = Console() if HAS_RICH else None

# Global chat context to maintain conversation history
chat_context = {
    "conversation_history": [],
    "analyzed_stocks": {},
    "session_start": datetime.now(),
    "total_queries": 0
}

# Web Search Agent using compound-beta model for enhanced analysis
web_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=Groq(id="compound-beta"),
    instructions="""You are a web search specialist using Groq's compound-beta model. Provide the most recent and accurate information available from your web search capabilities.

🔍 **SEARCH INSTRUCTIONS:**
- Search for the most recent and relevant information
- Focus on the user's specific query without over-processing
- Provide direct, factual information from your search results
- Include all relevant sources and links you find

📋 **OUTPUT FORMAT:**
- Present information clearly and directly
- Include all source URLs and publication dates you find
- Use markdown formatting for readability
- Don't over-structure the response - let the search results speak for themselves

🎯 **PRIORITY:**
- Accuracy and recency of information
- Comprehensive source attribution
- Direct relevance to the user's question

Search and provide the most current information available.""",
    markdown=True,
    show_tool_calls=False,
)

reasoning_agent = Agent(
    model=Groq(id=DEFAULT_MODEL_ID),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True, stock_fundamentals=True, income_statements=True, key_financial_ratios=True, technical_indicators=True, historical_prices=True),
    ],
    instructions="""Create comprehensive stock analysis with beautiful markdown formatting:

🚨 **CRITICAL EXECUTION RULES**:
- ALWAYS execute your tools to get real data - do not show tool call syntax
- NEVER show function calls like [get_stock_price=...] or [think=...] in your response
- Execute tools silently and present only the final formatted results
- If tools fail, say "Data not available" instead of showing error syntax

📊 FORMATTING REQUIREMENTS:
- Use clear section headers with ## and ###
- Create well-structured tables with proper alignment
- Use **bold** for important metrics and company names
- Use *italics* for emphasis and notes
- Add emojis for visual appeal (📈 📉 💰 🏢 📊 ⚡ 🎯)
- Format numbers with proper currency symbols and percentages
- Use horizontal rules (---) to separate major sections
- Create comparison tables when analyzing multiple metrics
- Add bullet points for key insights
- Use > blockquotes for important warnings or recommendations

📋 TABLE FORMATTING:
- Align numerical data to the right
- Use consistent column widths
- Add units (USD, %, B, M, etc.) in headers or data
- Format large numbers with commas or abbreviations (2.5T, 1.2B)
- Use colors in text like 🟢 for positive, 🔴 for negative trends

🎯 CONTENT STRUCTURE:
1. Company Overview (name, symbol, sector, market cap)
2. Current Performance (price, changes, volume)
3. Financial Metrics (P/E, EPS, revenue, margins)
4. Historical Performance (multiple time periods)
5. Analyst Recommendations
6. Recent News Summary
7. Key Insights & Analysis
8. Risk Assessment
9. Investment Outlook

IMPORTANT: 
- Execute all tools and present clean, formatted results only
- Do not include any debug output, reasoning steps, or function call syntax
- Only provide the clean, formatted analysis report with real data
- If you cannot get data, state "Data not available" clearly

Make it visually stunning and easy to read in a terminal!""",
    markdown=True,
    show_tool_calls=False,
)

# Enhanced Natural Language Processing Agent with chat context
nlp_agent = Agent(
    model=Groq(id=DEFAULT_MODEL_ID),
    instructions="""You are an expert command parser for a stock analysis tool. Your job is to understand natural language requests and convert them into structured commands.

🎯 **CORE MISSION**: Intelligently detect user intent - stock analysis, web search, or general chat.

📊 **AVAILABLE COMMANDS**:
1. **"analyze"** - Analyze individual stocks (when specific stocks mentioned)
2. **"compare"** - Compare multiple stocks (when multiple stocks mentioned)  
3. **"market"** - Market sentiment analysis (when asking about overall market)
4. **"search"** - Web search for information (when asking for news, research, recent developments, or current events)
5. **"chat"** - General conversation (when no specific stocks mentioned and no search intent)

🔍 **STOCK DETECTION RULES** (BE VERY AGGRESSIVE):
- **ALWAYS look for stock symbols or company names** in the user's request
- **ANY 2-5 letter uppercase combo** could be a stock symbol
- **ANY mention of "stock"** likely means they want stock analysis
- **Convert company names to ticker symbols** using your knowledge (Apple→AAPL, Tesla→TSLA, etc.)
- **Include ETFs and indices** (SPY, QQQ, DIA, VTI, etc.)
- **Include all sectors** (tech, biotech, finance, energy, etc.)

🔍 **SEARCH DETECTION RULES** (BE AGGRESSIVE):
- **Information-seeking keywords**: "what's", "what is", "how is", "tell me about", "find", "search", "look up"
- **Temporal keywords**: "latest", "recent", "recently", "current", "new", "today", "now", "happening"
- **News/research keywords**: "news", "updates", "developments", "research", "reports", "announcements"
- **Company activities**: "doing", "working on", "building", "developing", "launching", "planning"
- **Questions about people**: "what's [person] doing", "how is [person]", especially CEOs like "zuckerberg", "musk", "cook"
- **Technology trends**: "ai", "artificial intelligence", "machine learning", "blockchain", "crypto"
- **Business activities**: "partnerships", "acquisitions", "earnings", "revenue", "growth"

🚨 **CRITICAL PARSING EXAMPLES**:
- "how is apple doing" → {"command": "analyze", "stocks": ["AAPL"], "intent": "analyze Apple performance", "time_periods": []}
- "compare tesla and ford" → {"command": "compare", "stocks": ["TSLA", "F"], "intent": "compare Tesla vs Ford", "time_periods": []}
- "what's the latest news on nvidia" → {"command": "search", "stocks": ["NVDA"], "intent": "search for NVIDIA news", "time_periods": []}
- "what's zuckerberg doing with ai recently" → {"command": "search", "stocks": ["META"], "intent": "search for Zuckerberg/Meta AI developments", "time_periods": []}
- "what's happening with openai lately" → {"command": "search", "stocks": [], "intent": "search for OpenAI developments", "time_periods": []}
- "recent ai developments at microsoft" → {"command": "search", "stocks": ["MSFT"], "intent": "search for Microsoft AI developments", "time_periods": []}
- "what's the market doing" → {"command": "market", "stocks": [], "intent": "market sentiment analysis", "time_periods": []}
- "should I buy tesla now" → {"command": "chat", "stocks": ["TSLA"], "intent": "chat about Tesla investment advice", "time_periods": []}

📋 **COMPANY NAME TO TICKER MAPPING**:
Use your knowledge of company names and ticker symbols. Common patterns:
- Company names → ticker symbols (Apple→AAPL, Tesla→TSLA, etc.)
- Handle variations (Meta/Facebook→META, Google/Alphabet→GOOGL, etc.)
- CEO names → company tickers (Zuckerberg→META, Musk→TSLA, Cook→AAPL, Nadella→MSFT, Pichai→GOOGL)
- Include ETFs and indices (S&P 500→SPY, Nasdaq→QQQ, etc.)

⏰ **TIME PERIOD DETECTION**:
- "last year", "1 year", "yearly", "from last 1 year" → ["1y"]
- "6 months", "half year" → ["6m"] 
- "month", "monthly" → ["1m"]
- "week", "weekly" → ["5d"]
- "ytd", "year to date" → ["ytd"]
- No time mentioned → [] (use defaults)

🎯 **OUTPUT FORMAT** (JSON only):
{
    "command": "analyze|compare|market|search|chat",
    "stocks": ["SYMBOL1", "SYMBOL2"],
    "intent": "clear description of request",
    "time_periods": ["1y", "6m", etc.]
}

**DECISION PRIORITY**:
1. **SEARCH FIRST**: If query has search keywords (what's, latest, recent, news, developments, etc.) → use "search"
2. **STOCK ANALYSIS**: If specific stocks mentioned → use "analyze" or "compare"  
3. **MARKET**: If asking about overall market → use "market"
4. **CHAT LAST**: Only use "chat" for conversational questions without search intent

**REMEMBER**: 
- Be AGGRESSIVE in detecting search intent! 
- Questions starting with "what's", "what is", "how is" often need web search!
- CEO names and company activities usually need current information → "search"!
- When in doubt between "search" and "chat", choose "search"!""",
    markdown=False,
    show_tool_calls=False,
)

# Chat agent for conversational responses
chat_agent = Agent(
    model=Groq(id=DEFAULT_MODEL_ID),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True, stock_fundamentals=True, income_statements=True, key_financial_ratios=True, technical_indicators=True, historical_prices=True),
    ],
    instructions="""You are a friendly, conversational stock market expert. Respond naturally like you're chatting with a friend about stocks, not writing a formal report.

CRITICAL STOCK SYMBOL RULES:
- ONLY mention REAL, VERIFIED stock ticker symbols (like AAPL, MSFT, GOOGL, TSLA, NVDA, AMD, etc.)
- If you're not 100% sure a ticker symbol is real and actively traded, DON'T mention it
- When discussing smaller/lesser-known companies, focus on general categories rather than specific made-up tickers
- Example: "Some smaller AI companies" instead of making up fake ticker symbols

CRITICAL CONTEXT RULES:
- ALWAYS remember what you said in previous responses - if you mentioned specific stocks, remember them
- When user says "those stocks" or "look up those prices", refer back to stocks you mentioned earlier
- Keep track of the conversation flow - don't act like each message is the first

CONVERSATION STYLE:
- Be direct and conversational - start with your answer, not "Based on the conversation history..."
- Use "I think...", "You're right that...", "Actually...", "Yeah..." - natural conversation starters
- Reference previous topics naturally: "Like I mentioned, those AI stocks..." or "The companies we were discussing..."
- Ask follow-up questions to keep the conversation going
- Be concise but helpful
- Use beautiful markdown formatting with tables, emojis, and clear structure when presenting data
- Format any numbers, prices, or percentages clearly with proper symbols ($ % 📈 📉)
- Use bullet points and sections to organize information clearly

WHEN TO USE TOOLS:
- When user asks for stock prices, data, or analysis - USE YOUR TOOLS to get real data
- Present the data naturally: "Let me check those prices for you..." then show the results
- Don't mention function names or technical details - just get the data and present it conversationally
- If you mentioned stocks earlier and user asks about them, remember which ones and look them up

EXAMPLES OF GOOD RESPONSES:
- "Some smaller AI companies like C3.ai (AI) and Palantir (PLTR) might be worth looking at..."
- "Yeah, those AI stocks I mentioned - let me get their latest data..."
- "Actually, let me pull up some current info on the companies we discussed..."

AVOID:
- Making up fake ticker symbols or company names
- "Based on the conversation history..."
- "From the analysis..."
- Showing function call syntax like <function=...>
- Asking user to specify stocks you already mentioned
- Formal report language
- Acting like you don't remember previous parts of the conversation""",
    markdown=True,
    show_tool_calls=False,
)

def add_to_chat_context(user_input, response_type, response_data):
    """Add interaction to chat context"""
    chat_context["conversation_history"].append({
        "timestamp": datetime.now(),
        "user_input": user_input,
        "response_type": response_type,
        "response_data": response_data
    })
    chat_context["total_queries"] += 1

def get_historical_performance_summary(stocks):
    """Get historical performance summary - simplified to let agents handle real data"""
    # Instead of pre-generating fake data, we'll let the agents use their YFinanceTools
    # to get real data when they need it. This function now just returns a placeholder
    # that indicates the agents should fetch real data.
    performance_data = {}
    
    for stock in stocks:
        performance_data[stock] = {
            "note": f"Real-time data for {stock} will be fetched by agents when needed"
        }
    
    return performance_data

def create_performance_table(stocks, performance_data):
    """Create a rich table showing historical performance"""
    if not HAS_RICH:
        return
    
    # Skip the table since we're now letting agents fetch real data
    console.print(f"[yellow]📈 Historical performance data will be included in the detailed analysis above[/yellow]")
    console.print(f"[dim]The agents use real-time YFinance data for accurate performance metrics[/dim]")

def quick_ticker_lookup(tickers):
    """Quick lookup to validate and preview ticker information"""
    if not tickers:
        return {}
    
    lookup_results = {}
    
    if HAS_RICH:
        console.print(f"\n[bold cyan]🔍 Quick Ticker Lookup[/bold cyan]")
    else:
        print(f"\n🔍 Quick Ticker Lookup")
    
    try:
        import yfinance as yf
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                company_name = info.get('longName', 'N/A')
                sector = info.get('sector', 'N/A')
                industry = info.get('industry', 'N/A')
                current_price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
                market_cap = info.get('marketCap', 'N/A')
                
                lookup_results[ticker] = {
                    'company_name': company_name,
                    'sector': sector,
                    'industry': industry,
                    'current_price': current_price,
                    'market_cap': market_cap,
                    'valid': company_name != 'N/A'
                }
                
                if HAS_RICH:
                    price_str = f"${current_price:.2f}" if isinstance(current_price, (int, float)) else str(current_price)
                    mcap_str = f"${market_cap:,.0f}" if isinstance(market_cap, (int, float)) and market_cap > 0 else "N/A"
                    
                    console.print(f"[green]✅ {ticker}[/green]: [bold]{company_name}[/bold]")
                    console.print(f"    💰 {price_str} | 🏢 {sector} - {industry} | 📊 Market Cap: {mcap_str}")
                else:
                    price_str = f"${current_price:.2f}" if isinstance(current_price, (int, float)) else str(current_price)
                    mcap_str = f"${market_cap:,.0f}" if isinstance(market_cap, (int, float)) and market_cap > 0 else "N/A"
                    
                    print(f"✅ {ticker}: {company_name}")
                    print(f"    💰 {price_str} | 🏢 {sector} - {industry} | 📊 Market Cap: {mcap_str}")
                    
            except Exception as e:
                lookup_results[ticker] = {
                    'company_name': 'Error',
                    'sector': 'N/A',
                    'industry': 'N/A', 
                    'current_price': 'N/A',
                    'market_cap': 'N/A',
                    'valid': False,
                    'error': str(e)
                }
                
                if HAS_RICH:
                    console.print(f"[red]❌ {ticker}[/red]: Error fetching data - {str(e)[:50]}...")
                else:
                    print(f"❌ {ticker}: Error fetching data - {str(e)[:50]}...")
                    
    except ImportError:
        if HAS_RICH:
            console.print("[yellow]⚠️  yfinance not available for ticker lookup[/yellow]")
        else:
            print("⚠️  yfinance not available for ticker lookup")
    
    return lookup_results

def parse_natural_language(user_input):
    """Parse natural language input into structured commands"""
    
    # Build context from recent conversation to help with parsing
    context_for_parsing = ""
    if chat_context["conversation_history"]:
        recent_history = chat_context["conversation_history"][-3:]  # Last 3 interactions
        context_for_parsing = "Recent conversation context:\n"
        for interaction in recent_history:
            context_for_parsing += f"- User: {interaction['user_input']}\n"
            if interaction['response_type'] == 'chat' and interaction.get('response_data'):
                # Extract stock symbols from previous responses - look for ticker patterns
                response_text = str(interaction['response_data'])
                context_for_parsing += f"- Response mentioned: {response_text[:200]}\n"
                
                # Extract potential ticker symbols from the response
                ticker_pattern = r'\b([A-Z]{1,5})\b'
                potential_tickers = re.findall(ticker_pattern, response_text)
                # Filter to likely stock tickers (exclude common words, but keep known tickers like AI)
                exclude_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WHO', 'BOY', 'DID', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE'}
                # Keep known stock tickers even if they look like words
                known_tickers = {'AI', 'FSLY', 'PLTR', 'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'META', 'NVDA', 'AMD'}
                likely_tickers = [t for t in potential_tickers if (t not in exclude_words or t in known_tickers) and len(t) <= 5]
                if likely_tickers:
                    context_for_parsing += f"- Potential stock symbols mentioned: {', '.join(likely_tickers)}\n"
    
    parsing_prompt = f"""🎯 **PARSE THIS REQUEST**: '{user_input}'

{context_for_parsing}

🔍 **YOUR TASK**: Analyze this request and determine what the user wants to do.

🚨 **KEY DETECTION POINTS** (BE AGGRESSIVE):
- **SEARCH INTENT**: Look for "what's", "what is", "how is", "latest", "recent", "news", "developments", "doing", "happening"
- **STOCK SYMBOLS**: Look for ANY stock symbols (SPY, AAPL, TSLA, ARMP, etc.) - even if lowercase
- **COMPANY NAMES**: Look for company names (Apple, Tesla, Microsoft, Meta, etc.)
- **CEO NAMES**: Look for CEO names (Zuckerberg→META, Musk→TSLA, Cook→AAPL, etc.)
- **STOCK ANALYSIS**: Look for the word "stock", chart/graph requests ("show", "chart", "graph", "plot")
- **TIME PERIODS**: Look for time periods ("year", "month", "6m", "ytd", "from last", etc.)
- **COMPARISON**: Look for comparison words ("vs", "compare", "against")

📊 **DECISION LOGIC** (PRIORITY ORDER):
1. **SEARCH FIRST**: If you see search keywords (what's, latest, recent, news, developments, doing, happening) → use "search"
2. **STOCK ANALYSIS**: If you see specific stock symbols/companies → use "analyze" or "compare"
3. **MARKET**: If you see words like "market", "overall", "sentiment" → use "market"
4. **CHAT LAST**: Only if no search intent and no specific stocks → use "chat"

🎯 **EXAMPLES FOR THIS REQUEST**:
- "what's zuckerberg doing with ai recently" → {{"command": "search", "stocks": ["META"], "intent": "search for Zuckerberg/Meta AI developments", "time_periods": []}}
- "how is apple doing" → {{"command": "analyze", "stocks": ["AAPL"], "intent": "analyze Apple performance", "time_periods": []}}
- "compare tesla vs ford" → {{"command": "compare", "stocks": ["TSLA", "F"], "intent": "compare Tesla vs Ford", "time_periods": []}}
- "latest news on nvidia" → {{"command": "search", "stocks": ["NVDA"], "intent": "search for NVIDIA news", "time_periods": []}}
- "what's happening with openai" → {{"command": "search", "stocks": [], "intent": "search for OpenAI developments", "time_periods": []}}

⚡ **RESPOND WITH JSON ONLY** - no explanations, just the JSON structure:
{{"command": "analyze|compare|market|search|chat", "stocks": ["SYMBOLS"], "intent": "description", "time_periods": ["periods"]}}

🚨 **CRITICAL**: The user said '{user_input}' - prioritize SEARCH if this looks like an information-seeking question!"""
    
    if HAS_RICH:
        with console.status("[bold blue]🧠 Understanding your request...") as status:
            response = nlp_agent.run(parsing_prompt)
    else:
        print("🧠 Understanding your request...")
        response = nlp_agent.run(parsing_prompt)
    

    
    try:
        # Extract JSON from response
        content = response.content.strip() if response and response.content else ""
        
        if not content:
            # If no content, default to chat command
            return {
                "command": "chat",
                "stocks": [],
                "intent": f"chat about: {user_input}",
                "time_periods": []
            }
        
        # Handle cases where the response might have extra text
        if '```json' in content:
            json_start = content.find('```json') + 7
            json_end = content.find('```', json_start)
            if json_end == -1:
                json_end = len(content)
            content = content[json_start:json_end].strip()
        elif '{' in content and '}' in content:
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            content = content[json_start:json_end]
        else:
            # If no JSON structure found, treat as chat
            return {
                "command": "chat",
                "stocks": [],
                "intent": f"chat about: {user_input}",
                "time_periods": []
            }
        
        parsed_command = json.loads(content)
        
        # Validate and fix the parsed command
        if not isinstance(parsed_command, dict):
            parsed_command = {}
        
        # Ensure required fields exist with defaults
        if 'command' not in parsed_command:
            parsed_command['command'] = 'chat'
        if 'stocks' not in parsed_command:
            parsed_command['stocks'] = []
        if 'intent' not in parsed_command:
            parsed_command['intent'] = f"chat about: {user_input}"
        if 'time_periods' not in parsed_command:
            parsed_command['time_periods'] = []
        
        # Validate stock symbols - only keep real-looking ones
        if parsed_command['stocks']:
            valid_stocks = []
            for stock in parsed_command['stocks']:
                stock = str(stock).strip().upper()
                # Basic validation: 1-5 uppercase letters
                if re.match(r'^[A-Z]{1,5}$', stock):
                    valid_stocks.append(stock)
            parsed_command['stocks'] = valid_stocks
            
            # 🎯 NEW: Do quick ticker lookup for any stocks found
            if valid_stocks:
                lookup_results = quick_ticker_lookup(valid_stocks)
                # Store lookup results in the parsed command for later use
                parsed_command['ticker_lookup'] = lookup_results
        
        return parsed_command
        
    except (json.JSONDecodeError, ValueError, AttributeError) as e:
        # Fallback to chat command instead of showing error
        if HAS_RICH:
            console.print(f"[yellow]🤔 I'll treat that as a general question...[/yellow]")
        else:
            print("🤔 I'll treat that as a general question...")
        
        return {
            "command": "chat",
            "stocks": [],
            "intent": f"chat about: {user_input}",
            "time_periods": []
        }

def execute_natural_command(parsed_command, user_input):
    """Execute a parsed natural language command"""
    command = parsed_command.get('command', '').lower()
    stocks = parsed_command.get('stocks', [])
    intent = parsed_command.get('intent', '')
    time_periods = parsed_command.get('time_periods', [])
    ticker_lookup = parsed_command.get('ticker_lookup', {})
    
    # Ensure stocks and time_periods are lists of strings
    stocks = [str(s) for s in stocks] if stocks else []
    time_periods = [str(t) for t in time_periods] if time_periods else []
    
    if HAS_RICH:
        console.print(f"[bold green]✅ Understood:[/bold green] {intent}")
        console.print(f"[yellow]Command:[/yellow] {command}")
        if stocks:
            console.print(f"[yellow]Stocks:[/yellow] {', '.join(stocks)}")
        if time_periods:
            console.print(f"[yellow]Time Periods:[/yellow] {', '.join(time_periods)}")
    else:
        print(f"✅ Understood: {intent}")
        print(f"Command: {command}")
        if stocks:
            print(f"Stocks: {', '.join(stocks)}")
        if time_periods:
            print(f"Time Periods: {', '.join(time_periods)}")
    
    result = None
    if command == 'analyze':
        if stocks:
            result = analyze_specific_stocks(stocks, time_periods, ticker_lookup)
        else:
            # Fallback to manual input if no stocks detected
            result = analyze_stocks()
    elif command == 'compare':
        if stocks:
            result = compare_specific_stocks(stocks, time_periods, ticker_lookup)
        else:
            # Fallback to manual input if no stocks detected
            result = compare_stocks()
    elif command == 'market':
        result = market_sentiment_analysis()
    elif command == 'chat':
        result = handle_chat_request(user_input, stocks)
    elif command == 'search':
        result = handle_search_request(user_input, stocks)
    else:
        if HAS_RICH:
            console.print(f"[red]❌ Unknown command: {command}[/red]")
        else:
            print(f"❌ Unknown command: {command}")
        return None
    
    # Add to chat context
    add_to_chat_context(user_input, command, result)
    return result

def handle_chat_request(user_input, stocks):
    """Handle conversational chat requests"""
    # Build context from conversation history
    context_summary = f"""
    Session started: {chat_context['session_start'].strftime('%Y-%m-%d %H:%M')}
    Total queries: {chat_context['total_queries']}
    
    Recent conversation history (what we've been talking about):
    """
    
    # Add last 5 interactions for context with full response content
    recent_history = chat_context["conversation_history"][-5:]
    for interaction in recent_history:
        context_summary += f"- User asked: {interaction['user_input']}\n"
        if interaction['response_type'] == 'chat' and interaction.get('response_data'):
            # Include the actual response content for better context
            response_preview = str(interaction['response_data'])[:200] + "..." if len(str(interaction['response_data'])) > 200 else str(interaction['response_data'])
            context_summary += f"  I responded: {response_preview}\n"
        else:
            context_summary += f"  Action: {interaction['response_type']}\n"
    
    # Add analyzed stocks info
    if chat_context["analyzed_stocks"]:
        context_summary += f"\nStocks we've analyzed this session: {', '.join(chat_context['analyzed_stocks'].keys())}\n"
    
    prompt = f"""
    CONVERSATION CONTEXT: {context_summary}
    
    USER'S CURRENT QUESTION: {user_input}
    
    Respond naturally based on our conversation. If I mentioned specific stocks in my previous response, remember them. If the user is asking for stock data or prices, use your tools to get the real information and present it conversationally.
    """
    
    if HAS_RICH:
        with console.status("[bold green]💭 Thinking about your question...") as status:
            response = chat_agent.run(prompt)
        
        render_markdown_with_alternatives(
            response.content,
            title="💬 Chat Response",
            border_style="cyan"
        )
    else:
        print("💭 Thinking about your question...")
        response = chat_agent.run(prompt)
        print(response.content)
    
    return response.content

def analyze_specific_stocks(stocks, time_periods=None, ticker_lookup=None):
    """Analyze specific stocks with historical performance data"""
    if not time_periods:
        time_periods = ["1d", "5d", "1m", "6m", "ytd", "1y"]
    
    if HAS_RICH:
        console.print(f"\n[bold green]🔍 Analyzing {len(stocks)} stocks: {', '.join(stocks)}[/bold green]")
        console.print(f"[yellow]Time periods: {', '.join(time_periods)}[/yellow]")
        display_rich_progress(stocks)
    else:
        print(f"\n🔍 Analyzing {len(stocks)} stocks: {', '.join(stocks)}")
        print(f"Time periods: {', '.join(time_periods)}")
        print("=" * 50)
    
    # Store basic info in chat context (agents will fetch real data)
    performance_data = get_historical_performance_summary(stocks)
    
    analysis_results = {}
    
    for stock in stocks:
        # Store in analyzed stocks
        chat_context["analyzed_stocks"][stock] = {
            "timestamp": datetime.now(),
            "note": f"Analysis requested for {stock}"
        }
        
        # Get ticker lookup info for context
        lookup_info = ticker_lookup.get(stock, {}) if ticker_lookup else {}
        company_name = lookup_info.get('company_name', 'Unknown')
        sector = lookup_info.get('sector', 'Unknown')
        
        # Get web search enhancement
        web_insights = get_web_enhanced_analysis(stock, company_name, sector)
        
        # Create enhanced prompt with ticker lookup context and web insights
        enhanced_prompt = f"""🎯 **COMPREHENSIVE STOCK ANALYSIS REQUEST**

You MUST use your YFinance tools to get REAL data for **{stock}**.

📋 **TICKER CONTEXT** (from direct yfinance lookup):
- Expected Company: {company_name}
- Expected Sector: {sector}
- This should help you verify you're getting data for the RIGHT company

🌐 **WEB INSIGHTS** (latest information from web search):
{web_insights if web_insights else "Web search data not available - proceed with YFinance tools only"}

🚨 **CRITICAL DATA VERIFICATION**:
- Double-check that the company name from your tools matches: "{company_name}"
- If you get different company data, there may be a data source issue
- If data seems inconsistent, clearly state "DATA MISMATCH DETECTED" in your response

**REQUIRED TOOL CALLS:**
1. FIRST: Use stock_price tool to get current price, volume, market cap
2. THEN: Use company_info tool to get company details, sector, financials
3. THEN: Use analyst_recommendations tool to get real analyst data
4. THEN: Use company_news tool to get recent news

📊 **REAL DATA TO GATHER:**
- Current stock price, volume, market cap (from stock_price tool)
- Company information and financials (from company_info tool)
- Analyst recommendations and price targets (from analyst_recommendations tool)
- Recent company news (from company_news tool)

🚨 **CRITICAL REQUIREMENTS:**
- ONLY use data from your YFinance tools for financial metrics
- Incorporate web insights for context and recent developments
- DO NOT make up any prices, percentages, or financial metrics
- If a tool doesn't return data, say "Data not available" instead of inventing numbers
- If you detect data for wrong company, clearly flag this as an error
- Format real data beautifully with emojis and tables
- Include sources from web insights when referencing external information

🎨 **FORMATTING:**
- Use emojis (📈 📉 💰 🏢 ⚡ 🎯)
- Create tables with real data only
- Use **bold** for company names and key metrics
- Format real numbers with currency symbols and percentages
- Include web sources with 🔗 when referencing external information

REMEMBER: Expected company is "{company_name}" - verify this matches your tool results!"""
        
        if HAS_RICH:
            with console.status(f"[bold green]Analyzing {stock}...") as status:
                response = reasoning_agent.run(enhanced_prompt)
                analysis_results[stock] = response.content
            
            # Display individual stock analysis with proper markdown rendering
            render_markdown_with_alternatives(
                response.content,
                title=f"📊 {stock} Analysis",
                border_style="blue"
            )
        else:
            print(f"\n📊 Analyzing {stock}...")
            response = reasoning_agent.run(enhanced_prompt)
            analysis_results[stock] = response.content
            print(response.content)
        
        print("-" * 50)

    # Show historical performance table
    if HAS_RICH:
        console.print("\n[bold yellow]📈 Historical Performance Summary[/bold yellow]")
        create_performance_table(stocks, performance_data)
    
    # Create charts for each stock
    if HAS_RICH:
        console.print("\n[bold cyan]📊 Generating Charts...[/bold cyan]")
    
    for stock in stocks:
        # Try to get real historical data using YFinance tools
        if HAS_RICH:
            console.print(f"[yellow]📊 Attempting to fetch real historical data for {stock}...[/yellow]")
        else:
            print(f"📊 Attempting to fetch real historical data for {stock}...")
        
        # Try to get real historical data from the reasoning agent
        try:
            historical_prompt = f"""Use your historical_prices tool to get real historical price data for {stock} over the last 30 days. 
            Return the data in JSON format with dates and prices only, no analysis or formatting.
            Format: {{"data": [{{"date": "YYYY-MM-DD", "price": 123.45}}, ...]}}"""
            
            historical_response = reasoning_agent.run(historical_prompt)
            
            # Try to extract JSON data from the response
            import json
            content = historical_response.content.strip() if historical_response and historical_response.content else ""
            
            # Look for JSON in the response
            chart_data = None
            if '{' in content and '}' in content:
                try:
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    json_content = content[json_start:json_end]
                    parsed_data = json.loads(json_content)
                    
                    if 'data' in parsed_data and isinstance(parsed_data['data'], list):
                        chart_data = parsed_data['data']
                        if HAS_RICH:
                            console.print(f"[green]✅ Got real historical data for {stock} ({len(chart_data)} data points)[/green]")
                        else:
                            print(f"✅ Got real historical data for {stock} ({len(chart_data)} data points)")
                except (json.JSONDecodeError, KeyError):
                    pass
            
            # If we couldn't get real data, show error instead of fake data
            if not chart_data:
                if HAS_RICH:
                    console.print(f"[red]❌ Unable to fetch historical data for {stock} - chart generation skipped[/red]")
                else:
                    print(f"❌ Unable to fetch historical data for {stock} - chart generation skipped")
                continue
                
        except Exception as e:
            if HAS_RICH:
                console.print(f"[red]❌ Error fetching historical data for {stock}: {str(e)[:100]}...[/red]")
            else:
                print(f"❌ Error fetching historical data for {stock}: {str(e)[:100]}...")
            continue
        
        # Create terminal chart (primary) - only if we have real data
        create_stock_price_chart(chart_data, stock, "terminal")
        
        # Optionally create matplotlib chart if available
        if HAS_MATPLOTLIB:
            create_stock_price_chart(chart_data, stock, "matplotlib")
    
    return analysis_results

def compare_specific_stocks(stocks, time_periods=None, ticker_lookup=None):
    """Compare specific stocks with historical performance context"""
    if not time_periods:
        time_periods = ["1d", "5d", "1m", "6m", "ytd", "1y"]
    
    if HAS_RICH:
        console.print(f"\n[bold cyan]🏆 Comparing {len(stocks)} stocks: {', '.join(stocks)}[/bold cyan]")
        console.print(f"[yellow]Time periods: {', '.join(time_periods)}[/yellow]")
    else:
        print(f"\n🏆 Comparing {len(stocks)} stocks: {', '.join(stocks)}")
        print(f"Time periods: {', '.join(time_periods)}")
    print("=" * 50)
    
    # Store basic info in chat context (agents will fetch real data)
    performance_data = get_historical_performance_summary(stocks)
    
    # Build context from ticker lookup
    lookup_context = ""
    if ticker_lookup:
        lookup_context = "\n📋 **TICKER CONTEXT** (from direct yfinance lookup):\n"
        for stock in stocks:
            lookup_info = ticker_lookup.get(stock, {})
            company_name = lookup_info.get('company_name', 'Unknown')
            sector = lookup_info.get('sector', 'Unknown')
            lookup_context += f"- {stock}: {company_name} ({sector})\n"
        lookup_context += "\nUse this to verify you're getting data for the RIGHT companies.\n"
    
    # Get web insights for comparison enhancement
    comparison_web_insights = []
    for stock in stocks:
        lookup_info = ticker_lookup.get(stock, {}) if ticker_lookup else {}
        company_name = lookup_info.get('company_name', 'Unknown')
        sector = lookup_info.get('sector', 'Unknown')
        web_insight = get_web_enhanced_analysis(stock, company_name, sector)
        if web_insight:
            comparison_web_insights.append(f"**{stock}**: {web_insight[:300]}...")
    
    web_context = ""
    if comparison_web_insights:
        web_context = f"\n🌐 **WEB INSIGHTS FOR COMPARISON:**\n" + "\n\n".join(comparison_web_insights) + "\n"
    
    enhanced_prompt = f"""🏆 **COMPREHENSIVE STOCK COMPARISON REQUEST**

Compare **{', '.join(stocks)}** stocks with beautiful side-by-side analysis.
{lookup_context}
{web_context}
📊 **COMPARISON DATA TO GATHER:**
- Current prices, market caps, and trading volumes
- Historical performance for periods: {', '.join(time_periods)}
- Analyst recommendations and price targets
- Key financial ratios (P/E, P/B, ROE, Profit Margins)
- Recent news sentiment and developments
- Sector positioning and competitive advantages

🎨 **FORMATTING REQUIREMENTS:**
- Create side-by-side comparison tables
- Use emojis for visual appeal (🥇 🥈 🥉 📈 📉 💰 ⚡)
- Highlight winners/losers with visual indicators
- Use **bold** for best performers in each category
- Add summary scorecards for each stock
- Include > blockquote recommendations
- Use horizontal rules between major sections
- Include web sources with 🔗 when referencing external information

🎯 **ANALYSIS GOALS:**
- Rank stocks by different criteria (growth, value, stability)
- Identify the best investment opportunity and explain why
- Highlight key risks and opportunities for each
- Provide clear buy/hold/sell guidance
- Incorporate latest web insights for comprehensive analysis

Make this comparison visually stunning and actionable!"""
    
    if HAS_RICH:
        with console.status("[bold green]Generating comparison analysis...") as status:
            response = reasoning_agent.run(enhanced_prompt)
        
        # Show historical performance first
        console.print("\n[bold yellow]📈 Historical Performance Comparison[/bold yellow]")
        create_performance_table(stocks, performance_data)
        
        render_markdown_with_alternatives(
            response.content,
            title="🏆 Stock Comparison Analysis",
            border_style="green"
        )
    else:
        print("Generating comparison analysis...")
        response = reasoning_agent.run(enhanced_prompt)
    print(response.content)
    
    # Create comparison charts
    if HAS_RICH:
        console.print("\n[bold cyan]📊 Generating Comparison Charts...[/bold cyan]")
    
    # Try to gather real chart data for all stocks
    stocks_chart_data = {}
    for stock in stocks:
        try:
            historical_prompt = f"""Use your historical_prices tool to get real historical price data for {stock} over the last 30 days. 
            Return the data in JSON format with dates and prices only, no analysis or formatting.
            Format: {{"data": [{{"date": "YYYY-MM-DD", "price": 123.45}}, ...]}}"""
            
            historical_response = reasoning_agent.run(historical_prompt)
            content = historical_response.content.strip() if historical_response and historical_response.content else ""
            
            chart_data = None
            if '{' in content and '}' in content:
                try:
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    json_content = content[json_start:json_end]
                    parsed_data = json.loads(json_content)
                    
                    if 'data' in parsed_data and isinstance(parsed_data['data'], list):
                        chart_data = parsed_data['data']
                        stocks_chart_data[stock] = chart_data
                except (json.JSONDecodeError, KeyError):
                    pass
            
            if not chart_data:
                if HAS_RICH:
                    console.print(f"[yellow]⚠️  No historical data available for {stock} in comparison[/yellow]")
                else:
                    print(f"⚠️  No historical data available for {stock} in comparison")
                    
        except Exception as e:
            if HAS_RICH:
                console.print(f"[yellow]⚠️  Error fetching data for {stock}: {str(e)[:50]}...[/yellow]")
            else:
                print(f"⚠️  Error fetching data for {stock}: {str(e)[:50]}...")
    
    # Only create comparison chart if we have data for at least one stock
    if stocks_chart_data:
        create_comparison_chart(stocks_chart_data, "terminal")
        
        # Also create matplotlib chart if available
        if HAS_MATPLOTLIB:
            create_comparison_chart(stocks_chart_data, "matplotlib")
    else:
        if HAS_RICH:
            console.print("[red]❌ No historical data available for any stocks - comparison chart skipped[/red]")
        else:
            print("❌ No historical data available for any stocks - comparison chart skipped")
    
    # Store in chat context
    for stock in stocks:
        chat_context["analyzed_stocks"][stock] = {
            "timestamp": datetime.now(),
            "note": f"Comparison analysis requested for {stock}"
        }
    
    return response.content

def show_chat_context():
    """Display current chat context and session info"""
    if HAS_RICH:
        info_text = f"""
Session Duration: {datetime.now() - chat_context['session_start']}
Total Queries: {chat_context['total_queries']}
Stocks Analyzed: {len(chat_context['analyzed_stocks'])}
Conversation History: {len(chat_context['conversation_history'])} interactions
        """
        
        console.print(Panel(
            info_text.strip(),
            title="💬 Session Info",
            border_style="yellow"
        ))
        
        if chat_context["analyzed_stocks"]:
            console.print("\n[bold yellow]📊 Stocks in Memory:[/bold yellow]")
            for stock, data in chat_context["analyzed_stocks"].items():
                console.print(f"  • [cyan]{stock}[/cyan] - analyzed {data['timestamp'].strftime('%H:%M')}")
    else:
        print(f"\n💬 Session Info:")
        print(f"Session Duration: {datetime.now() - chat_context['session_start']}")
        print(f"Total Queries: {chat_context['total_queries']}")
        print(f"Stocks Analyzed: {len(chat_context['analyzed_stocks'])}")
        print(f"Conversation History: {len(chat_context['conversation_history'])} interactions")
        
        if chat_context["analyzed_stocks"]:
            print("\n📊 Stocks in Memory:")
            for stock, data in chat_context["analyzed_stocks"].items():
                print(f"  • {stock} - analyzed {data['timestamp'].strftime('%H:%M')}")

def get_user_stocks():
    """Get stock symbols from user input with validation"""
    if HAS_RICH:
        console.print("\n[bold cyan]📈 Stock Analysis Tool[/bold cyan]")
        console.print("[yellow]Enter stock symbols separated by commas (e.g., AAPL, GOOGL, MSFT)[/yellow]")
        
        stocks_input = Prompt.ask(
            "[green]Stock symbols[/green]",
            default="AAPL,GOOGL,MSFT,TSLA"
        )
    else:
        print("\n📈 Stock Analysis Tool")
        print("Enter stock symbols separated by commas (e.g., AAPL, GOOGL, MSFT)")
        stocks_input = input("Stock symbols (default: AAPL,GOOGL,MSFT,TSLA): ").strip()
        if not stocks_input:
            stocks_input = "AAPL,GOOGL,MSFT,TSLA"
    
    # Clean and validate stock symbols
    stocks = [stock.strip().upper() for stock in stocks_input.split(',')]
    stocks = [stock for stock in stocks if stock and re.match(r'^[A-Z]{1,5}$', stock)]
    
    if not stocks:
        if HAS_RICH:
            console.print("[red]No valid stock symbols entered. Using defaults.[/red]")
        else:
            print("No valid stock symbols entered. Using defaults.")
        stocks = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    
    return stocks

# Removed fake chart functions - agents provide real data analysis instead

def display_rich_progress(stocks):
    """Display a rich progress bar while analyzing stocks"""
    if not HAS_RICH:
        return
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
        transient=True,
    ) as progress:
        
        task = progress.add_task("Analyzing stocks...", total=len(stocks))
        
        for stock in stocks:
            progress.update(task, description=f"Analyzing {stock}...")
            time.sleep(0.5)  # Simulate processing time
            progress.advance(task)

# Removed fake comparison table - agents provide real data analysis instead

def analyze_stocks():
    """Analyze multiple stocks and display results with CLI visualizations"""
    stocks = get_user_stocks()
    return analyze_specific_stocks(stocks)

def compare_stocks():
    """Compare stocks side by side with enhanced visualization"""
    stocks = get_user_stocks()
    return compare_specific_stocks(stocks)

def market_sentiment_analysis():
    """Analyze overall market sentiment using multiple data points"""
    if HAS_RICH:
        console.print("\n[bold magenta]📈 Market Sentiment Analysis[/bold magenta]")
        
        # Get web insights for market analysis
        market_web_search = """Search for the latest market sentiment and analysis:
        
        Focus on:
        - Overall market trends and sentiment
        - Federal Reserve policy and interest rate impacts
        - Economic indicators (inflation, employment, GDP)
        - Geopolitical events affecting markets
        - Sector rotation and institutional flows
        - Analyst market outlook and predictions
        - Recent market volatility and catalysts
        
        Provide comprehensive market analysis with reliable sources."""
        
        with console.status("[bold green]Analyzing market sentiment with web insights...") as status:
            # Get web insights first
            web_insights = web_agent.run(market_web_search)
            
            response = reasoning_agent.run(
                f"""📈 **COMPREHENSIVE MARKET SENTIMENT ANALYSIS**

Analyze current market sentiment with beautiful, detailed formatting.

🌐 **LATEST WEB INSIGHTS:**
{web_insights.content if hasattr(web_insights, 'content') else web_insights}

📊 **MARKET DATA TO ANALYZE:**
- Major indices: SPY, QQQ, DIA, VIX
- Key individual stocks: AAPL, MSFT, GOOGL, TSLA, NVDA
- Sector performance (Tech, Finance, Healthcare, Energy)
- Recent economic indicators and news sentiment

🎨 **FORMATTING REQUIREMENTS:**
- Use market emojis (📈 📉 🐂 🐻 ⚡ 🔥 ❄️ 🌊)
- Create beautiful comparison tables
- Use **bold** for key metrics and trends
- Color indicators: 🟢 for bullish, 🔴 for bearish, 🟡 for neutral
- Add horizontal rules between sections
- Format percentages and dollar amounts clearly
- Include web sources with 🔗 when referencing external information

🎯 **ANALYSIS STRUCTURE:**
1. **Market Overview** - Current state and trends
2. **Index Performance** - SPY, QQQ, DIA comparison
3. **Sector Analysis** - Winners and losers
4. **Individual Stock Highlights** - Key movers
5. **Sentiment Indicators** - VIX, news sentiment
6. **Economic Factors** - Interest rates, inflation, etc.
7. **Market Outlook** - Bullish/bearish forecast
8. **Web Insights Summary** - Key findings from latest research

> **Goal:** Provide clear, actionable market insights with stunning visual presentation and latest web intelligence!
                """
            )
        
        render_markdown_with_alternatives(
            response.content,
            title="📈 Market Sentiment Analysis",
            border_style="magenta"
        )
    else:
        print("\n📈 Market Sentiment Analysis")
        print("=" * 50)
        
        # Get web insights for market analysis
        market_web_search = """Search for the latest market sentiment and analysis:
        
        Focus on:
        - Overall market trends and sentiment
        - Federal Reserve policy and interest rate impacts
        - Economic indicators (inflation, employment, GDP)
        - Geopolitical events affecting markets
        - Sector rotation and institutional flows
        - Analyst market outlook and predictions
        - Recent market volatility and catalysts
        
        Provide comprehensive market analysis with reliable sources."""
        
        print("🌐 Gathering latest market insights...")
        web_insights = web_agent.run(market_web_search)
        
        response = reasoning_agent.run(
            f"""📈 **COMPREHENSIVE MARKET SENTIMENT ANALYSIS**

Analyze current market sentiment with beautiful, detailed formatting.

🌐 **LATEST WEB INSIGHTS:**
{web_insights.content if hasattr(web_insights, 'content') else web_insights}

📊 **MARKET DATA TO ANALYZE:**
- Major indices: SPY, QQQ, DIA, VIX
- Key individual stocks: AAPL, MSFT, GOOGL, TSLA, NVDA
- Sector performance (Tech, Finance, Healthcare, Energy)
- Recent economic indicators and news sentiment

🎨 **FORMATTING REQUIREMENTS:**
- Use market emojis (📈 📉 🐂 🐻 ⚡ 🔥 ❄️ 🌊)
- Create beautiful comparison tables
- Use **bold** for key metrics and trends
- Color indicators: 🟢 for bullish, 🔴 for bearish, 🟡 for neutral
- Add horizontal rules between sections
- Format percentages and dollar amounts clearly
- Include web sources with 🔗 when referencing external information

🎯 **ANALYSIS STRUCTURE:**
1. **Market Overview** - Current state and trends
2. **Index Performance** - SPY, QQQ, DIA comparison
3. **Sector Analysis** - Winners and losers
4. **Individual Stock Highlights** - Key movers
5. **Sentiment Indicators** - VIX, news sentiment
6. **Economic Factors** - Interest rates, inflation, etc.
7. **Market Outlook** - Bullish/bearish forecast
8. **Web Insights Summary** - Key findings from latest research

> **Goal:** Provide clear, actionable market insights with stunning visual presentation and latest web intelligence!
            """
        )
        print(response.content)
    
    # Add to chat context
    add_to_chat_context("market sentiment analysis", "market", response.content)
    return response.content

def handle_input(user_input):
    """Handle both numbered menu choices and natural language input"""
    user_input = user_input.strip()
    
    # Check for special commands
    if user_input.lower() in ['context', 'session', 'info']:
        return "context", None
    
    if user_input.lower() in ['demo', 'chart demo', 'charts', 'demo charts']:
        return "demo", None
    
    # Check if it's a menu number
    if user_input in ["1", "2", "3", "4", "5", "6"]:
        return "menu", user_input
    
    # Check for exit commands
    if user_input.lower() in ['exit', 'quit', 'q', 'bye']:
        return "exit", None
    
    # Otherwise, treat as natural language
    return "natural", user_input

def process_menu_choice(choice):
    """Process numbered menu choices"""
    if choice == "1":
        return analyze_stocks()
    elif choice == "2":
        return compare_stocks()
    elif choice == "3":
        return market_sentiment_analysis()
    elif choice == "4":
        # Web search for stock info
        if HAS_RICH:
            search_query = Prompt.ask(
                "[green]Enter your search query (e.g., 'latest news on AAPL', 'Tesla earnings')[/green]"
            )
        else:
            search_query = input("Enter your search query (e.g., 'latest news on AAPL', 'Tesla earnings'): ").strip()
        
        if search_query:
            # Parse the search query to extract stocks if any
            parsed_command = parse_natural_language(search_query)
            parsed_command['command'] = 'search'  # Force search command
            return execute_natural_command(parsed_command, search_query)
        else:
            if HAS_RICH:
                console.print("[red]No search query provided.[/red]")
            else:
                print("No search query provided.")
            return None
    elif choice == "5":
        show_chat_context()
        return None
    elif choice == "6":
        if HAS_RICH:
            console.print("[bold green]👋 Thanks for using the Stock Analysis Tool![/bold green]")
        else:
            print("👋 Thanks for using the Stock Analysis Tool!")
        return "exit"
    else:
        if HAS_RICH:
            console.print("[red]Invalid menu choice.[/red]")
        else:
            print("Invalid menu choice.")
        return None

def show_menu():
    """Display the main menu with options"""
    if HAS_RICH:
        console.print("\n[bold cyan]📈 Stock Analysis Menu[/bold cyan]")
        console.print("[1] Analyze Custom Stocks")
        console.print("[2] Compare Stocks") 
        console.print("[3] Market Sentiment Analysis")
        console.print("[4] Web Search for Stock Info")
        console.print("[5] Show Session Info")
        console.print("[6] Exit")
        console.print("\n[dim]💡 After any analysis, just keep typing to chat about the results or request new analysis![/dim]")
        console.print("\n[bold yellow]🌐 Enhanced with Web Search:[/bold yellow] [dim]All analyses now include latest web insights with sources![/dim]")
    else:
        print("\n📈 Stock Analysis Menu")
        print("1. Analyze Custom Stocks")
        print("2. Compare Stocks")
        print("3. Market Sentiment Analysis")
        print("4. Web Search for Stock Info") 
        print("5. Show Session Info")
        print("6. Exit")
        print("\n💡 After any analysis, just keep typing to chat about the results or request new analysis!")
        print("\n🌐 Enhanced with Web Search: All analyses now include latest web insights with sources!")

def show_natural_language_examples():
    """Show examples of natural language commands"""
    if HAS_RICH:
        console.print("\n[bold yellow]💡 You can type a number (1-6) or use natural language:[/bold yellow]")
        
        examples = [
            ("📊 Analysis", [
                "analyze apple stock",
                "how is tesla doing over the past year?",
                "analyze microsoft, google and amazon performance",
                "tell me about nvidia's 5-year performance"
            ]),
            ("⚖️  Comparison", [
                "compare apple with microsoft over 6 months",
                "apple vs google vs microsoft ytd performance",
                "which performed better: tesla or ford over 1 year?",
                "compare tech stocks over max time period"
            ]),
            ("📈 Market", [
                "how is the market doing?",
                "market sentiment analysis",
                "overall market outlook"
            ]),
            ("🔍 Web Search", [
                "what's zuckerberg doing with ai recently?",
                "search for tesla's latest innovations",
                "recent ai developments at microsoft",
                "find information about nvidia's partnerships",
                "what's happening with openai lately?"
            ]),
            ("💬 Chat", [
                "tell me more about the apple analysis",
                "what do you think about tesla's recent performance?",
                "which stock should I focus on?",
                "should I buy tesla now?",
                "what's the best performing stock today?",
                "context / session / info - show session details"
            ])
        ]
        
        for category, cmds in examples:
            console.print(f"\n[bold]{category}[/bold]")
            for cmd in cmds:
                console.print(f"  • [green]'{cmd}'[/green]")
    else:
        print("\n💡 You can type a number (1-6) or use natural language:")
        print("\n📊 Analysis:")
        print("  • 'analyze apple stock'")
        print("  • 'how is tesla doing over the past year?'")
        print("  • 'analyze microsoft, google and amazon performance'")
        print("  • 'tell me about nvidia's 5-year performance'")
        print("\n⚖️  Comparison:")
        print("  • 'compare apple with microsoft over 6 months'")
        print("  • 'apple vs google vs microsoft ytd performance'")
        print("  • 'which performed better: tesla or ford over 1 year?'")
        print("  • 'compare tech stocks over max time period'")
        print("\n📈 Market:")
        print("  • 'how is the market doing?'")
        print("  • 'market sentiment analysis'")
        print("  • 'overall market outlook'")
        print("\n🔍 Web Search:")
        print("  • 'what's zuckerberg doing with ai recently?'")
        print("  • 'search for tesla's latest innovations'")
        print("  • 'recent ai developments at microsoft'")
        print("  • 'find information about nvidia's partnerships'")
        print("  • 'what's happening with openai lately?'")
        print("\n💬 Chat:")
        print("  • 'tell me more about the apple analysis'")
        print("  • 'what do you think about tesla's recent performance?'")
        print("  • 'which stock should I focus on?'")
        print("  • 'should I buy tesla now?'")
        print("  • 'what's the best performing stock today?'")
        print("  • 'context / session / info' - show session details")

def main_interactive():
    """Main interactive function with hybrid menu/natural language system"""
    if HAS_RICH:
        console.print("[bold green]🚀 Welcome to the Enhanced Stock Analysis Tool![/bold green]")
        console.print("[bold cyan]✨ Now with Web Search, Historical Performance & Chat Memory![/bold cyan]")
        console.print("[bold blue]🌐 Powered by Groq's compound-beta model for real-time web insights![/bold blue]")
        
        # Show available features
        features = []
        features.append("🔍 Real-time Web Search with Sources")
        features.append("🗣️  Natural Language + Menu Numbers")
        features.append("📈 Historical Performance (1d/5d/1m/6m/ytd/1y/5y/max)")
        features.append("💬 Persistent Chat Context")
        if HAS_RICH:
            features.append("✅ Rich CLI interface with Markdown Tables")
        if HAS_MDV:
            features.append("✅ MDV Markdown renderer")
        if HAS_PLOTEXT:
            features.append("✅ Terminal charts")
        if HAS_MATPLOTLIB:
            features.append("✅ Advanced plotting with PNG export")
        if HAS_PLOTEXT:
            features.append("✅ Terminal line charts")
        
        if features:
            console.print(f"[yellow]Available features: {', '.join(features)}[/yellow]")
        
        # Show markdown rendering info
        console.print(f"[dim]📝 Markdown rendering: {'Rich (tables supported)' if HAS_RICH else 'Basic text'}[/dim]")
        console.print(f"[dim]🔗 Web search: Latest news, analyst reports, and market insights with sources[/dim]")
        
        # Show natural language examples
        show_natural_language_examples()
    else:
        print("🚀 Welcome to the Enhanced Stock Analysis Tool!")
        print("✨ Now with Web Search, Historical Performance & Chat Memory!")
        print("🌐 Powered by Groq's compound-beta model for real-time web insights!")
        print("For better experience with beautiful markdown tables, install:")
        print("  pip install rich mdv")
        print("  or: pip install rich plotext matplotlib mdv")
        show_natural_language_examples()
    
    while True:
        show_menu()
        
        if HAS_RICH:
            user_input = Prompt.ask(
                "\n[green]Enter menu number (1-5) OR type your request naturally[/green]",
                default=""
            )
        else:
            user_input = input("\nEnter menu number (1-5) OR type your request naturally: ").strip()
        
        if not user_input:
            continue
            
        input_type, processed_input = handle_input(user_input)
        
        if input_type == "exit":
            if HAS_RICH:
                console.print("[bold green]👋 Thanks for using the Stock Analysis Tool![/bold green]")
            else:
                print("👋 Thanks for using the Stock Analysis Tool!")
            break
        elif input_type == "context":
            show_chat_context()
        elif input_type == "demo":
            demo_charts()
        elif input_type == "menu":
            result = process_menu_choice(processed_input)
            if result == "exit":
                break
        elif input_type == "natural":
            parsed_command = parse_natural_language(user_input)
            if parsed_command:
                execute_natural_command(parsed_command, user_input)
        
        # No continue prompt - just keep the conversation flowing

def render_markdown_with_alternatives(content, title=None, border_style="blue"):
    """Render markdown with multiple fallback options for best table display"""
    if not content:
        return
    
    # Option 1: Rich Markdown (primary choice - best table support)
    if HAS_RICH:
        # Enhanced Rich rendering with better styling
        markdown = Markdown(content, code_theme="monokai", hyperlinks=True)
        
        if title:
            # Create a more beautiful panel with enhanced styling
            console.print("\n")  # Add space before
            console.print(Panel(
                markdown,
                title=f"[bold white]{title}[/bold white]",
                border_style=border_style,
                expand=False,
                padding=(1, 2),
                title_align="left",
                subtitle="[dim]Powered by Rich Markdown[/dim]",
                subtitle_align="right"
            ))
            console.print()  # Add space after
        else:
            console.print(markdown)
        return
    
    # Option 2: MDV (Terminal markdown viewer - excellent table support)
    if HAS_MDV:
        if title:
            console.print(f"\n{'='*60}")
            console.print(f"📊 {title}")
            console.print('='*60)
        
        try:
            # MDV renders markdown beautifully in terminal with colored tables
            rendered = mdv.main(content, theme='solarized', cols=120, tab_length=4)
            print(rendered)
        except Exception as e:
            # Fallback to basic mdv if theme fails
            try:
                rendered = mdv.main(content, cols=120)
                print(rendered)
            except:
                print(content)
        return
    
    # Option 3: Fallback to plain text with better formatting
    if title:
        print(f"\n{'='*60}")
        print(f"📊 {title}")
        print('='*60)
    print(content)
    print("-" * 60)

def render_markdown_content(content, title=None, border_style="blue"):
    """Render markdown content with proper table formatting using Rich"""
    if not HAS_RICH:
        print(content)
        return
    
    # Create markdown object with enhanced styling
    markdown = Markdown(content, code_theme="monokai", hyperlinks=True)
    
    if title:
        console.print(Panel(
            markdown,
            title=title,
            border_style=border_style,
            expand=False,
            padding=(1, 2),  # Add padding for better spacing
            title_align="left"
        ))
    else:
        console.print(markdown)

def create_stock_price_chart(stock_data, stock_symbol, chart_type="terminal"):
    """Create beautiful line charts for stock price data"""
    if not stock_data or len(stock_data) < 2:
        if HAS_RICH:
            console.print(f"[yellow]⚠️  Insufficient data to create chart for {stock_symbol}[/yellow]")
        else:
            print(f"⚠️  Insufficient data to create chart for {stock_symbol}")
        return
    
    if chart_type == "terminal" and HAS_PLOTEXT:
        create_terminal_chart(stock_data, stock_symbol)
    elif chart_type == "matplotlib" and HAS_MATPLOTLIB:
        create_matplotlib_chart(stock_data, stock_symbol)
    else:
        # Fallback to simple ASCII chart
        create_ascii_chart(stock_data, stock_symbol)

def create_terminal_chart(stock_data, stock_symbol):
    """Create beautiful terminal-based line chart using plotext with dark theme"""
    try:
        import plotext as plt
        
        # Extract prices and use simple numeric indices instead of dates
        prices = []
        labels = []
        
        for i, item in enumerate(stock_data):
            price = float(item.get('price', 0))
            if price > 0:
                prices.append(price)
                # Use simple day numbers instead of problematic date formatting
                labels.append(f"Day {i+1}")
        
        if not prices or all(p == 0 for p in prices):
            return
        
        # Clear previous plot
        plt.clear_data()
        plt.clear_figure()
        
        # Set dark theme
        plt.theme("dark")
        
        # Create the line plot with numeric x-axis
        x_values = list(range(1, len(prices) + 1))
        plt.plot(x_values, prices, marker="braille", color="white")
        
        # Customize the plot
        plt.title(f"📈 {stock_symbol} Stock Price Chart (Last {len(prices)} Days)")
        plt.xlabel("📅 Days Ago → Recent")
        plt.ylabel("💰 Price (USD)")
        
        # Set plot size for terminal
        plt.plotsize(80, 20)
        
        # Add grid
        plt.grid(True, True)
        
        # Show the plot
        plt.show()
        
        # Chart displayed successfully (no verbose output)
            
    except Exception as e:
        if HAS_RICH:
            console.print(f"[red]❌ Error creating terminal chart: {e}[/red]")
        else:
            print(f"❌ Error creating terminal chart: {e}")

def create_matplotlib_chart(stock_data, stock_symbol):
    """Create advanced chart using matplotlib with dark color scheme"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
        
        # Extract and process data
        dates = []
        prices = []
        
        for item in stock_data:
            try:
                date_str = item.get('date', '')
                price = float(item.get('price', 0))
                
                if date_str and price > 0:
                    # Parse date string
                    if isinstance(date_str, str):
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    else:
                        date_obj = date_str
                    
                    dates.append(date_obj)
                    prices.append(price)
            except (ValueError, TypeError):
                continue
        
        if len(dates) < 2:
            return
        
        # Create the plot with dark theme
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Set dark background
        fig.patch.set_facecolor('#1a1a1a')  # Dark background
        ax.set_facecolor('#2d2d2d')  # Darker plot area
        
        # Plot the line chart with bright, visible colors
        ax.plot(dates, prices, linewidth=3, color='#00ff88', marker='o', markersize=5, 
                markerfacecolor='#ffff00', markeredgecolor='#ffffff', markeredgewidth=1)
        
        # Customize the chart with light colors for dark background
        ax.set_title(f'{stock_symbol} Stock Price Chart', fontsize=16, fontweight='bold', 
                    pad=20, color='#ffffff')
        ax.set_xlabel('Date', fontsize=12, color='#ffffff')
        ax.set_ylabel('Price (USD)', fontsize=12, color='#ffffff')
        
        # Format x-axis dates with light colors
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45, color='#ffffff')
        plt.yticks(color='#ffffff')
        
        # Add grid with visible dark theme colors
        ax.grid(True, alpha=0.6, color='#555555', linestyle='-', linewidth=0.5)
        
        # Style the chart spines with bright colors
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#ffffff')
        ax.spines['bottom'].set_color('#ffffff')
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        
        # Tight layout
        plt.tight_layout()
        
        # Don't save charts automatically (keep folder clean)
        # filename = f"{stock_symbol}_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        # plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        
        # Don't show plot to avoid popup windows
        
        plt.close()
        
    except Exception as e:
        if HAS_RICH:
            console.print(f"[red]❌ Error creating matplotlib chart: {e}[/red]")
        else:
            print(f"❌ Error creating matplotlib chart: {e}")

def create_ascii_chart(stock_data, stock_symbol):
    """Create simple ASCII line chart as fallback"""
    try:
        prices = [float(item.get('price', 0)) for item in stock_data if item.get('price', 0) > 0]
        
        if len(prices) < 2:
            return
        
        # Normalize prices to fit in terminal width
        min_price = min(prices)
        max_price = max(prices)
        price_range = max_price - min_price
        
        if price_range == 0:
            return
        
        # Create ASCII chart
        chart_height = 10
        chart_width = min(len(prices), 60)
        
        if HAS_RICH:
            console.print(f"\n[bold cyan]📈 {stock_symbol} Price Trend (ASCII)[/bold cyan]")
        else:
            print(f"\n📈 {stock_symbol} Price Trend (ASCII)")
        
        print(f"High: ${max_price:.2f}")
        
        # Draw the chart
        for row in range(chart_height):
            line = ""
            threshold = max_price - (row / chart_height) * price_range
            
            for i in range(min(len(prices), chart_width)):
                if prices[i] >= threshold:
                    line += "█"
                else:
                    line += " "
            
            print(f"{threshold:6.2f} |{line}")
        
        print(f" Low: ${min_price:.2f}")
        print("       " + "-" * chart_width)
        
    except Exception as e:
        if HAS_RICH:
            console.print(f"[red]❌ Error creating ASCII chart: {e}[/red]")
        else:
            print(f"❌ Error creating ASCII chart: {e}")

def create_comparison_chart(stocks_data, chart_type="terminal"):
    """Create comparison charts for multiple stocks"""
    if not stocks_data or len(stocks_data) < 2:
        return
    
    if chart_type == "terminal" and HAS_PLOTEXT:
        create_terminal_comparison_chart(stocks_data)
    elif chart_type == "matplotlib" and HAS_MATPLOTLIB:
        create_matplotlib_comparison_chart(stocks_data)
    else:
        # Fallback to simple text-based comparison
        if HAS_RICH:
            console.print("[yellow]📊 Chart libraries not available, showing data summary[/yellow]")
        for symbol, data in stocks_data.items():
            if data:
                prices = [float(item.get('price', 0)) for item in data]
                if prices:
                    print(f"{symbol}: ${prices[0]:.2f} → ${prices[-1]:.2f} ({((prices[-1]/prices[0]-1)*100):+.1f}%)")

def create_terminal_comparison_chart(stocks_data):
    """Create terminal comparison chart for multiple stocks with dark theme"""
    try:
        import plotext as plt
        
        plt.clear_data()
        plt.clear_figure()
        
        # Set dark theme
        plt.theme("dark")
        
        # Use bright colors that work well on dark background
        colors = ["white", "cyan", "yellow", "green", "red", "magenta"]
        
        for i, (stock_symbol, data) in enumerate(stocks_data.items()):
            if not data:
                continue
                
            prices = []
            
            for item in data:
                price = float(item.get('price', 0))
                if price > 0:
                    prices.append(price)
            
            if prices:
                # Use numeric x-axis instead of dates
                x_values = list(range(1, len(prices) + 1))
                color = colors[i % len(colors)]
                plt.plot(x_values, prices, marker="braille", color=color, label=stock_symbol)
        
        plt.title("📊 Stock Price Comparison")
        plt.xlabel("📅 Days (1 = Oldest, Higher = More Recent)")
        plt.ylabel("💰 Price (USD)")
        plt.plotsize(100, 25)
        plt.grid(True, True)
        
        # Show legend
        plt.show()
        
        # Terminal comparison chart displayed silently
            
    except Exception as e:
        if HAS_RICH:
            console.print(f"[red]❌ Error creating comparison chart: {e}[/red]")
        else:
            print(f"❌ Error creating comparison chart: {e}")

def create_matplotlib_comparison_chart(stocks_data):
    """Create matplotlib comparison chart for multiple stocks with dark theme"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Set dark background
        fig.patch.set_facecolor('#1a1a1a')  # Dark background
        ax.set_facecolor('#2d2d2d')  # Darker plot area
        
        # Use bright, contrasting colors for dark background
        colors = ['#00ff88', '#ff4444', '#44aaff', '#ffaa00', '#ff00ff', '#00ffff']
        
        for i, (stock_symbol, data) in enumerate(stocks_data.items()):
            if not data:
                continue
                
            dates = []
            prices = []
            
            for item in data:
                try:
                    date_str = item.get('date', '')
                    price = float(item.get('price', 0))
                    
                    if date_str and price > 0:
                        if isinstance(date_str, str):
                            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                        else:
                            date_obj = date_str
                        
                        dates.append(date_obj)
                        prices.append(price)
                except (ValueError, TypeError):
                    continue
            
            if len(dates) >= 2:
                color = colors[i % len(colors)]
                ax.plot(dates, prices, linewidth=3, color=color, marker='o', 
                       markersize=4, label=stock_symbol, alpha=0.9,
                       markeredgecolor='#ffffff', markeredgewidth=1)
        
        # Customize the chart with light colors for dark background
        ax.set_title('Stock Price Comparison Chart', fontsize=18, fontweight='bold', 
                    pad=25, color='#ffffff')
        ax.set_xlabel('Date', fontsize=14, color='#ffffff')
        ax.set_ylabel('Price (USD)', fontsize=14, color='#ffffff')
        
        # Format x-axis dates with light colors
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        plt.xticks(rotation=45, color='#ffffff')
        plt.yticks(color='#ffffff')
        
        # Add grid and legend with dark theme styling
        ax.grid(True, alpha=0.6, color='#555555', linestyle='-', linewidth=0.5)
        legend = ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('#333333')
        legend.get_frame().set_edgecolor('#ffffff')
        for text in legend.get_texts():
            text.set_color('#ffffff')
        
        # Style the chart spines with bright colors
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#ffffff')
        ax.spines['bottom'].set_color('#ffffff')
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        
        plt.tight_layout()
        
        # Don't save comparison charts automatically (keep folder clean)
        # filename = f"comparison_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        # plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        
        # Don't show plot to avoid popup windows
        
        plt.close()
        
    except Exception as e:
        if HAS_RICH:
            console.print(f"[red]❌ Error creating matplotlib comparison chart: {e}[/red]")
        else:
            print(f"❌ Error creating matplotlib comparison chart: {e}")

# Removed get_sample_stock_data function - no more fake data generation

def demo_charts():
    """Demo the charting capabilities with real data"""
    if HAS_RICH:
        console.print("\n[bold cyan]🎨 Chart Demo - Testing Real Data Fetching[/bold cyan]")
        console.print("[yellow]Note: This will attempt to fetch real market data for demo purposes[/yellow]")
    else:
        print("\n🎨 Chart Demo - Testing Real Data Fetching")
        print("Note: This will attempt to fetch real market data for demo purposes")
    
    # Try to demo with well-known stocks that should have good data
    demo_stocks = ["AAPL", "MSFT"]
    
    if HAS_RICH:
        console.print(f"[yellow]Attempting to analyze {', '.join(demo_stocks)} for charting demo...[/yellow]")
    else:
        print(f"Attempting to analyze {', '.join(demo_stocks)} for charting demo...")
    
    # Use the existing analyze function which now handles real data properly
    analyze_specific_stocks(demo_stocks)
    
    if HAS_RICH:
        console.print("\n[green]✅ Chart demo completed![/green]")
        console.print("[dim]Charts use real market data when available:[/dim]")
        if HAS_PLOTEXT:
            console.print("[dim]  • Plotext for terminal charts[/dim]")
        if HAS_MATPLOTLIB:
            console.print("[dim]  • Matplotlib for PNG exports[/dim]")
        console.print("[dim]  • ASCII fallback when libraries unavailable[/dim]")
        console.print("[dim]  • Error messages when data unavailable (no fake data)[/dim]")
    else:
        print("\n✅ Chart demo completed!")

def handle_search_request(user_input, stocks):
    """Handle web search requests for any information"""
    if HAS_RICH:
        console.print(f"\n[bold blue]🔍 Web Search Request[/bold blue]")
        if stocks:
            console.print(f"[yellow]Related to: {', '.join(stocks)}[/yellow]")
    else:
        print("\n🔍 Web Search Request")
        if stocks:
            print(f"Related to: {', '.join(stocks)}")
    
    # Simple, direct search query to get raw compound-beta results
    search_query = user_input
    
    if HAS_RICH:
        with console.status("[bold blue]🌐 Searching with compound-beta model...") as status:
            response = web_agent.run(search_query)
        
        # Show raw compound-beta output with minimal processing
        console.print(f"\n[bold green]📡 Raw Compound-Beta Output:[/bold green]")
        console.print("[dim]Direct output from Groq's compound-beta web search model[/dim]")
        console.print("-" * 80)
        
        render_markdown_with_alternatives(
            response.content,
            title="🔍 Compound-Beta Search Results",
            border_style="blue"
        )
    else:
        print("🌐 Searching with compound-beta model...")
        response = web_agent.run(search_query)
        
        print(f"\n📡 Raw Compound-Beta Output:")
        print("Direct output from Groq's compound-beta web search model")
        print("-" * 80)
        print(response.content)
    
    return response.content

def get_web_enhanced_analysis(stock, company_name, sector):
    """Get web search results to enhance stock analysis"""
    search_query = f"""Search for recent information about {company_name} ({stock}):
    
    Please find relevant recent developments, focusing on:
    - Recent company news and announcements
    - Strategic initiatives and business developments  
    - Technology innovations or product launches
    - Industry trends affecting the {sector} sector
    - Partnership announcements or acquisitions
    - Regulatory developments if relevant
    
    Provide factual, recent information with sources. Focus on business developments rather than just financial metrics."""
    
    try:
        if HAS_RICH:
            with console.status(f"[bold blue]🌐 Gathering insights for {stock}...") as status:
                web_response = web_agent.run(search_query)
        else:
            print(f"🌐 Gathering insights for {stock}...")
            web_response = web_agent.run(search_query)
        
        return web_response.content
    except Exception as e:
        if HAS_RICH:
            console.print(f"[yellow]⚠️  Web search unavailable for {stock}: {str(e)[:50]}...[/yellow]")
        else:
            print(f"⚠️  Web search unavailable for {stock}: {str(e)[:50]}...")
        return None

if __name__ == "__main__":
    # Check if running interactively
    try:
        main_interactive()
    except KeyboardInterrupt:
        if HAS_RICH:
            console.print("\n[yellow]👋 Goodbye![/yellow]")
        else:
            print("\n👋 Goodbye!")
    except Exception as e:
        if HAS_RICH:
            console.print(f"[red]❌ Error: {e}[/red]")
        else:
            print(f"❌ Error: {e}")