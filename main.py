from agno.agent import Agent, RunResponse
from agno.models.groq import Groq
from dotenv import load_dotenv

load_dotenv()

agent = Agent(
    model=Groq(id="llama-3.1-8b-instant"),
    markdown=True
)

# Print the response in the terminal
agent.print_response("Share a 2 sentence horror story.")