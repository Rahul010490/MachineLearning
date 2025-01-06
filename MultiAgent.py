import os
import openai
from time import sleep
from serpapi import GoogleSearch
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)


# ------------------------------------------------------------------------------
# 1. UserInterfaceAgent
#    Handles user interaction (input prompts) and displays results
# ------------------------------------------------------------------------------
class UserInterfaceAgent:
    def __init__(self):
        """Agent responsible for collecting user input and displaying responses."""
        pass

    def get_user_query(self) -> str:
        """Ask the user for their search query."""
        query = input("Hello! What would you like to research today?\n> ")
        return query

    def display_response(self, response: str):
        """Display final summarized information to the user."""
        print("\n===== Summary of Research =====")
        print(response)
        print("\n===============================")


# ------------------------------------------------------------------------------
# 2. InternetSearchAgent
#    Interfaces with a search API (e.g., Google via SERPAPI) to retrieve top results
#    Includes rate limiting and error handling
# ------------------------------------------------------------------------------
class InternetSearchAgent:
    def __init__(self, serpapi_key: str, num_results: int = 10):
        """
        Agent responsible for searching the internet.
        
        :param serpapi_key: Your SERPAPI API key
        :param num_results: How many search results to fetch
        """
        self.serpapi_key = serpapi_key
        self.num_results = num_results

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=8),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception)
    )
    def search(self, query: str) -> list:
        """
        Performs an internet search using SERPAPI (as an example) and returns the top results.
        This method will retry up to 3 times with exponential backoff if errors occur.
        
        :param query: The user query (string)
        :return: A list of snippet texts from the top results
        """
        print(f"[InternetSearchAgent] Searching for: {query}")
        
        # Basic rate-limiting approach (sleep 1 second before each search).
        # Adjust or remove this depending on your usage and actual rate-limits.
        sleep(1)

        # Configure search parameters for SerpApi
        search_params = {
            "engine": "google",
            "q": query,
            "api_key": self.serpapi_key,
            "num": self.num_results,
        }

        # Perform the search
        try:
            search = GoogleSearch(search_params)
            results = search.get_dict()
        except Exception as e:
            print(f"[InternetSearchAgent] Error during search: {e}")
            # Reraise the exception to trigger the tenacity retry
            raise

        # Extract snippets from the search results
        snippets = []
        organic_results = results.get("organic_results", [])
        for item in organic_results:
            snippet = item.get("snippet", "")
            if snippet:
                snippets.append(snippet)

        if not snippets:
            print("[InternetSearchAgent] No snippets found.")
        else:
            print(f"[InternetSearchAgent] Retrieved {len(snippets)} snippets from search.")
        return snippets


# ------------------------------------------------------------------------------
# 3. SummarizerAgent
#    Uses OpenAI's GPT model to summarize text
#    Includes rate limiting and error handling
# ------------------------------------------------------------------------------
class SummarizerAgent:
    def __init__(self, openai_api_key: str, model_name: str = "gpt-3.5-turbo"):
        """
        Agent responsible for summarizing text using OpenAI's GPT models.
        
        :param openai_api_key: Your OpenAI API key
        :param model_name: The OpenAI model to use (e.g., 'gpt-3.5-turbo')
        """
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        openai.api_key = self.openai_api_key

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=8),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(openai.error.OpenAIError)
    )
    def summarize(self, texts: list, user_query: str) -> str:
        """
        Summarize the list of texts in the context of the user query.
        This method will retry up to 3 times with exponential backoff if OpenAI errors occur.
        
        :param texts: List of strings (e.g. search result snippets)
        :param user_query: The original user query for context
        :return: A string summarizing the findings
        """
        print("[SummarizerAgent] Summarizing search results with OpenAI...")

        # Rate limiting: sleep 1 second before each request (example).
        sleep(1)

        # Combine all snippets into a single string
        combined_text = "\n\n".join(texts)

        # System message instructions
        system_message = {
            "role": "system",
            "content": (
                "You are a helpful research assistant. "
                "Please provide a concise and clear summary of the key points from the text provided. "
                "Focus on the user's query and relevant information."
            )
        }
        # User message with all the snippets
        user_message = {
            "role": "user",
            "content": (
                f"The user wants information about: '{user_query}'. "
                "Below are snippets from top search results:\n\n"
                f"{combined_text}\n\n"
                "Please provide a concise summary, with key points that answer the user's research query."
            )
        }

        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[system_message, user_message],
                temperature=0.7,
            )
        except openai.error.OpenAIError as e:
            print(f"[SummarizerAgent] OpenAI API Error: {e}")
            raise  # Reraise to trigger the tenacity retry

        # Extract the assistant's response
        summarized_text = response["choices"][0]["message"]["content"]
        print("[SummarizerAgent] Summarization complete.")
        return summarized_text


# ------------------------------------------------------------------------------
# 4. ConversationOrchestrationAgent
#    Orchestrates the entire workflow among other agents
# ------------------------------------------------------------------------------
class ConversationOrchestrationAgent:
    def __init__(self, 
                 user_agent: UserInterfaceAgent, 
                 search_agent: InternetSearchAgent, 
                 summarizer_agent: SummarizerAgent):
        """
        Orchestrates the workflow: ask user, search internet, summarize, and display result.
        """
        self.user_agent = user_agent
        self.search_agent = search_agent
        self.summarizer_agent = summarizer_agent

    def run(self):
        """
        Main method to coordinate user input, searching, summarizing, and displaying the result.
        """
        # Step 1: Get the user's query
        user_query = self.user_agent.get_user_query()
        if not user_query.strip():
            print("[ConversationOrchestrationAgent] No valid query provided.")
            return

        # Step 2: Use the InternetSearchAgent to get top search results
        try:
            search_results = self.search_agent.search(user_query)
        except Exception as e:
            error_message = f"An error occurred while searching the web: {e}"
            self.user_agent.display_response(error_message)
            return

        if not search_results:
            self.user_agent.display_response("No search results found.")
            return

        # Step 3: Summarize the search results
        try:
            summary = self.summarizer_agent.summarize(search_results, user_query)
        except Exception as e:
            error_message = f"An error occurred while summarizing the content: {e}"
            self.user_agent.display_response(error_message)
            return

        # Step 4: Display the summary back to the user
        self.user_agent.display_response(summary)


# ------------------------------------------------------------------------------
# 5. Main entry point to run the multi-agent chatbot
# ------------------------------------------------------------------------------
def main():
    # Retrieve API keys from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    serpapi_key = os.getenv("SERPAPI_API_KEY")

    if not openai_api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    if not serpapi_key:
        raise ValueError("Please set the SERPAPI_API_KEY environment variable.")

    # Instantiate agents
    user_agent = UserInterfaceAgent()
    internet_search_agent = InternetSearchAgent(serpapi_key=serpapi_key, num_results=10)
    summarizer_agent = SummarizerAgent(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")

    # Create the orchestrator
    orchestrator = ConversationOrchestrationAgent(
        user_agent=user_agent,
        search_agent=internet_search_agent,
        summarizer_agent=summarizer_agent
    )

    # Run the conversation
    orchestrator.run()


if __name__ == "__main__":
    main()
