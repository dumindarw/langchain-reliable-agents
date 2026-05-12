from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

client = wrap_openai(OpenAI())

@traceable(run_type="tool")
def get_weather() -> str:
    """ Retreive current weather information """
    return "It is raining today"

# Define the tool schema for OpenAI
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather conditions",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}


def agent(question: str) -> dict:
    """ Agent that uses the weather tool to answer questions about the weather """
    
    messages = [{"role": "user", "content": question}]

    response = client.chat.completions.create(model="gpt-5.4-nano", messages=messages, tools=[WEATHER_TOOL], tool_choice="auto")

    response_message = response.choices[0].message

    if response_message.tool_calls:
        messages.append({
            "role": "assistant",
            "content": response_message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in response_message.tool_calls
            ]
        })

        for tool_call in response_message.tool_calls:
            if tool_call.function.name == "get_weather":
                result = get_weather()

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": "get_weather",
                    "content": result
                })

        # Make second API call with tool results
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            tools=[WEATHER_TOOL],
            tool_choice="auto"
        )
        response_message = response.choices[0].message

    messages.append({"role": "assistant", "content": response_message.content})
    return {"messages": messages, "output": response_message.content}

if __name__ == "__main__":
    result = agent("What is the weather today?")
    print(result["output"])