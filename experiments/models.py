
from openai import OpenAI
from azure.identity import DefaultAzureCredential, AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI
import boto3
import json
import os
from google import genai
from google.genai import types
from google.genai.types import HttpOptions
# Adjust path accordingly



def _vllm_base_url(port: int) -> str:
    """Return the base URL for a vLLM server on the given port.

    Checks VLLM_HOST_{port} env var first (for multi-node TACC runs where the
    model is on a different node).  Falls back to localhost for single-node runs.

    To use from a TACC startup script:
        export VLLM_HOST_8002=$NODE_2_IP
    """
    host = os.environ.get(f"VLLM_HOST_{port}", "localhost")
    return f"http://{host}:{port}/v1"


def _parse_vllm_model_path(model_path: str) -> tuple:
    """Parse optional @port suffix from a local vLLM model_path.

    Format: ``/path/to/model@PORT``  (e.g. ``/path/to/qwen27b@8002``)
    Returns ``(actual_path, port)``; default port is 8001.
    """
    at_idx = model_path.rfind("@")
    if at_idx != -1:
        suffix = model_path[at_idx + 1 :]
        if suffix.isdigit():
            return model_path[:at_idx], int(suffix)
    return model_path, 8001


def agent_response(model_path: str, prompt: str, max_tokens: int = 512):
    if model_path == "gemini":
        client = genai.Client(http_options=HttpOptions(api_version="v1"))
        response_content = client.models.generate_content(
        # model="gemini-2.5-flash-lite-preview-09-2025", # model = "deployment_name".
        model="gemini-3.1-flash-lite-preview",
        config=types.GenerateContentConfig(
        system_instruction='''You are playing an adventure game. You are given rules that describe what is needed to achieve a goal.''',
        temperature=0.1,
        thinking_config=types.ThinkingConfig(
        include_thoughts=True, thinking_budget = 512)),
        contents=f"""{prompt}""")
    
        # Extract token usage from Gemini response
        usage = response_content.usage_metadata
        token_counts = {
            'prompt_tokens': usage.prompt_token_count,
            'candidates_tokens': usage.candidates_token_count,
            'total_tokens': usage.total_token_count
        }
        
        # Extract response and reasoning
        reasoning = None
        response = None
        for part in response_content.candidates[0].content.parts:
            if not part.text:
                continue
            if part.thought:
                reasoning = part.text
            else:
                response = part.text
        return response, reasoning, token_counts
    elif model_path == 'claude4.5':
        bedrock = boto3.client(service_name='bedrock-runtime')
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 300,
            "messages": [
                {
                    "role": "user",
                    "content": f"{prompt}"
                }
            ], 
            "output_config": {
            "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                "action": {
                    "type": "string",
                    "description": "action"
                },
                "argument": {
                    "type": "string",
                    "description": "argument"
                },
                "reasoning": {
                    "type": "string",
                    "description": "reasoning"
                }
                },
                "required": [
                "action",
                "argument",
                "reasoning"
                ],
                "additionalProperties": False
            }
            }
            }
        })

        modelId = 'global.anthropic.claude-sonnet-4-5-20250929-v1:0'
        accept = 'application/json'
        contentType = 'application/json'

        response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)

        response_body = json.loads(response.get('body').read())
        usage = response_body['usage']
        token_counts = {
            'prompt_tokens': usage['input_tokens'],
            'candidates_tokens': usage['output_tokens'],
            'total_tokens': usage['input_tokens'] + usage['output_tokens']
        }
        return response_body['content'][0]['text'], None, token_counts

    elif model_path == 'gpt-5.4-mini':
        token_provider = get_bearer_token_provider(
            AzureCliCredential(), "https://cognitiveservices.azure.com/.default"
        )

        client = AzureOpenAI(
            api_version="2024-12-01-preview",  # <-- update this line
            azure_endpoint="https://discolab-openai.openai.azure.com/",
            azure_ad_token_provider=token_provider
        )
        response_content = client.chat.completions.create(
        model="gpt-5.4-mini",
        timeout=120,
        reasoning_effort="low",
        max_completion_tokens= max(max_tokens, 3000),
        messages=[{"role": "user", "content": f"""{prompt}"""}]
    )
        usage = response_content.usage
        token_counts = {
            'prompt_tokens': usage.prompt_tokens,              # FIXED: was prompt_token_count
            'candidates_tokens': usage.completion_tokens,      # FIXED: was candidates_token_count
            'total_tokens': usage.total_tokens                # FIXED: was total_token_count
        }
        content = response_content.choices[0].message.content
        return (content.strip() if content is not None else ""), None, token_counts

    elif model_path == 'qwen_thinking':
        _, _port = _parse_vllm_model_path(model_path)
        client = OpenAI(
        api_key="EMPTY",
        base_url=_vllm_base_url(_port),
        timeout=180,
    )
        model = client.chat.completions.create(
            # model="/home/az22555/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Thinking-2507/snapshots/144afc2f379b542fdd4e85a1fcd5e1f79112d95d"
            # , # model = "deployment_name".
            model="/home/az22555/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18",
            temperature=0.2,
            frequency_penalty=1.2,
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ],
            extra_body={
        "repetition_penalty": 1.2}

        )
        full_response = model.choices[0].message.content.strip()

        thinking = None
        response_content = full_response
    
        if '</think>' in full_response:
            parts = full_response.split('</think>', 1)
            # Extract thinking (remove <think> tag)
            thinking = parts[0].replace('<think>', '').strip()
            # Extract JSON response (everything after </think>)
            response_content = parts[1].strip()
        return response_content, thinking, None  # Return None for token_counts for non-Gemini 
    
    elif model_path.startswith('tacc/'):
        model_name = model_path[len('tacc/'):]
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url="https://ai.tejas.tacc.utexas.edu/v1",
            timeout=180,
        )
        model = client.chat.completions.create(
            model=model_name,
            temperature=1.0,
            top_p=0.95,
            presence_penalty=1.5,
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        response_content = model.choices[0].message.content.strip()
        return response_content, None, None

    elif model_path == 'openai/gpt-oss-120b':
        _actual_path, _port = _parse_vllm_model_path(model_path)
        client = OpenAI(
        api_key="EMPTY",
        base_url=_vllm_base_url(_port),
        timeout=180,
    )
        model = client.chat.completions.create(
            model=_actual_path,
            temperature=1.0,
            top_p=0.95,
            presence_penalty=1.5,
            messages=[
                {"role": "user", "content": prompt}
            ],
            extra_body={
                "top_k": 20,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        response_content = model.choices[0].message.content.strip()
        return response_content, None, None  # Return None for token_counts for non-Gemini

    else:
        _actual_path, _port = _parse_vllm_model_path(model_path)
        client = OpenAI(
        api_key="EMPTY",
        base_url=_vllm_base_url(_port),
        timeout=180,
    )
        model = client.chat.completions.create(
            model=_actual_path,
            temperature=1.0,
            top_p=0.95,
            presence_penalty=1.5,
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ],
            extra_body={
                "top_k": 20,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        response_content = model.choices[0].message.content.strip()
        return response_content, None, None  # Return None for token_counts for non-Gemini

_JSON_CONVERTER_INSTRUCTION = (
    "You are a helpful assistant that understands the text and EXTRACTS the final action "
    "the model wants to perform at this turn. DO NOT MAKE ANY CHANGES TO THE TEXT."
    "Then CONVERT it into JSON format following schema {json_schema}. "
    "The action and argument should be in one of the following format: "
    "go [location], get [object], buy [object], defeat [enemy], rescue [npc], "
    "check_inventory, check_location."
)


def json_converter(response_content: str, json_schema: dict) -> str:
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )

    client = AzureOpenAI(
        api_version="2024-12-01-preview",  # <-- update this line
        azure_endpoint="https://discolab-openai.openai.azure.com/",
        azure_ad_token_provider=token_provider
    )

    instruction = _JSON_CONVERTER_INSTRUCTION.format(json_schema=json_schema)
    converter = client.chat.completions.create(
        model="gpt-4o-mini",
        timeout=120,
        messages=[{"role": "user", "content": f"{instruction} Here is the model generation: {response_content}"}],
    )
    return converter.choices[0].message.content.strip()

def json_converter_small(response_content:str, json_schema: dict) -> str:
    import httpx
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
        timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0),
    )

    instruction = _JSON_CONVERTER_INSTRUCTION.format(json_schema=json_schema)
    response = client.chat.completions.create(
    model="/home/az22555/.cache/huggingface/hub/models--osmosis-ai--Osmosis-Structure-0.6B/snapshots/7986fe308790f94b768060255c06823445f2922c",
    messages=[{"role": "user", "content": f"{instruction} Here is the model generation: {response_content}"}],
    temperature=0,
    max_tokens=1000,
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "action", "schema": json_schema},
    },
    )
    # structured = json.dumps(json.loads(response.choices[0].message.content), indent=2)
    return response.choices[0].message.content.strip()


def json_converter_gemini(response_content:str, json_schema: dict) -> str:
    client = genai.Client(http_options=HttpOptions(api_version="v1"))
    response_content = client.models.generate_content(
    model="gemini-2.5-flash-lite-preview-09-2025", # model = "deployment_name".
    config=types.GenerateContentConfig(
    system_instruction=_JSON_CONVERTER_INSTRUCTION.format(json_schema=json_schema),
    temperature=0.1,
    thinking_config=types.ThinkingConfig(
    include_thoughts=False, thinking_budget = 0)),
    contents=f"""Here is the model generation:{response_content}""")

    # Extract token usage from Gemini response
    usage = response_content.usage_metadata
    token_counts = {
        'prompt_tokens': usage.prompt_token_count,
        'candidates_tokens': usage.candidates_token_count,
        'total_tokens': usage.total_token_count
    }
    
    # Extract response and reasoning
    reasoning = None
    response = None
    for part in response_content.candidates[0].content.parts:
        if not part.text:
            continue
        if part.thought:
            reasoning = part.text
        else:
            response = part.text
    return response


_TEACHER_JSON_REPAIR_INSTRUCTION = (
    "You are a helpful assistant. The following text is an LLM response that should contain "
    "structured data but may be malformed or wrapped in extra prose. "
    "Extract the relevant data and return it as valid JSON strictly following the provided schema. "
    "Do NOT invent or add data that is not present in the input."
)


def teacher_json_repair_small(text: str, json_schema: dict) -> str:
    """Repair/extract teacher JSON output using the local small model."""
    import httpx
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
        timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0),
    )
    response = client.chat.completions.create(
        model="/home/az22555/.cache/huggingface/hub/models--osmosis-ai--Osmosis-Structure-0.6B/snapshots/7986fe308790f94b768060255c06823445f2922c",
        messages=[{
            "role": "user",
            "content": (
                f"{_TEACHER_JSON_REPAIR_INSTRUCTION}\n"
                f"Schema: {json_schema}\n"
                f"LLM response:\n{text}"
            ),
        }],
        temperature=0,
        max_tokens=1000,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "output", "schema": json_schema},
        },
    )
    return response.choices[0].message.content.strip()


def teacher_json_repair_gemini(text: str, json_schema: dict) -> str:
    """Repair/extract teacher JSON output using Gemini."""
    client = genai.Client(http_options=HttpOptions(api_version="v1"))
    resp = client.models.generate_content(
        model="gemini-2.5-flash-lite-preview-09-2025",
        config=types.GenerateContentConfig(
            system_instruction=_TEACHER_JSON_REPAIR_INSTRUCTION,
            temperature=0.1,
            thinking_config=types.ThinkingConfig(include_thoughts=False, thinking_budget=0),
        ),
        contents=f"Schema: {json_schema}\nLLM response:\n{text}",
    )
    for part in resp.candidates[0].content.parts:
        if part.text and not part.thought:
            return part.text
    return ""