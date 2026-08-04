import json
import logging
import re
import sys
from typing import Optional

import fire
import requests
import torch
from bs4 import BeautifulSoup
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache

logger = logging.getLogger(__name__)

if not hasattr(DynamicCache, "seen_tokens"):
    DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    logger.debug("Patched DynamicCache.seen_tokens -> get_seq_length()")

if not hasattr(DynamicCache, "get_max_length"):
    DynamicCache.get_max_length = lambda self, *a, **kw: (
        self.get_max_cache_shape() if hasattr(self, "get_max_cache_shape") else 0
    )
    logger.debug("Patched DynamicCache.get_max_length -> get_max_cache_shape()")

if not hasattr(DynamicCache, "get_usable_length"):
    DynamicCache.get_usable_length = lambda self, new_seq_length, layer_idx=None: (
        self.get_seq_length() if layer_idx is None or not hasattr(self, "get_seq_length") else self.get_seq_length(layer_idx)
    )
    logger.debug("Patched DynamicCache.get_usable_length -> get_seq_length()")

SYSTEM_PROMPT = """You are WebAgent, an AI assistant that can browse the internet to research topics and provide summaries. You have access to a web search tool.

You MUST follow this workflow:
1. When asked about a topic, use the web_search tool to find current information
2. Wait for the search results
3. Read and analyze the results
4. Provide a comprehensive, well-structured summary

<tool_calls>
[{"name": "web_search", "arguments": {"query": "<search query>"}}]
</tool_calls>"""

SEARCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def setup_logging(verbose: bool = False):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def extract_tool_call(text: str) -> Optional[dict]:
    pattern = r'<tool_calls>\s*(\[.*?\])\s*</tool_calls>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))[0]
        except (json.JSONDecodeError, IndexError):
            return None
    return None


def web_search(query: str) -> str:
    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    try:
        resp = requests.get(search_url, headers=SEARCH_HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for g in soup.find_all("div", class_="g")[:5]:
            title_el = g.find("h3")
            snippet_el = g.find("div", class_="VwiC3b") or g.find("span", class_="aCOpRe")
            if title_el:
                title = title_el.get_text(strip=True)
                snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                results.append(f"{title}: {snippet}")
        if results:
            return "\n".join(results[:5])
        return f"Search results for '{query}': [Simulated content] The topic covers recent developments and key information based on web sources."
    except Exception as e:
        return f"[Search results for: {query}] Unable to fetch live results ({e}). Using fallback content about this topic."


class WebAgent:
    def __init__(
        self,
        model_path: str = "./output",
        base_model: str = "microsoft/Phi-3-mini-4k-instruct",
        temperature: float = 0.7,
        max_tokens: int = 512,
        verbose: bool = False,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading model from %s", model_path)

        model_config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)

        rope_scaling = getattr(model_config, "rope_scaling", None)
        if isinstance(rope_scaling, dict) and "type" not in rope_scaling:
            logger.warning("Invalid rope_scaling config detected — disabling")
            model_config.rope_scaling = None

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            config=model_config,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )

        base.config.use_cache = False
        if hasattr(base, "generation_config"):
            base.generation_config.use_cache = False
        logger.info("Disabled KV cache for Phi-3 DynamicCache compatibility")

        try:
            self.model = PeftModel.from_pretrained(base, model_path)
            logger.info("Loaded fine-tuned LoRA adapter from %s", model_path)
        except Exception as e:
            logger.warning("No LoRA adapter found at %s, using base model: %s", model_path, e)
            self.model = base

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path if model_path else base_model,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.unk_token if self.tokenizer.unk_token else self.tokenizer.eos_token
            )
        logger.info("WebAgent initialized (device=%s, temp=%.2f, max_tokens=%d)", self.device, temperature, max_tokens)

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.device)
        input_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                use_cache=False,
            )
        new_ids = outputs[0][input_len:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def run(self, user_query: str, max_tool_rounds: int = 3) -> str:
        logger.info("Processing query: %s", user_query)
        print(f"\n{'='*60}")
        print(f"USER: {user_query}")
        print(f"{'='*60}\n")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]

        for turn in tqdm(range(max_tool_rounds), desc="Agent loop", unit="turn", disable=not self.verbose):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            response = self.generate(prompt)
            print(f"[ASSISTANT TURN {turn + 1}]:\n{response}\n")

            tool_call = extract_tool_call(response)
            if tool_call and tool_call.get("name") == "web_search":
                query = tool_call["arguments"]["query"]
                logger.info("Tool call: web_search(query='%s')", query)
                print(f"[WEB SEARCH]: Searching for '{query}'...")
                search_results = web_search(query)
                logger.debug("Search results (%d chars): %s...", len(search_results), search_results[:200])
                print(f"[SEARCH RESULTS]:\n{search_results[:300]}...\n")

                tool_call_tag = response.split("</tool_calls>")[0] + "</tool_calls>"
                messages.append({"role": "assistant", "content": tool_call_tag})
                messages.append(
                    {"role": "tool", "content": json.dumps({"result": search_results})}
                )
            else:
                logger.info("No tool call detected — final answer")
                print(f"[FINAL ANSWER]:\n{response}\n")
                return response

        logger.warning("Max tool rounds (%d) reached without final answer, forcing generation", max_tool_rounds)
        final_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        final_response = self.generate(final_prompt)
        print(f"[FINAL ANSWER]:\n{final_response}\n")
        return final_response


def interactive(agent: WebAgent):
    print("WebAgent Fine-Tuned Model — Interactive Mode")
    print("Type 'quit' to exit\n")
    while True:
        user_input = input("\n>>> ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue
        agent.run(user_input)


def main(
    model_path: str = "./output",
    base_model: str = "microsoft/Phi-3-mini-4k-instruct",
    mode: str = "interactive",
    query: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
    verbose: bool = False,
):
    setup_logging(verbose=verbose)
    agent = WebAgent(
        model_path=model_path,
        base_model=base_model,
        temperature=temperature,
        max_tokens=max_tokens,
        verbose=verbose,
    )

    if mode == "interactive":
        interactive(agent)
    elif mode == "single":
        if not query:
            logger.error("--query is required in 'single' mode")
            sys.exit(1)
        agent.run(query)
    else:
        logger.error("Unknown mode: %s. Use 'interactive' or 'single'.", mode)


if __name__ == "__main__":
    fire.Fire(main)
