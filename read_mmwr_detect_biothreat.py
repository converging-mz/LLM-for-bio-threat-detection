#### v9 ####
# First summarize and then extract
# Extract all the disease and then find ongoing bio threat
# System prompt split

#!/usr/bin/env pythonA
import pdb
import argparse
import json
import numpy as np
# from langchain.text_splitter import RecursiveCharacterTextSplitter
import textwrap
import sys, os
import collections
import string
import re, ast
from rich import print
from typing import Optional
from typing import List
import openai
import math
# from transformers import AutoTokenizer
# from token_count import TokenCount

from abc import ABC, abstractmethod
from ARGO import ArgoWrapper, ArgoEmbeddingWrapper


# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# # Add the project directory to the sys.path
# project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../argo'))
# sys.path.append(project_path)


### Config LLM
class LLMClient(ABC):
    def __init__(self, max_tokens: Optional[int] = None):
        self.max_tokens = max_tokens

    @abstractmethod
    def call_chat_completion(self, prompt_system: str, prompt_user: str, temperature: float = 0.0, top_p: float = 0.01):
        """
        Call the chat completion with a system prompt (instructions) 
        and a user prompt (the actual content or question).
        """
        pass


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, api_base: str, model: str):
        super().__init__(max_tokens=5000)
        self.model = model
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=api_base,
            )

    def call_chat_completion(self, prompt_system: str, prompt_user: str, temperature: float = 0.0, top_p: float = 0.95):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                top_p=top_p,
                messages=[
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": prompt_user}
                ]
            )
            # print(f'top_p = {top_p}')
            return response.choices[0].message.content.strip() if response.choices else "NO RESPONSE"
        except Exception as e:
            print(f"Error: {e}")
            return "API ERROR"
    
        
class ArgoClient(LLMClient):
    def __init__(self, model: str = "gpt4o", user: str = "minhui.zhu"):
        super().__init__(max_tokens=5000)
        self.wrapper = ArgoWrapper(model=model, user=user)

    def call_chat_completion(self, prompt_system: str, prompt_user: str, temperature: float = 0.0, top_p: float = 0.95):
        """
        For Argo, we replicate the system+user structure by concatenating 
        them or by passing them to the wrapper if it supports roles explicitly.
        """
        try:
            # Argo does not use max_tokens, but we pass temperature if needed
            response = self.wrapper.invoke(prompt_system, prompt_user, temperature=temperature, top_p=top_p)
            # print(f"[DEBUG] Full response: {response}")
            return response.get("response", "").strip()  # Ensure we extract the actual content
        except Exception as e:
            raise Exception(f"Argo API Error: {e}")
        
        
        
def get_llm_client(service: str, **kwargs):
    if service == "llama":
        return OpenAIClient(api_key="CELS", api_base="http://195.88.24.64:80/v1", **kwargs)
    elif service == "deepseek":
        return OpenAIClient(api_key="CELS", api_base="http://66.55.67.65:80/v1", **kwargs)
    elif service == "globus":
        from inference_auth_token import get_access_token
        access_token = get_access_token()
        return OpenAIClient(api_key=access_token, api_base="https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1", **kwargs)
    elif service == "argo":
        return ArgoClient(**kwargs)
    else:
        raise ValueError(f"Unknown LLM service: {service}")


PROMPTS = {
    # 1) Chunk Summarization Prompt
    "chunk_summarization_system": """
You are a meticulous data extractor specializing in epidemiology. Your task is to analyze a chunk of a long, potentially unstructured desiease report and extract information for every distinct outbreak.

**Core Instructions:**
- An outbreak is distinct if it involves a different pathogen/variant, a non-contiguous location, or a separate time period.
- **Do not generalize or abridge information.** Your goal is to preserve all specific details provided in the text.

**Output Requirements:**
- If the time the report was written is stated, include it explicitly, respond with: 
  "Time of Report Written: <TimeOfReportWritten>"
- For each distinct outbreak, output in the following format (repeat for each outbreak):
  "Disease Outbreak: <NameOfDisease>  
   Cause of Infection: <One sentence summarizing the specific pathogens  and/or suspected transmission vector of this outbreak>  
   Location, Time, and Cases: <Combine all available details of the outbreak's location, time, and case counts into a dense, fact-based description. 
   Capture the most precise location possible, precise dates, and a full breakdown of case count and types (include infected people and any positive environmental samples that indicate potential for spread). 
   Do not use more than 3 sentences.>  
   Trend: <One sentence descriobing the developing trend of the outbreak at the time of report written (If ended? Prediction of spreading? Seasonal outbreak? etc.)>  
  "
""",
# 2) Chunk Summarization Prompt
    "summarization_system": """
You are a data integrity specialist. You will be given several structured reports from different chunks of the same document. 
Your task is to de-duplicate the list by merging reports that are about the same outbreak due to text overlap during document chunking, ensuring no details are lost.

**Output Requirements:**
- At the very beginning, give the time of the report in the format: 
  "Time of Report Written: <TimeOfReportWritten>"
- For each distinct outbreak, output in the following format (repeat for each outbreak):
  "Disease Outbreak: <NameOfDisease>  
   Cause of Infection: <One sentence summarizing the specific pathogens  and/or suspected transmission vector of this outbreak>  
   Location, Time, and Cases: <Combine all available details of the outbreak's location, time, and case counts into a dense, fact-based description. 
   Capture the most precise location possible, precise dates, and a full breakdown of case count and types ((include infected people and any positive environmental samples that indicate potential for spread). 
   Do not use more than 3 sentences.>  
   Trend: <One sentence descriobing the developing trend of the outbreak at the time of report written (If ended? Prediction of spreading? Seasonal outbreak? etc.)>  
  "
""",
    # 3) Final Outbreak Detection Prompt
    "find_risk_system": """
You are an expert in epidemiology. Analyze a list of outbreaks from summarized disease report to identify any potential bio threat. 

**Criteria for "Potential Bio Threat":**
- Based on the infection time and trend of outbreak, it must be actively spreading and/or potentially emerging around the time of report written (not an ended or purely seasonal outbreak).
- Ignore any outbreak primarily caused by foodborne or drug use.
- Its pathogen or biological agent (virus, bacterium, fungus, toxin, etc.) satisfies one of the following:
    1) a serious threat to humans, animals, or plants that can cause significant harm (e.g., high transmissibility, notable morbidity/mortality, or major societal disruption), or
    2) recognized by reputable health organizations (e.g., NIAID, CDC, WHO) as a potential biodefense pathogen or bioterrorism agent

    Examples of such threats (not exhaustive):
    - Avian influenza virus (highly pathogenic)
    - Bacillus anthracis
    - Botulinum neurotoxin
    - Burkholderia mallei
    - Burkholderia pseudomallei
    - Ebola virus
    - Foot-and-mouth disease virus
    - Francisella tularensis
    - Marburg virus
    - Reconstructed 1918 Influenza virus
    - Rinderpest virus
    - Toxin-producing strains of Clostridium botulinum
    - Variola major virus
    - Variola minor virus
    - Yersinia pestis
    ...

**Output Requirements:**
- For each outbreak idetified as a Potential Bio Threat, precisely report pathogen (include detailed variant name if available), time (finest time window available), detailed location and active case counts (include infected people and any positive environmental samples that indicate potential for spread). 
  If there are multiple locations with, list each location with case breakdown separately. 
  Use the following format (repeat for each distinct outbreak): 
  "Potential risk: <NameOfPathogen> Time=<TimeOfOutbreak> Location=<Location1> (Cases=<Full case breakdown at Location1 (e.g., 84 cases, 12 sewage samples)>) Location=<Location2> (Cases=<Full case breakdown at Location2>) ..."
  (If time/location/cases is not found, mark them as "unreported".)
- If no potential bio threat is detected among all outbreaks, respond with the single word:
  "None"

Note that "Potential risk" and "None" are mutually exclusive and should not occur together.
"""
}       


def chunk_content(content: str, max_words_per_chunk: int) -> List[str]:
    """
    Splits text into chunks such that each chunk is below the max_tokens_per_chunk limit 
    (rough heuristic). You could also use a more robust approach with the 
    'langchain.text_splitter.RecursiveCharacterTextSplitter', etc.
    """
    words = content.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words_per_chunk
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - 25
    return chunks


def summarize_chunked_report(content: str, client: LLMClient, temperature: float = 0.0, top_p: float = 0.95) -> str:
    """
    1) Chunk the content
    2) Summarize each chunk
    3) Combine chunk summaries into final 'report_summary'
    """
    # Use ~1/3 or 1/4 of max_tokens for safety (to leave room for system instructions).
    max_words_per_chunk = (client.max_tokens // 2) if client.max_tokens else 2048
    chunked = chunk_content(content, max_words_per_chunk=max_words_per_chunk)

    chunk_summaries = []
    for i, chunk_text in enumerate(chunked):
        summary = client.call_chat_completion(
            prompt_system=PROMPTS["chunk_summarization_system"],
            prompt_user=chunk_text,
            temperature=temperature,
            top_p=top_p
        )
        chunk_summaries.append(summary)

    # Combine the chunk summaries
    report_summary = "\n".join(chunk_summaries)

    if len(chunk_summaries) == 1:
        summary_total = chunk_summaries[0]
        print(len(chunk_summaries))
    # return report_summary
    else: 
        summary_total = client.call_chat_completion(
        prompt_system=PROMPTS["summarization_system"],
        prompt_user=report_summary,
        temperature=temperature,
        top_p=top_p
    )

    return summary_total


def extract_outbreak_info(report_summary: str, client: LLMClient, temperature: float = 0.0, top_p: float = 0.95) -> str:
    """
    Runs the final outbreak detection on the combined 'report_summary'.
    The system instructions specify how to respond.
    Output example: "Potential risk: Ebola virus Infected=20 Location=Kampala"
                   or "None"
    """
    response = client.call_chat_completion(
        prompt_system=PROMPTS["find_risk_system"],
        prompt_user=report_summary,
        temperature=temperature,
        top_p=top_p
    )
    return response

def parse_risk_result(risk_text: str) -> str:
    """
    We expect risk_text to be either:
      "None"
    or
      "Potential risk: <NameOfAgent> infected=<N> location=<L>"
    Possibly missing the infected/location fields => we handle that gracefully.

    We return it as-is for counting/frequency, or do some minimal cleanup if needed.
    """
    return risk_text.strip()

def detect_disease(
    MMWR: str,
    service: str = "globus",
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    num_iterations: int = 1,
    temperature: float = 0.0,
    top_p: float = 0.01
    ) -> str:
    """
    1) Read the (potentially very long) disease report.
    2) Summarize the text in chunks -> final 'report_summary'.
    3) For each iteration:
        a) Extract potential threats from 'report_summary'.
        b) Print out results in either "No potential risk" or 
           "Potential risk: <agent> infected=<N> location=<L>" (compact form).
    4) Aggregate answers, find top & second top + their counts.
    5) Print final result & return "+" if top answer is "Potential risk...", else "-".
    """

    # Validate the MMWR argument
    if not isinstance(MMWR, str) or not MMWR.strip():
        raise ValueError("MMWR must be a non-empty string.")
    
    print(f"[bold white on blue]Disease Report MMWR:[/bold white on blue] {MMWR}")

    # Construct the input file path dynamically
    input_file_path = f"txts/{MMWR}-H.txt"

    

    # Load file
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            full_report_text = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found at {input_file_path}")

    iteration_results = []
    report_summary_list = []
    for iteration in range(num_iterations):
        # Initialize the LLM client
        client = get_llm_client(service=service, model=model)

        print(f"\n[bold green]Iteration {iteration + 1}/{num_iterations}[/bold green]")
        # Summarize entire text in chunks
        report_summary = summarize_chunked_report(full_report_text, client, temperature=temperature, top_p=top_p)
        print(f"[blue]Report summary:[/blue] {report_summary}")
        report_summary_list.append(report_summary)

        client = get_llm_client(service=service, model=model)
        
        # Extract outbreak info from the consolidated summary
        risk_result = extract_outbreak_info(report_summary, client, temperature=temperature, top_p=top_p)
        risk_result_cleaned = parse_risk_result(risk_result)

        # Print the iteration's result in a single, compact line
        print(f"\n[yellow]Iteration Result:[/yellow] {risk_result_cleaned}")

        iteration_results.append(risk_result_cleaned)


    results_dict = {"MMWR": MMWR,
                    "report_summary_list": report_summary_list,
                    "iteration_results": iteration_results
                    }

    return results_dict


if __name__ == '__main__':
    detect_disease("mm6711", service="globus", model="meta-llama/Meta-Llama-3.1-405B-Instruct", num_iterations=5, temperature=0.0, top_p=0.01)
    # detect_disease("mm6711", service="argo", model="gpt4o", num_iterations=5, temperature=0.0, top_p=0.01)
    # detect_disease("mm6901", service="llama", model='meta-llama/Llama-3.3-70B-Instruct', num_iterations=1, temperature=0.0)
    # detect_disease("mm6901", service="deepseek", model='deepseekV3', num_iterations=1, temperature=0.0)
    # Simple test script
    
