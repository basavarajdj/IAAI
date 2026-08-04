import json
import random
from typing import Dict, List

import numpy as np

SYSTEM_PROMPT = """You are WebAgent, an AI assistant that can browse the internet to research topics and provide summaries. You have access to a web search tool.

You MUST follow this workflow:
1. When asked about a topic, use the web_search tool to find current information
2. Wait for the search results
3. Read and analyze the results
4. Provide a comprehensive, well-structured summary

<tool_calls>
[{"name": "web_search", "arguments": {"query": "<search query>"}}]
</tool_calls>"""

TOPICS: List[Dict[str, str]] = [
    {"area": "technology", "query": "quantum computing breakthroughs {year}", "answer": "Quantum computing has seen major breakthroughs in {year}, including advancements in qubit stability and error correction. Companies like IBM, Google, and startups have demonstrated quantum supremacy in specific tasks, with implications for cryptography and drug discovery."},
    {"area": "technology", "query": "AI regulation 2025 global policy", "answer": "Global AI regulation efforts have intensified, with the EU AI Act serving as a landmark framework. The US, China, and UK have respectively introduced guidelines focusing on safety, transparency, and accountability. Key debates center around open-source models vs proprietary systems."},
    {"area": "technology", "query": "fusion energy recent progress 2025", "answer": "Recent progress in fusion energy has been remarkable. ITER continues construction, while private companies like Commonwealth Fusion Systems and Helion have demonstrated significant milestones in plasma containment and net-positive energy experiments."},
    {"area": "technology", "query": "solid state battery advancements 2025", "answer": "Solid-state battery technology has advanced significantly, with Toyota and Samsung announcing production timelines. These batteries offer higher energy density, faster charging, and improved safety compared to traditional lithium-ion batteries."},
    {"area": "technology", "query": "autonomous vehicles 2025 latest", "answer": "Autonomous vehicle technology has progressed with Waymo and Cruise expanding robotaxi services. Tesla's FSD continues to improve, while Chinese companies like Baidu and Pony.ai lead in urban autonomous driving deployments."},
    {"area": "technology", "query": "space exploration 2025 missions", "answer": "Space exploration in 2025 includes NASA's Artemis program aiming for lunar return, SpaceX's Starship testing, and ESA's Jupiter missions. Private space stations and Mars sample return missions are key milestones."},
    {"area": "technology", "query": "gene editing CRISPR 2025 developments", "answer": "CRISPR gene editing has seen breakthroughs in treating sickle cell disease and beta-thalassemia. Newer techniques like base editing and prime editing offer more precise modifications with fewer off-target effects."},
    {"area": "technology", "query": "renewable energy storage solutions 2025", "answer": "Energy storage solutions have expanded beyond lithium-ion to include iron-air batteries, flow batteries, and green hydrogen. Grid-scale storage capacity has doubled, supporting greater renewable energy adoption."},
    {"area": "science", "query": "James Webb telescope discoveries {year}", "answer": "The James Webb Space Telescope continues to revolutionize astronomy with discoveries of ancient galaxies, exoplanet atmospheres, and detailed views of star formation. Recent findings include carbon compounds in early universe and potential biosignatures."},
    {"area": "science", "query": "longevity research 2025 anti aging", "answer": "Longevity research has made strides in understanding cellular senescence and epigenetic reprogramming. Drugs like rapamycin and metformin show promise in clinical trials. Companies are targeting aging as a treatable condition."},
    {"area": "science", "query": "neuralink brain computer interface 2025", "answer": "Neuralink and other BCI companies have demonstrated human trials with brain implants allowing paralyzed patients to control computers. Wireless high-bandwidth interfaces are the next frontier."},
    {"area": "science", "query": "climate change solutions 2025", "answer": "Climate solutions include direct air capture scaling up, methane reduction technologies, and improved carbon accounting. Nature-based solutions like reforestation and soil carbon sequestration complement technological approaches."},
    {"area": "science", "query": "mRNA technology beyond covid vaccines", "answer": "mRNA technology has expanded beyond COVID-19 to include cancer vaccines, rare disease treatments, and influenza vaccines. Personalized cancer vaccines using mRNA show promising Phase 2 results."},
    {"area": "science", "query": "dark matter research 2025 update", "answer": "Dark matter research continues with experiments like LZ and XENONnT setting new sensitivity records. Alternative theories including modified gravity and axion models are being tested with new observational data."},
    {"area": "health", "query": "Alzheimer's treatment breakthrough 2025", "answer": "New Alzheimer's treatments including lecanemab and donanemab show modest but meaningful slowing of cognitive decline. Tau-targeting therapies and combination approaches are in advanced clinical trials."},
    {"area": "health", "query": "weight loss drugs ozempic wegovy 2025", "answer": "GLP-1 receptor agonists like Ozempic and Wegovy have transformed obesity treatment. New oral formulations and next-generation drugs with improved side effect profiles are in development."},
    {"area": "health", "query": "mental health digital therapeutics 2025", "answer": "Digital therapeutics for mental health have gained FDA approval, with app-based CBT, AI-powered therapy assistants, and VR exposure therapy showing efficacy for anxiety and depression."},
    {"area": "business", "query": "cryptocurrency regulation 2025", "answer": "Cryptocurrency regulation has matured with spot Bitcoin ETFs approved in multiple markets. MiCA in Europe provides a comprehensive framework, while the US debates stablecoin legislation and SEC jurisdiction."},
    {"area": "business", "query": "remote work trends 2025", "answer": "Remote work has stabilized as hybrid models become standard. Companies have invested in digital collaboration tools, asynchronous workflows, and AI assistants. Return-to-office mandates vary by industry."},
    {"area": "business", "query": "semiconductor industry 2025 chips", "answer": "The semiconductor industry has rebounded with AI chip demand driving growth. TSMC and Intel are building new fabs under the CHIPS Act. Advanced packaging and 2nm processes are entering production."},
    {"area": "general", "query": "Renaissance art history summary", "answer": "The Renaissance period (14th-17th centuries) marked a cultural rebirth in Europe. Key artists include Leonardo da Vinci, Michelangelo, and Raphael. The movement emphasized humanism, perspective, and classical influences."},
    {"area": "general", "query": "ancient Egyptian pyramids construction", "answer": "The Egyptian pyramids, built 2600-2500 BCE, showcase remarkable engineering. Recent theories suggest internal ramps and water lubrication for transport. The Great Pyramid remained the tallest structure for 3800 years."},
    {"area": "general", "query": "world war 2 causes summary", "answer": "World War II (1939-1945) was caused by Treaty of Versailles grievances, rise of fascism, appeasement policies, and failed diplomacy. It involved the Axis vs Allied powers and resulted in 70-85 million casualties."},
    {"area": "general", "query": "evolution theory Darwin summary", "answer": "Charles Darwin's theory of evolution by natural selection, published in 1859, explains how species adapt over generations. Modern synthesis incorporates genetics. It remains the foundation of modern biology."},
    {"area": "general", "query": "machine learning types explained", "answer": "Machine learning has three main types: supervised learning (labeled data), unsupervised learning (finding patterns), and reinforcement learning (reward-based). Deep learning uses neural networks for complex tasks."},
    {"area": "general", "query": "blockchain technology explained simple", "answer": "Blockchain is a decentralized digital ledger where transactions are recorded in linked blocks. It enables cryptocurrency, smart contracts, and supply chain tracking. Key features include immutability and transparency."},
    {"area": "general", "query": "climate change basics explained", "answer": "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily driven by greenhouse gas emissions from human activities. Key impacts include rising sea levels, extreme weather, and ecosystem disruption."},
]

SHORT_QUERIES = [
    "Tell me about {topic}",
    "What's the latest on {topic}?",
    "Can you research {topic} and summarize it?",
    "I need a summary of {topic}",
    "Browse the web and tell me about {topic}",
    "Search for information on {topic}",
    "What are the recent developments in {topic}?",
    "Give me an overview of {topic}",
    "I want to learn about {topic}, can you help?",
    "Research {topic} for me please",
]


def _format_tool_call(query: str) -> str:
    return json.dumps({"name": "web_search", "arguments": {"query": query}})


def _format_tool_response(content: str) -> str:
    return json.dumps({"result": content})


def _wrap_tool_call(query: str) -> str:
    return f"<tool_calls>\n[{_format_tool_call(query)}]\n</tool_calls>"


def _wrap_tool_response(content: str) -> str:
    return f"<tool_results>\n{_format_tool_response(content)}\n</tool_results>"


def _format_sample(query_topic: str, search_query: str, web_content: str, summary: str) -> str:
    segments = [
        f"<|system|>\n{SYSTEM_PROMPT}</s>",
        f"<|user|>\n{query_topic}</s>",
        f"<|assistant|>\n{_wrap_tool_call(search_query)}</s>",
        f"<|tool|>\n{_wrap_tool_response(web_content)}</s>",
        f"<|assistant|>\n{summary}</s>",
    ]
    return "\n".join(segments)


def _generate_web_content(topic: Dict, summary: str) -> str:
    area = topic["area"]
    sources = random.sample(
        [
            f"According to a {random.choice(['Nature', 'Science', 'Cell'])} publication",
            f"A report from {random.choice(['Reuters', 'Bloomberg', 'AP News'])} states",
            f"Research published in {random.choice(['arXiv', 'PNAS', 'Nature Communications'])} indicates",
            f"Analysis by {random.choice(['McKinsey', 'Gartner', 'Forrester'])} shows",
            f"Experts at {random.choice(['MIT', 'Stanford', 'Oxford', 'Cambridge'])} have found",
        ],
        random.randint(2, 3),
    )

    detail_sentences = []
    sentences = summary.rstrip(".").split(".")
    for s in sentences:
        s = s.strip()
        if s:
            detail_sentences.append(
                f"{random.choice(sources)}, {s.lower()}. This represents a significant development in {area}."
            )
    return " ".join(detail_sentences[:5])


def generate_dataset(num_samples: int = 500, seed: int = 42) -> List[Dict[str, str]]:
    np.random.seed(seed)
    random.seed(seed)
    samples = []

    for _ in range(num_samples):
        topic = random.choice(TOPICS)
        year = random.choice(["2024", "2025", "2026"])
        base_answer = topic["answer"].format(year=year)
        search_query = topic["query"].format(year=year)

        topic_text = topic["area"].replace("_", " ")
        query_template = random.choice(SHORT_QUERIES)
        user_query = query_template.format(topic=topic_text)

        web_content = _generate_web_content(topic, base_answer)

        summary_variations = [
            f"Based on my research, here is a summary of {topic_text}:\n\n{base_answer}\n\nKey takeaways: {_generate_key_takeaways(base_answer, topic_text)}",
            f"After searching the web, here's what I found about {topic_text}:\n\n{base_answer}\n\n{_generate_key_takeaways(base_answer, topic_text)}",
            f"Here is a comprehensive summary of {topic_text} based on current web information:\n\n## Overview\n{base_answer}\n\n## Key Points\n{_generate_bullet_takeaways(base_answer)}",
            f"I researched {topic_text} for you. Here is the summary:\n\n{base_answer}\n\n## Highlights\n{_generate_bullet_takeaways(base_answer)}",
        ]

        summary = random.choice(summary_variations)
        formatted = _format_sample(user_query, search_query, web_content, summary)
        samples.append({"text": formatted, "area": topic["area"]})

    return samples


def _generate_key_takeaways(answer: str, topic: str) -> str:
    sentences = [s.strip() for s in answer.split(".") if s.strip()]
    takeaways = []
    for s in sentences[:3]:
        takeaways.append(f"The {topic} sector has seen {s.lower()}, which is noteworthy.")
    return " ".join(takeaways)


def _generate_bullet_takeaways(answer: str) -> str:
    sentences = [s.strip() for s in answer.split(".") if s.strip()]
    bullets = []
    for s in sentences[:4]:
        s_clean = s.lower()
        bullets.append(f"- {s_clean}")
    return "\n".join(bullets)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--output", type=str, default="data/train_dataset.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = generate_dataset(args.num_samples, args.seed)
    with open(args.output, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"Generated {len(samples)} samples -> {args.output}")
    print("Areas distribution:")
    areas = {}
    for s in samples:
        a = s["area"]
        areas[a] = areas.get(a, 0) + 1
    for a, c in sorted(areas.items()):
        print(f"  {a}: {c}")
