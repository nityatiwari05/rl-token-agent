"""
Synthetic world generator.

A "world" = a small set of documents about fictional entities
(flights, employees, orders, etc.) plus a question whose answer
is deterministically computable from exactly one relevant document
(1-hop) or a short chain of documents (multi-hop, for later).

Ground truth is computed programmatically -- never by an LLM --
so reward is trustworthy.

Each relevant document embeds a literal "ANSWER_MARKER:<value>" token.
This is a deliberate simplification: it lets us validate the FULL
RL pipeline (retrieval -> context selection -> LLM -> reward) without
fighting real-world NLP noise first. Swap in a real
retriever/LLM later; the env doesn't change.
"""
import random
import string
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Document:
    doc_id: str
    text: str
    is_relevant: bool
    hop: int  # which hop in the reasoning chain this doc supports (1 = first/only hop)


@dataclass
class World:
    world_id: str
    documents: List[Document]
    question: str
    answer: str
    n_hops: int


ENTITY_NAMES = [
    "Flight A", "Flight B", "Flight C", "Order 1042", "Order 7781",
    "Employee Chen", "Employee Patel", "Server cluster West",
    "Invoice 220", "Shipment Delta", "Ticket 99", "Build job 14",
]

ATTRIBUTES = [
    "departure time", "status", "owner", "total cost", "location",
    "priority level", "last updated by", "current stage",
]

DISTRACTOR_TEMPLATES = [
    "{entity} was mentioned in a meeting on {day}.",
    "{entity} has historically been associated with {topic}.",
    "Someone asked about {entity} last {day}, but the thread went unanswered.",
    "{entity} appears in the system logs with no further detail.",
    "A note was left about {entity} regarding an unrelated {topic} matter.",
]

TOPICS = ["budget review", "scheduling", "compliance", "onboarding", "maintenance"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def _rand_value(rng: random.Random, length: int = 6) -> str:
    return "".join(rng.choices(string.ascii_uppercase + string.digits, k=length))


def generate_world(
    world_id: str,
    rng: Optional[random.Random] = None,
    n_distractors: int = 6,
    n_hops: int = 1,
) -> World:
    """
    Generate a single synthetic world.

    n_distractors: number of irrelevant documents mixed in with the
                   relevant one(s). Controls retrieval difficulty.
    n_hops: 1 = single relevant doc directly answers the question.
            >1 = answer requires chaining through n_hops documents
                 (each hop's doc references the next entity).
                 NOT implemented in this first pass -- raises for now,
                 added in generalization-test phase.
    """
    if n_hops != 1:
        raise NotImplementedError("Multi-hop worlds come in phase 2 (generalization test).")

    rng = rng or random.Random()
    entity = rng.choice(ENTITY_NAMES)
    attribute = rng.choice(ATTRIBUTES)
    answer = _rand_value(rng)

    relevant_doc = Document(
        doc_id="DOC0",
        text=f"{entity}: the {attribute} is recorded as ANSWER_MARKER:{answer} .",
        is_relevant=True,
        hop=1,
    )

    distractors: List[Document] = []
    for i in range(n_distractors):
        template = rng.choice(DISTRACTOR_TEMPLATES)
        # Sometimes reference the SAME entity (true distractor / red herring),
        # sometimes a different one (pure noise).
        dist_entity = entity if rng.random() < 0.4 else rng.choice(ENTITY_NAMES)
        text = template.format(entity=dist_entity, day=rng.choice(DAYS), topic=rng.choice(TOPICS))
        distractors.append(Document(doc_id=f"DOC{i+1}", text=text, is_relevant=False, hop=0))

    all_docs = [relevant_doc] + distractors
    rng.shuffle(all_docs)

    question = f"What is the {attribute} for {entity}?"

    return World(
        world_id=world_id,
        documents=all_docs,
        question=question,
        answer=answer,
        n_hops=n_hops,
    )


def generate_dataset(
    n_worlds: int,
    seed: int = 0,
    n_distractors_range=(3, 12),
) -> List[World]:
    """Generate a batch of worlds with varying distractor counts (= varying difficulty)."""
    rng = random.Random(seed)
    worlds = []
    for i in range(n_worlds):
        n_distractors = rng.randint(*n_distractors_range)
        worlds.append(generate_world(world_id=f"world_{i}", rng=rng, n_distractors=n_distractors))
    return worlds