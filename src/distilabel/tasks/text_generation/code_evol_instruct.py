import sys
from dataclasses import dataclass

if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources

import random
import string

from typing import Any, Dict, List, Literal, Optional, get_args

from distilabel.logger import get_logger
from distilabel.tasks.base import get_template
from distilabel.tasks.prompt import Prompt
from distilabel.tasks.text_generation.base import TextGenerationTask
from distilabel.tasks.text_generation.mixins import InstructTaskMixin

logger = get_logger()


_CODE_EVOL_INSTRUCT_TEMPLATE = get_template("code-evol-instruct.jinja2")

CodeEvolutionMethod = Literal[
    "constraints",
    "breadth",
    "deepen",
    "concretizing",
    "reasoning",
    "clean_code_principles",
    "SOLID_principles",
    "TDD",
    "refactoring",
    "agile_methodologies",
    "pair_programming",
    "CI_CD",
    "software_craftsmanship",
    "design_patterns",
    "code_smells_and_refactoring",
    "dependency_inversion",
    "encapsulation_and_modularity",
    "immutability",
    "functional_programming",
    "clean_architecture",
    "code_simplicity",
    "DDD",
    "unit_testing",
    "clean_UI_design",
    "scalability",
    "performance_optimization",
    "understand_the_problem",
    "devise_a_plan",
    "carry_out_the_plan",
    "review_and_extend",
    "analogy_in_problem_solving",
    "generalize_the_solution",
    "working_backwards",
    "inductive_reasoning",
    "deductive_reasoning",
    "divide_and_conquer",
    "iterative_improvement",
    "look_for_a_pattern",
    "specialization",
    "experimentation",
    "visualization",
    "reformulation",
    "simplification",
    "analogy_with_known_solutions",
    "logical_analysis",
    "sequential_reasoning",
    "inversion",
    "combination_of_methods",
    "symmetry",
    "recursion",
    "constraint_identification_and_relaxation",
    "proof_by_contradiction",
    "problem_decomposition",
    "abstraction",
    "optimization",
    "heuristic_development",
    "modeling",
]

def _get_stopwords() -> List[str]:
    """Gets the list of english stopwords from nltk package.

    Returns:
        List[str]: stopwords list.
    """
    try:
        with (
            importlib_resources.files("distilabel") / "tasks/_internal/stopwords_en.txt"
        ).open("r") as f:
            return f.read().split("\n")
    except FileNotFoundError:
        return []

@dataclass
class CodeEvolInstructTask(InstructTaskMixin, TextGenerationTask):
    """A `TextGenerationTask` tailored for evolving coding instructions to adhere to high-quality software development principles, as advocated by Martin Fowler.

    This task aims to refine and elevate coding instructions to promote practices such as clean code, SOLID principles, design patterns, and refactoring. It's designed to help developers write code that's not only functional but also maintainable, scalable, and understandable.

    Args:
        system_prompt (str, optional): The system prompt to be used. Not defined for this task, as the focus is on evolving code instructions.

    """

    system_prompt: str = ""

    __jinja2_template__: str = _CODE_EVOL_INSTRUCT_TEMPLATE




    def generate_prompt(
        self, input: str, evolution_method: Optional[CodeEvolutionMethod] = None, **_: Any
    ) -> Prompt:
        """Generates a prompt that focuses on evolving a coding instruction to incorporate key software development principles.

        Args:
            input (str): The initial code instruction to evolve.
            evolution_method (Optional[CodeEvolutionMethod], optional): The method or principle to apply for evolution. If not specified, a suitable method will be selected based on the input's context.

        Returns:
            Prompt: The evolved code instruction prompt.
        """
        if evolution_method is None:
            evolution_method = self._select_evolution_method(evolution_method,CodeEvolutionMethod)

        render_kwargs = {
            "evol_method": evolution_method,
            "instruction": input,
        }
      
      
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=self.template.render(**render_kwargs),
        )
    

    def _select_evolution_method(
        self, chosen_method: CodeEvolutionMethod, available_methods: CodeEvolutionMethod
    ) -> None:
        available_methods = get_args(available_methods)
        if not chosen_method:
            chosen_method = random.choice(available_methods)
        if chosen_method not in available_methods:
            raise ValueError(
                f"Evolution method {chosen_method} is not available. Available ones are: {available_methods}"
            )
        return chosen_method

    @property
    def output_args_names(self) -> List[str]:
        return ["instructions"]

    def parse_output(self, output: str) -> Dict[str, List[str]]:
        """Parses the evolved instruction output from the model, ensuring it aligns with the desired code evolution principles.

        Args:
            output (str): The model's output, containing the evolved code instruction.

        Returns:
            Dict[str, List[str]]: The parsed and potentially refined evolved instruction.
        """
        # Implement parsing and validation logic here to ensure the output aligns with high-quality coding standards.
        evolved_instruction = self._refine_output(output)
        return {"evolved_instructions": [evolved_instruction]}

    def _refine_output(
        self, output: str, response_words: Optional[List[str]] = None
    ) -> Optional[str]:
        """Performs the elimination step of the Evol-Instruct task, steps 2-4 in the paper:

        1. [NOT IMPLEMENTED] The evolved instruction does not provide any information gain compared
        to the original one. Use ChatGPT to make this determination, this is outlined in Appendix G of the original paper.
        2. The evolved instruction makes it difficult for the LLM to generate a response. We found that
        when the generated response contains “sorry” and is relatively short in length (i.e., less than
        80 words), it often indicates that the LLM struggles to respond to the evolved instruction.
        So we can use this rule to make a judgment.
        3. The response generated by the LLM only contains punctuation and stop words.
        4. The evolved instruction obviously copies some words from the evolving prompt, such as
        “given prompt”, “rewritten prompt”, “#Rewritten Prompt#”, etc.
        """
        output = output.strip()
        if output == "":
            return

        # 2) The evolved instruction makes it difficult for the LLM to generate a response.
        if "sorry" in output.lower() and len(output.split(" ")) < 80:
            logger.info(
                f"Evolution step removed the output, it's hard for the LLM to generate a response: {output}"
            )
            return

        # 3) The output only contains punctuation and stop words
        stopwords = _get_stopwords()
        clean_output = [word for word in output.split(" ") if word not in stopwords]
        if set(clean_output).difference(set(string.punctuation)) == 0:
            logger.info(
                f"Evolution step removed the output, it only contains punctuation and stop words: {output}"
            )
            return

        # 4) Remove copied words from the prompt
        prompt_words = {
            "#Given Prompt#",
            "#Created Prompt#",
            "given prompt",
            "created prompt",
            "#The Given Prompt#",
            "#Rewritten Prompt#",
            "rewritten prompt",
        }
        if response_words:
            prompt_words = prompt_words.union(response_words)
        if any(word in output for word in prompt_words):
            logger.info(
                f"Evolution step removed the output due to word repetition from the prompt: {output}"
            )
            return

        return output