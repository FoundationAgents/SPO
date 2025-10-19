# -*- coding: utf-8 -*-
# @Date    : 8/23/2024 10:00 AM
# @Author  : all
# @Desc    : Evaluation for different datasets
import asyncio
import random
from typing import Any, Dict, Optional

from prompts.evaluate_prompt import EVALUATE_PROMPT
from utils import load
from utils.llm_client import SPO_LLM, RequestType, extract_content
from loguru import logger


class QuickExecute:
    """
    Execute Prompt
    """

    def __init__(self, prompt: str):
        self.prompt = prompt
        self.llm = SPO_LLM.get_instance()

    async def prompt_execute(self) -> tuple[Any]:
        _, _, qa, _ = load.load_meta_data()
        answers = []

        async def fetch_answer(q: str) -> Dict[str, Any]:
            messages = [{"role": "user", "content": f"{self.prompt}\n\n{q}"}]
            try:
                answer = await self.llm.responser(request_type=RequestType.EXECUTE, messages=messages)
                return {"question": q, "answer": answer}
            except Exception as e:
                return {"question": q, "answer": str(e)}

        tasks = [fetch_answer(item["question"]) for item in qa]
        answers = await asyncio.gather(*tasks)

        return answers


class QuickEvaluate:
    """
    Complete the evaluation for different answers here.
    """

    def __init__(self):
        self.llm = SPO_LLM.get_instance()

    async def prompt_evaluate(self, samples: dict, new_samples: dict) -> Optional[bool]:
        """
        Evaluate two samples and return the evaluation result
        Returns:
            True: New sample is better
            False: Old sample is better
            None: Neutral/Unable to judge/Evaluation failed
        """
        _, requirement, qa, _ = load.load_meta_data()

        if random.random() < 0.5:
            samples, new_samples = new_samples, samples
            is_swapped = True
        else:
            is_swapped = False

        messages = [
            {
                "role": "user",
                "content": EVALUATE_PROMPT.format(
                    requirement=requirement,
                    answers=samples.get("answers"),
                    new_answers=new_samples.get("answers"),
                    ground_truth=str(qa),
                ),
            }
        ]

        try:
            response = await self.llm.responser(request_type=RequestType.EVALUATE, messages=messages)
            choose = extract_content(response, "choose")
            
            # Check if choose is a valid value
            if choose is None:
                logger.warning("Unable to extract choose tag from LLM response")
                return None
            
            # Normalize choose value, remove spaces and convert to uppercase
            choose = choose.strip().upper()
            
            # Check if it's A or B
            if choose == "B":
                return True if not is_swapped else False
            elif choose == "A":
                return False if not is_swapped else True
            else:
                # Neither A nor B, return neutral value
                logger.info(f"LLM returned non-A/B choice: '{choose}', returning neutral value")
                return None

        except Exception as e:
            logger.error(f"Error occurred during evaluation: {e}")
            return None
