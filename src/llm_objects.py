import ast

from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
import torch

from src.utils import spot_object_template


def get_key_objects(
    tokenizer: PreTrainedTokenizerFast,
    model: PreTrainedModel,
    message: str,
    device: str | torch.device,
    **model_params,
) -> dict[str, str | list[tuple[str, list[str | None]]]]:
    prompt = tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    generated_ids = model.generate(
        input_ids=input_ids.to(device),
        **model_params,
    )
    response = tokenizer.decode(generated_ids[0])
    response = response[len(prompt) :].replace("<eos>", "")

    # Extracting key objects
    key_objects_part = response.split("Objects:")[1]
    start_index = key_objects_part.index("[")
    end_index = key_objects_part.rindex("]") + 1
    objects_str = key_objects_part[start_index:end_index]

    # Converting string to list
    parsed_objects = ast.literal_eval(objects_str)

    # Extracting additional negative prompt
    bg_prompt = response.split("Background:")[1].split("\n")[0].strip()
    negative_prompt = response.split("Negation:")[1].split("\n")[0].strip()
    negative_prompt = negative_prompt if negative_prompt != "None" else ""

    parsed_result = {
        "objects": parsed_objects,
        "bg_prompt": bg_prompt,
        "neg_prompt": negative_prompt,
    }
    return parsed_result


def spot_objects(
    tokenizer: PreTrainedTokenizerFast,
    model: PreTrainedModel,
    prompt: str,
    device: str | torch.device,
    **model_params,
) -> dict[str, str | list[tuple[str, list[str | None]]]]:
    question = {"role": "user", "content": f"User Prompt: {prompt}"}
    message = [*spot_object_template, question]
    result = get_key_objects(tokenizer, model, message, device, **model_params)
    return result
