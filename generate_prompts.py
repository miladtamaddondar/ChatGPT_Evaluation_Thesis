import os
import json
import base64
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROMPT_TEMPLATES = {
    "SCoT": (
        "Solve the linear algebra problem step-by-step, ensuring alignment with code logic.\n"
        "Problem: {content}\n"
        "First, outline the mathematical steps in natural language. "
        "Then, generate executable Python code to verify the solution. "
        "Make sure the answer you provide is in LaTex format and sutiable for a .tex document."
    ),
    "Chain-of-Table": (
        "Given an augmented matrix representing a system of linear equations, "
        "perform step-by-step row operations and show each transformation as a chain of tables. "
        "Each table should represent the matrix at a specific step, with the operation used labeled clearly between them. "
        "When providing the answer, use plain text for explanations and LaTeX format for all mathematical notations.",
        "Present the step-by-step explanation such that each step—including both the description and the calculation—is written in a single line without any line breaks between the steps.",
        "For example it should be: 1. Description and calculation. 2. Description and calculation.",
        "Ensure the final step or answer is also numbered and formatted in a single line.",
        "Maintain this format consistently so that each step occupies exactly one line.\n"
        "Problem: {content}"
    ),
    "LogiCoT": (
        "Solve the equation step-by-step.\n"
        "Problem: {content}\n"
        "After each step, verify its correctness using principles of linear algebra (e.g., properties of determinants, rank, or eigenvalues).",
        "If a step contains errors, revise it and proceed.",
        "When providing the answer, use plain text for explanations and LaTeX format for all mathematical notations.",
        "Present the step-by-step explanation such that each step—including both the description and the calculation—is written in a single line without any line breaks between the steps.",
        "For example it should be: 1. Description and calculation. 2. Description and calculation.",
        "Ensure the final step or answer is also numbered and formatted in a single line.",
        "Maintain this format consistently so that each step occupies exactly one line."
    ),
    "Persona": (
        "Imagine you are a linear algebra university professor. "
        "A student provides you with a question, solve this question step-by-step with principles "
        "of linear algebra validating your reasoning for each step and provide a final answer. "
        "When providing the answer, use plain text for explanations and LaTeX format for all mathematical notationnotations.",
        "Present the step-by-step explanation such that each step—including both the description and the calculation—is written in a single line without any line breaks between the steps.",
        "For example it should be: 1. Description and calculation. 2. Description and calculation.",
        "Ensure the final step or answer is also numbered and formatted in a single line.",
        "Maintain this format consistently so that each step occupies exactly one line.\n"
        "The question is: {content}"
    )
}

def generate_prompt(prompt_type, content, tailoring=""):
    template = PROMPT_TEMPLATES.get(prompt_type)
    if not template:
        raise ValueError(f"Invalid prompt type: {prompt_type}")
    
    if isinstance(template, tuple):
        template = "\n".join(template)
    base_prompt = template.format(content=content)

    
    if tailoring and str(tailoring).strip():
        return f"{base_prompt}\n\nAdditional Instructions: {tailoring}"
    return base_prompt

def send_to_llm(prompt, image_path=None, model="gpt-4.1", test_mode=False):
    if test_mode:
        print("[TEST MODE] Skipping real API call.")
        return f"[MOCK RESPONSE] This is a test response for the prompt:\n\n{prompt}"

    messages = [
        {"role": "system", "content": "You are a helpful linear algebra assistant."}
    ]

    if image_path:
        with open(image_path, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode("utf-8")

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_data}"
                    }
                }
            ]
        })
    else:
        messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
    )

    return response.choices[0].message.content

    '''
    An example of a problem with an image: (make a file called images and store the images there 
    then add them like this in the problems.json)
    {
        "type": "Chain-of-Table",
        "content": "Solve the following matrix problem:",
        "image": "images/matrix_problem_3.png"
    }
    '''


def run_all():
    test_mode = False #set to False later when using actual API call

    with open("problems.json", "r") as f:
        problems = json.load(f)

    if os.path.exists("outputs") and not os.path.isdir("outputs"):
        raise Exception("'outputs' exists but is not a directory. Please rename or delete it.")
    os.makedirs("outputs", exist_ok=True)

    # Track counters by prompt type and input tag (text/image)
    type_tag_counters = {}

    for i, item in enumerate(problems):
        tailoring = item.get("prompt_tailoring", "")
        image_path = item.get("image")
        prompt_type = item["type"]

        # Determine input tag: "image" or "text"
        input_tag = "image" if image_path else "text"
        tag_key = (prompt_type, input_tag)

        # Initialize the counter if this combination is new
        if tag_key not in type_tag_counters:
            type_tag_counters[tag_key] = 1

        count = type_tag_counters[tag_key]  # This stays the same across 10 repetitions

        for repeat in range(1, 2):  # Generate 1 version instead of the original 10
            prompt = generate_prompt(prompt_type, item["content"], tailoring)
            response = send_to_llm(prompt, image_path=image_path, test_mode=test_mode)

            # e.g., Persona_text_01_01.txt
            filename = f"{prompt_type}_{input_tag}_{count:02d}_{repeat:02d}.txt"

            with open(os.path.join("outputs", filename), "w", encoding="utf-8") as f:
                f.write(response)

            print(f"[✓] {filename} saved.")

        # Increment counter only after completing all 10 variations
        type_tag_counters[tag_key] += 1




if __name__ == "__main__":
    run_all()
