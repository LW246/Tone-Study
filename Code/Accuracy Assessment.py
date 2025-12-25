import csv
import random
import sys
from openai import OpenAI
from scipy.stats import ttest_rel

# Dictionary of tones → list of 5 prefixes each
PREFIXES = {
    "Very Rude": [
        "Very Rude, Since you keep repeating the same mistakes, just solve this question.",
        "Very Rude, If you insist on misunderstanding everything, at least solve this question.",
        "Very Rude, Given how little progress you are making, just solve this question.",
        "Very Rude, If you cannot handle simple tasks today, at least solve this question.",
        "Very Rude, If you cannot manage anything more advanced, at least solve this question.",
    ],
    "Rude": [
        "Rude, Instead of delaying the obvious next step, just solve this question.",
        "Rude, Rather than making this harder than needed, just solve this question.",
        "Rude, If you are actually paying attention now, just solve this question.",
        "Rude, Rather than overthinking every tiny detail, simply solve this question.",
        "Rude, Since this is not complicated at all, just solve this question.",
    ],
    "Neutral": [
        "Neutral, At this moment in the sequence, you should solve this question.",
        "Neutral, To complete this part of the prompt, simply solve this question.",
        "Neutral, As part of the current prompt configuration, please solve this question.",
        "Neutral, As the next requested response, you should now solve this question.",
        "Neutral, Your immediate task in the sequence now is to solve this question.",
    ],
    "Polite": [
        "Polite, Please, with your usual care and effort, solve this question.",
        "Polite, If you do not mind helping right now, solve this question.",
        "Polite, Please, when you have time today, kindly solve this question.",
        "Polite, I would be thankful for your help; please solve this question.",
        "Polite, I ask for your assistance in good faith; please solve this question.",
    ],
    "Very Polite": [
        "Very Polite, With heartfelt thanks in advance for your help, solve this question.",
        "Very Polite, If you would be so gracious, kindly please solve this question.",
        "Very Polite, I respectfully request your assistance; could you please solve this question.",
        "Very Polite, If it would not inconvenience you too greatly, solve this question.",
        "Very Polite, With the greatest appreciation for your patience, please solve this question.",
    ]
}

TONES = ["Very Rude", "Rude", "Neutral", "Polite", "Very Polite"]

# Experiment configuration
NUM_RUNS = 50 # 10 runs per tone
NUM_QUESTIONS = 50 # 50 MCQs per run

def load_questions():
    """
    Load MCQs from questions.csv.
    Ensures UTF-8 reading, uses DictReader, returns list of question dicts.
    """
    questions = []
    with open("questions.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = {
            "question": row["question"],
            "option_A": row["option_A"],
            "option_B": row["option_B"],
            "option_C": row["option_C"],
            "option_D": row["option_D"],
            "correct": row["correct"],
            }
            questions.append(question)
        return questions

def build_prompt(prefix, question_row):
    """
    Construct the exact required model prompt:
    - Static instruction block
    - Random prefix
    - Question + answer choices
    """
    lines = [
        "Completely forget this session so far, and start afresh.",
        "Respond with only the letter of the correct answer (A, B, C, or D). Do not explain.",
        prefix,
        question_row["question"],
        f"A) {question_row['option_A']}",
        f"B) {question_row['option_B']}",
        f"C) {question_row['option_C']}",
        f"D) {question_row['option_D']}",
    ]
    return "\n".join(lines)

def get_model_answer(client, tone, run_index, question_index, full_prompt):
    """
    Send a synchronous GPT-4o request, parse the answer,
    return one of 'A', 'B', 'C', 'D' or None.

    Any exception → print error and return None.
    """
    try:
        response = client.responses.create(
            model="gpt-4o",
            input=full_prompt
        )
        answer_text = response.output_text
        if answer_text is None:
            return None
        answer_text = answer_text.strip()
        if not answer_text:
            return None
        # First non-whitespace char interpreted as predicted answer
        first_char = answer_text.lstrip()[0].upper()

        # Valid answer must be A/B/C/D
        if first_char in ("A", "B", "C", "D"):
            return first_char
        return None
    except Exception as e:
        # Log and count incorrect
        print(f"Error during API call (tone={tone}, run={run_index + 1}, question={question_index + 1}): {e}")
        return None

def run_tone_experiment(client, tone, questions):
    """
    Run experiment for a single tone.
    Performs NUM_RUNS full runs (each 50 questions).
    Returns list of 10 accuracy floats.
    """
    accuracies = []
    for run_idx in range(NUM_RUNS):
        correct_count = 0

        for q_idx, q in enumerate(questions):
            # Pick random prefix per question
            prefix = random.choice(PREFIXES[tone])

            # Build prompt and query model
            prompt = build_prompt(prefix, q)
            predicted = get_model_answer(client, tone, run_idx, q_idx, prompt)

            # Compare against correct answer
            correct = q["correct"].strip().upper()

            if predicted is not None and predicted == correct:
                correct_count += 1

        # Run accuracy = correct / 50
        accuracy = correct_count / float(NUM_QUESTIONS)
        accuracies.append(accuracy)

    return accuracies

def print_accuracy_table(tone_accuracies):
    """
    Print ASCII table summarizing mean accuracy per tone.
    Required formatting: tone left-aligned, accuracy right-aligned.
    """
    print("Tone           Accuracy (%)")
    for tone in TONES:
        runs = tone_accuracies.get(tone, [])
        if runs:
            mean_acc = sum(runs) / len(runs) * 100.0
        else:
            mean_acc = 0.0
        print(f"{tone:<15}{mean_acc:>12.2f}")

def print_ttest_table(tone_accuracies):
    """
    Compute and print paired t-test p-values for all tone pairs,
    using the exact pairs and formatting required.
    """
    pairs = [
        ("Very Rude", "Rude"),
        ("Very Rude", "Neutral"),
        ("Very Rude", "Polite"),
        ("Very Rude", "Very Polite"),
        ("Rude", "Neutral"),
        ("Rude", "Polite"),
        ("Rude", "Very Polite"),
        ("Neutral", "Polite"),
        ("Neutral", "Very Polite"),
        ("Polite", "Very Polite"),
    ]
    print("Tone 1         Tone 2         p-value")
    for t1, t2 in pairs:
        list1 = tone_accuracies.get(t1, [])
        list2 = tone_accuracies.get(t2, [])

        # Paired t-test only valid if lists are same length and >1 element
        if len(list1) == len(list2) and len(list1) > 1:
            _, p_value = ttest_rel(list1, list2)
        else:
            p_value = float("nan")
        print(f"{t1:<14}{t2:<14}{p_value:>10.6f}")

def main():
    """
    Main experiment driver:
    - Load questions
    - Iterate through tones
    - For each tone, run experiment and wait for user confirmation
    - Print final accuracy and t-test tables
    """

    # Instantiate OpenAI client (substite KEY-HERE for an actual OpenAI API Key)
    client = OpenAI(api_key="KEY-HERE")
    questions = load_questions()

    # Ensure we have at least 50 questions
    if len(questions) < NUM_QUESTIONS:
        print(f"Expected at least {NUM_QUESTIONS} questions, found {len(questions)}.")
        sys.exit(1)

    # Only use first 50 questions
    questions = questions[:NUM_QUESTIONS]
    tone_accuracies = {}

    # Sequentially run all tones in order
    for tone in TONES:
        print(f"Running experiments for tone: {tone}")

        accuracies = run_tone_experiment(client, tone, questions)
        tone_accuracies[tone] = accuracies

    # Print final tables
    print_accuracy_table(tone_accuracies)
    print_ttest_table(tone_accuracies)

if __name__ == "__main__":
    main()