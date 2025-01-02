import re


def preprocess_for_bigram_model(input_file_path):
    with open(input_file_path, "r", encoding="utf-8") as file:
        text = file.read()

    text = re.sub(r"[^\w\s]", "", text).lower()
    words = text.split()
    bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
    bigram_text = "\n".join([f"{bigram[0]} {bigram[1]}" for bigram in bigrams])
    output_file_path = input_file_path.replace(
        ".txt", "_preprocessed_for_bi_gram_lm.txt"
    )

    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write(bigram_text)


def preprocess_for_q2_lm(input_file_path):
    with open(input_file_path, "r", encoding="utf-8") as file:
        text = file.read()

    text = re.sub(r"[^\w\s]", "", text).lower()
    output_file_path = input_file_path.replace(".txt", "_preprocessed_for_q2_lm.txt")

    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write(text)


if __name__ == "__main__":
    preprocess_for_bigram_model("shakespeare_for_perplexity.txt")
    preprocess_for_bigram_model("wikipedia_for_perplexity.txt")
    preprocess_for_q2_lm("shakespeare_for_perplexity.txt")
    preprocess_for_q2_lm("wikipedia_for_perplexity.txt")
