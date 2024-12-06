# **AulSign: Advanced Use of LLMs for Sign Language Translation**

## **Table of Contents**

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Translation Process Explainability](#translation-process-explainability)
6. [Online Demo](#online-demo)
7. [Dataset](#dataset)
8. [Contributions](#contributions)
9. [License](#license)

---

## **Overview**

* **Handle low-resource languages** in LLM training data.
* Operate in **data-scarce environments**.
* Translate between text and **Formal SignWriting (FSW)**, or vice versa.
* **Provide explainability**, detailing the steps taken by the model during translation.

---

## **Key Features**

* **Bidirectional translation:** Supports translation in both directions between text and FSW.
* **Data-scarcity handling:** Designed for sign languages with limited datasets.
* **Explainability:** Offers transparent insights into the model's internal processes during translation.
* **Extensibility:** Easily adaptable to new datasets and languages.

---

## **Installation**

### **Prerequisites**

* **No training required:** The model is ready to use.
* Recommended GPU: Necessary for running LLMs efficiently.
* Alternatively, you can use third-party inference services like **Openai Api** or similar.
* Python 3.9 or higher.
* Required libraries (see `requirements.txt`).

### **Instructions**

1. Clone the repository:

   ```bash
   git clone https://github.com/gabrix00/aulsign.git
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Configure the model:

   Run the script to generate all required files from scratch:

   ```bash
   python main_scripts/pipeline.py
   ```
4. Replicate the experiments:

   ```bash
   python aulsign.py --mode text2sign setup --full
   ```

   also setup `filtered` and `filtered_01` are available
5. Evaluate results

```bash
result/get_results_pipeline.py path --"result/text2sign_2024_12_06_12_17/result_2024_12_06_12_17.csv"
```

---

## **Usage**

### **Quick Example**

Run a simple translation command:

```bash
python aulsign.py --mode text2sign --sentence_input "This is a new ASL translator" 
```

### **Available Modes**

* **`text2sign`**: Translate from text to Formal SignWriting.

```bash
python aulsign.py --mode text2sign --input "Hello!"
```

* **`sign2text`**: Translate from Formal SignWriting to text.

```bash
python aulsign.py --mode sign2text --input "<FSW_code>"
```

---

## **Translation Process Explainability**

**AulSign** not only delivers accurate translations but also makes the process **explainable**. Here's how it works:

1. **Input analysis:**

   * In `text2sign` mode, the model segments the input text into semantic units.
   * In `sign2text` mode, the model decodes FSW symbols into intermediate representations.
2. **Representation matching:** A similarity-based metric aligns meanings between text and FSW using embeddings.
3. **Translation generation:** Each step is documented, enabling users to understand how the translation was produced.

**Example of explainable output:**

```json
{
  "input": "This is a new ASL translator",
  "steps": [
    {
      "step": "LLM answer (tokenization)",
      "details": ["this|that", "be|is", "fresh|new", "asl|american sign language", "translator|interpreter"]
    },
    {
      "step": "Match through Vocabulary",
      "details": [
        {
          "word": "this|that", 
          "match": "this", 
          "sim": 0.9287521840015376, 
          "fsw": "M510x527S10004495x473S22a04490x512"
        },
        {
          "word": "be|is", 
          "match": "is|are", 
          "sim": 0.8637841304248837, 
          "fsw": "AS33b00S19210S20500S26504M519x547S33b00482x482S20500466x512S26504464x532S19210498x511"
        },
        {
          "word": "fresh|new", 
          "match": "new|fresh", 
          "sim": 0.9754369661042025, 
          "fsw": "M530x522S15a36502x510S1813e501x503S2890f470x478"
        },
        {
          "word": "asl|american sign language", 
          "match": "asl|american sign language", 
          "sim": 1, 
          "fsw": "M512x535S1f720492x466S20320497x485S1dc20488x505"
        },
        {
          "word": "translator|interpreter", 
          "match": "translator|interpreter", 
          "sim": 1, 
          "fsw": "M528x595S10009483x405S10021473x422S2e024488x453S10001491x488S10029493x504S15a48477x548S15a40515x548S22a14476x580S22a04515x580"
        }
      ]
    },
    {
      "step": "Generate output",
      "details": "FSW translation generated"
    }
  ],
  "output": "M510x527S10004495x473S22a04490x512 AS33b00S19210S20500S26504M519x547S33b00482x482S20500466x512S26504464x532S19210498x511 M530x522S15a36502x510S1813e501x503S2890f470x478 M512x535S1f720492x466S20320497x485S1dc20488x505 M528x595S10009483x405S10021473x422S2e024488x453S10001491x488S10029493x504S15a48477x548S15a40515x548S22a14476x580S22a04515x580"
}

```

---

## **Online Demo**

You can test the model in an interactive environment using the **GitHub demo**. Access the demo via the following link:

[**AulSign Demo on GitHub**](https://github.com/your-username/aulsign-demo)

This demo allows you to upload text or FSW symbols as input and receive instant translations.

---

## **Dataset**

### **Data Sources**

* Predefined datasets `SignBank3.csv`: Available in the `/data` directory

### **Preparation**

* Run the script to generate all required files from scratch:

  ```bash
  python main_scripts/pipeline.py
  ```

---

## **Contributions**

Contribute to the project by following these steps:

1. Create an **issue** to suggest improvements or report bugs.
2. Fork the repository.
3. Create a new branch:
   ```bash
   git checkout -b feature/feature-name
   ```
4. Open a **pull request**, describing your proposed changes.

---

## **License**

This project is distributed under the **MIT License**. See the [LICENSE](https://github.com/your-username/aulsign/LICENSE) file for more details.

---

Let me know if you’d like further refinements!
