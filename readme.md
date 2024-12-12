# AulSign: Advanced Use of LLMs for Sign Language Translation 🌍

**AulSign** is a novel approach that leverages Large Language Models (LLMs) combined with few-shot learning and sample selection techniques to bridge the gap between natural languages and sign languages.

## **Overview** 🔍

* **Handle low-resource languages** in LLM training data.
* Operate in **data-scarce environments**.
* Translate between text and **Formal SignWriting (FSW)**, or vice versa.
* **Provide explainability**, detailing the steps taken by the model during translation.

---

## **Key Features** ✨

* **Bidirectional translation:** Supports translation in both directions between text and FSW.
* **Data-scarcity handling:** Designed for sign languages with limited datasets.
* **Explainability:** Offers transparent insights into the model's internal processes during translation.
* **Extensibility:** Easily adaptable to new datasets and languages.

---

## **Installation** 🛠️

### **Prerequisites** ⚙️

* **No training required:** The model is ready to use.
* Recommended GPU: Necessary for running LLMs efficiently.
* Alternatively, you can use third-party inference services like **OpenAI API** or similar.
* Python 3.9 or higher.
* Required libraries (see `requirements.txt`).
* **Environment Variables:**
  To use the `aulsign.py` script with OpenAI GPT-3.5 Turbo, make sure to export your OpenAI credentials in your terminal session:
  ```bash
  export OPENAI_API_KEY="YOUR_API_KEY"
  export OPENAI_ORGANIZATION="YOUR_ORG_ID"
  export OPENAI_PROJECT="YOUR_PROJECT_ID"
  ```

### **Instructions** 📜

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
   python data/main_scripts/pipeline.py
   ```
4. Replicating the Experiments
   To replicate the experiments with the full dataset, use the following command:

   ```bash
   python aulsign.py --mode text2sign --setup full
   ```

   Additionally, you can replicate the experiments with reduced datasets for different data scarcity scenarios:

   * **Filtered Dataset (2301 training samples):**

   ```bash
   python aulsign.py --mode text2sign --setup filtered
   ```

   * **Highly Filtered Dataset (115 training samples):**

   ```bash
   python aulsign.py --mode text2sign --setup filtered_01
   ```
5. Evaluate AulSign results:
   From **text2sign** use the following command:
   ```bash
   python result/get_results_pipeline_text2sign.py --result result/text2sign_{current_datetime}/result_{current_datetime}.csv
   ```
   From **sign2text** use the following command:
   ```bash
   python result/get_results_pipeline_sign2text.py --result result/sign2text_{current_datetime}/result_{current_datetime}.csv
   ```
6. Evaluate competitor results:
   ```bash
   python jiang_results_text2sign/get_results_pipeline.py --result jiang_results_text2sign/{folder_name}/predictions.csv
   ```



### **Quick Example** ⚡

Run a simple translation command:

```bash
python aulsign.py --mode text2sign --input "This is a new ASL translator" 
```

### **Available Modes** 🔄

* **`text2sign`**: Translate from text to Formal SignWriting.

```bash
python aulsign.py --mode text2sign --input "<input_sentence>" 
```

* **`sign2text`**: Translate from Formal SignWriting to text.

```bash
python aulsign.py --mode sign2text --input "<FSW_code>"
```

---

## **Translation Process Explainability** 🧠

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

## **Dataset** 📊

### **Data Sources** 📁

* Predefined datasets `SignBank3.csv`: Available in the `/data` directory

---

## **Contributions** 🤝

Contribute to the project by following these steps:

1. Create an **issue** to suggest improvements or report bugs.
2. Fork the repository.
3. Create a new branch:
   ```bash
   git checkout -b feature/feature-name
   ```
4. Open a **pull request**, describing your proposed changes.

