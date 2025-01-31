
I have Pushed The model on the HUggingFace https://huggingface.co/Sara5115/dialect_conversion_model

**Overview**

This project focuses on converting text between different dialects using a fine-tuned transformer model. The model is based on facebook/bart-large and is trained on a dataset containing pairs of dialect-specific sentences (e.g., UK English to US English). The project utilizes the Hugging Face transformers library for training and evaluation.

**Features**

Converts text from one dialect to another (e.g., UK English → US English).

Utilizes BART for sequence-to-sequence text generation.

Implements training, evaluation, and inference functions.

Computes BLEU scores to evaluate model accuracy.

Deployable via the Hugging Face Model Hub.

**Installation**

Clone the repository or set up in a Colab environment.

**Install dependencies:**
pip install torch transformers datasets pandas numpy nltk scikit-learn

Authenticate with Hugging Face Hub (if needed):

from huggingface_hub import notebook_login notebook_login()

**Dataset**

The dataset (CozmoX.csv) consists of two columns:

input_text: Text in one dialect (e.g., UK English)

target_text: Equivalent text in another dialect (e.g., US English)

**Training**

Run the training process using:

train_dialect_model()

This trains the model using Seq2SeqTrainer and saves the trained model to ./dialect_conversion_model.

**Inference**

Use the convert_dialect function to convert text:

text = "I love the colours of autumn." converted_text = convert_dialect(text) print(converted_text)

**Evaluation**

The model's performance is measured using BLEU scores:

average_bleu = evaluate_model(val_df) print(f"Average BLEU Score: {average_bleu}")

Example Usage

text = "The theatre programme was fantastic!" converted = convert_dialect(text) print(f"Converted: {converted}")

**Future Improvements**

Expand the dataset with more dialect pairs.

Optimize model hyperparameters.

Fine-tune with additional pre-trained models.

Deploy as an API for real-time text conversion.
