# Customer Feedback Sentiment & Categorization

This tool uses OpenAI's GPT-4o-mini model to analyze customer survey feedback. It automatically classifies sentiment, identifies the main and sub-reasons for feedback, and assigns a high-level category (People, Process, Service, Technology).

## Features
- **Structured JSON Output:** Uses OpenAI's JSON mode for reliable data parsing.
- **Progress Tracking:** Includes a `tqdm` progress bar for batch processing.
- **Error Handling:** Robust handling for API timeouts or malformed responses.

## Prerequisites
- Python 3.8+
- OpenAI API Key

## Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/customer-feedback-analysis.git](https://github.com/yourusername/customer-feedback-analysis.git)
   cd customer-feedback-analysis