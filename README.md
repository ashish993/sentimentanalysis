# Customer Support Sentiment Analysis Tool

This application enhances customer support by providing in-depth sentiment analysis, emotion detection, intent analysis, brand impact insights, sensitivity assessment, and flags critical issues in customer conversations. It is designed to assist support teams in quickly understanding the tone, context, and urgency of interactions, helping them respond more effectively and improve overall customer satisfaction.

## Key Features

1. **Sentiment Analysis**:
   - Identifies the sentiment of each sentence as **positive**, **negative**, or **neutral**.

2. **Emotion Detection**:
   - Detects and lists the emotions expressed by both the customer and the agent in each sentence (e.g., **frustration**, **happiness**, **confusion**).

3. **Intent Analysis**:
   - Identifies the primary and secondary intents of the customer in each sentence (e.g., **seeking a refund**, **technical help**, **feedback**).

4. **Brand Impact**:
   - Analyzes the potential impact of each sentence on the brand (e.g., **positive**, **negative**, **neutral**).

5. **Sensitivity Assessment**:
   - Determines the sensitivity level of each sentence (e.g., **low**, **medium**, **high**).

6. **Flag Critical Issues**:
   - Flags critical issues within the conversation, such as **compliance violations**, **fraud**, **security concerns**, or any issue that requires immediate escalation.

## How to Use

### Option 1: Copy-Paste Conversation
- Copy and paste the conversation between the support agent and the customer into the designated input area.
- Click the **Submit** button to generate a detailed analysis.

### Option 2: Upload an Excel Sheet
- Upload a conversation stored in an Excel sheet for analysis.
- Refer to this sample sheet for formatting: [campaign_feedback.xlsx](https://github.com/ashish993/sentimentanalysis/blob/main/campaign_feedback.xlsx).
- Click the **Submit** button to generate a detailed analysis.

### Output Options:
- Download the analysis results in **CSV format**.
- View additional insights, such as a **word cloud**, **key metrics**, and **recommended actions** based on the sentiment analysis.

## Demo
To see a live demo of the application, visit: [Customer Support Sentiment Analysis Tool](https://customersupport-sentimentanalysis.streamlit.app/)

## Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ashish993/sentimentanalysis.git
   ```
2. **Add your Gork key**:
   Ensure your key is set up for the tool to function properly.
3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the application**:
   ```bash
   streamlit run main.py
   ```

## Contributions and Support
For issues, questions, or contributions, please refer to the [GitHub repository](https://github.com/ashish993/sentimentanalysis) for more details.

**Enjoy enhanced customer support management and data-driven insights with this powerful sentiment analysis tool!**
