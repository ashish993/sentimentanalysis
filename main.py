import streamlit as st
import pandas as pd
from groq import Groq
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Initialize client
groqapi_key=groqapi


client = Groq(api_key=groqapi_key)

#mistral_client = Mistral(api_key=mistral_api_key)

# Models
model = "llama3-70b-8192"

# Fuction to get response from LLM
def get_response(input_text):

 
    prompt = """
Analyze the following customer support chat conversation or marketing campaign feedback. Break down the conversation into individual sentences, and provide a detailed analysis based on the following aspects:

1. **Sentiment Analysis**: Identify the sentiment of each sentence as positive, negative, or neutral.
2. **Emotion Detection**: Detect and list the emotions expressed by both the customer and agent in each sentence (e.g., frustration, happiness, confusion, etc.).
3. **Intent Analysis**: Identify the primary and secondary intents of the customer in each sentence (e.g., seeking a refund, technical help, feedback, etc.).
4. **Brand Impact**: Analyze the potential impact of each sentence on the brand (e.g., positive, negative, neutral).
5. **Sensitivity**: Determine the level of sensitivity of each sentence (e.g., low, medium, high).
6. **Flag Critical Issues**: Flag any critical issues in the conversation such as compliance violations, fraud, security concerns, or any issue that needs immediate escalation.

For each sentence, output the analysis in the following tabular format:

| Sentence                                                                 | Sentiment | Emotion Detection                | Intent                                 | Brand Impact | Sensitivity | Flag Critical Issue |
|--------------------------------------------------------------------------|-----------|----------------------------------|----------------------------------------|--------------|-------------|----------------------|
| Sentence 1                                                              | Positive  | Happiness                        | Seeking Information                    | Positive     | Low         | No                   |
| Sentence 2                                                              | Negative  | Frustration                      | Requesting Refund                      | Negative     | High        | Yes                  |
| Sentence 3                                                              | Neutral   | Confusion                        | Clarification on Process               | Neutral      | Medium      | No                   |
| …                                                                        | …         | …                                | …                                      | …            | …           | …                    |

Make sure to:
- Treat each line as a separate sentence for analysis.
- Highlight important data (e.g., flagged critical issues or high brand impact) within the table.
- Avoid adding extra commentary; focus on clear, structured output in the table.

Do not include timestamps. do not add any additional notes.
Review the response below:
 """ + input_text
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

def get_bulk_response(input_texts):
    """
    Sends all input texts to the model at once and returns the responses.

    Args:
        input_texts (list): List of input texts to be analyzed.

    Returns:
        list: List of responses from the model.
    """
    prompt = """
Analyze the following customer support chat conversations or marketing campaign feedback. Break down the conversation into individual sentences, and provide a detailed analysis based on the following aspects:

1. **Sentiment Analysis**: Identify the sentiment of each sentence as positive, negative, or neutral.
2. **Emotion Detection**: Detect and list the emotions expressed by both the customer and agent in each sentence (e.g., frustration, happiness, confusion, etc.).
3. **Intent Analysis**: Identify the primary and secondary intents of the customer in each sentence (e.g., seeking a refund, technical help, feedback, etc.).
4. **Brand Impact**: Analyze the potential impact of each sentence on the brand (e.g., positive, negative, neutral).
5. **Sensitivity**: Determine the level of sensitivity of each sentence (e.g., low, medium, high).
6. **Flag Critical Issues**: Flag any critical issues in the conversation such as compliance violations, fraud, security concerns, or any issue that needs immediate escalation.

For each sentence, output the analysis in the following tabular format:

| Sentence                                                                 | Sentiment | Emotion Detection                | Intent                                 | Brand Impact | Sensitivity | Flag Critical Issue |
|--------------------------------------------------------------------------|-----------|----------------------------------|----------------------------------------|--------------|-------------|----------------------|
| Sentence 1                                                              | Positive  | Happiness                        | Seeking Information                    | Positive     | Low         | No                   |
| Sentence 2                                                              | Negative  | Frustration                      | Requesting Refund                      | Negative     | High        | Yes                  |
| Sentence 3                                                              | Neutral   | Confusion                        | Clarification on Process               | Neutral      | Medium      | No                   |
| …                                                                        | …         | …                                | …                                      | …            | …           | …                    |

Make sure to:
- Treat each line as a separate sentence for analysis.
- Highlight important data (e.g., flagged critical issues or high brand impact) within the table.
- Avoid adding extra commentary; focus on clear, structured output in the table.

Do not include timestamps. do not add any additional notes.
Review the responses below:
""" + "\n".join(input_texts)
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

def process_excel(file):
    """
    Processes an Excel file, sends all sentences to the model at once for analysis,
    and returns the results as a DataFrame.

    Args:
        file (UploadedFile): The uploaded Excel file containing sentences.

    Returns:
        DataFrame: A DataFrame containing the analysis results for each sentence.
    """
    df = pd.read_excel(file)  # Read the first row as header
    sentences = df.iloc[:, 0].tolist()  # Assuming each row contains one sentence in the first column
    response = get_bulk_response(sentences)
    # Parse the response into a DataFrame
    results = []
    for line in response.split("\n"):
        if line.startswith("|"):
            parts = line.split("|")[1:-1]
            results.append([part.strip() for part in parts])
    columns = ["Sentence", "Sentiment", "Emotion Detection", "Intent", "Brand Impact", "Sensitivity", "Flag Critical Issue"]
    return pd.DataFrame(results, columns=columns)

# Streamlit UI
st.title("Customer Support Sentiment Analysis")
# Input method selection
input_method = st.selectbox("Choose input method:", ["File Upload","Text"])

user_input = None


if input_method == "File Upload":
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
    if uploaded_file is not None:
        df = process_excel(uploaded_file)
if input_method == "Text":
    user_input = st.text_area("Copy paste chat here:")


if st.button("SUBMIT"):
    if user_input and input_method == "Text":
        response = get_response(user_input)
        st.info(response)
    elif input_method == "File Upload" and uploaded_file is not None:
        df = df.iloc[2:]  # Remove the first two rows
        st.dataframe(df)

        k= (' '.join(df['Sentence']))

        st.header("Word Cloud")
        wordcloud = WordCloud(width = 1000, height = 500).generate(k)
        plt.figure(figsize=(15,5))
        plt.imshow(wordcloud)
        plt.axis('off')
        st.pyplot(plt)



        st.header("Key Metrics")
        total_feedback = len(df)
        positive_count = df[df['Sentiment'] == 'Positive'].shape[0]
        negative_count = df[df['Sentiment'] == 'Negative'].shape[0]
        neutral_count = df[df['Sentiment'] == 'Neutral'].shape[0]
        # Total Feedback

        # Total Feedback (Black)
        st.markdown(
        f"""
        <div style="text-align:center; color: #000; font-size: 24px;">
            <strong>Total Feedback</strong><br>
            <span style="font-size: 36px;">{total_feedback}</span>
        </div>
        """,
            unsafe_allow_html=True
            )

        # Positive Feedback (Green)
        st.markdown(
        f"""
        <div style="text-align:center; color: #00cc66; font-size: 24px;">
            <strong>Positive Feedback</strong><br>
            <span style="font-size: 36px;">{positive_count}</span>
        </div>
        """,
            unsafe_allow_html=True
            )
 
    # Negative Feedback (Red)
        st.markdown(
        f"""
        <div style="text-align:center; color: #ff4d4d; font-size: 24px;">
            <strong>Negative Feedback</strong><br>
            <span style="font-size: 36px;">{negative_count}</span>
        </div>
        """,
             unsafe_allow_html=True
            )
         # Neutral Feedback (Black)
        st.markdown(
        f"""
        <div style="text-align:center; color: #000; font-size: 24px;">
            <strong>Neutral Feedback</strong><br>
            <span style="font-size: 36px;">{neutral_count}</span>
        </div>
        """,
            unsafe_allow_html=True
            )
        
        # Sentiment Distribution
        st.subheader("Sentiment Distribution")
        sentiment_counts = df['Sentiment'].value_counts()
        fig1 = go.Figure(go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=0.4,
            marker=dict(colors=["#00CC96", "#EF553B", "#636EFA"]),
            ))
        fig1.update_layout(title_text="Sentiment Distribution", title_x=0.5, template="plotly_dark")
        st.plotly_chart(fig1, use_container_width=True)
 
         #Brand Impact Analysis
        st.subheader("Brand Impact Analysis")
        brand_impact_counts = df['Brand Impact'].value_counts()
        fig2 = go.Figure(go.Bar(
        x=brand_impact_counts.index,
        y=brand_impact_counts.values,
        marker=dict(color="#19D3F3")
            ))
        fig2.update_layout(
            title_text="Brand Impact Distribution",
            xaxis_title="Brand Impact",
            yaxis_title="Count",
            template="plotly_dark"
            )
        st.plotly_chart(fig2, use_container_width=True)
 
        # Critical Issues by Sentiment
        st.subheader("Critical Issues by Sentiment")
        critical_issues = df[df['Flag Critical Issue'].str.contains('Yes', case=False, na=False)]
        critical_sentiment_counts = critical_issues['Sentiment'].value_counts()
        fig3 = go.Figure(go.Bar(
        x=critical_sentiment_counts.index,
        y=critical_sentiment_counts.values,
        marker=dict(color="#EF553B")
            ))
        fig3.update_layout(
        title="Critical Issues by Sentiment",
        xaxis_title="Sentiment",
        yaxis_title="Critical Issue Count",
        template="plotly_dark"
    )
        st.plotly_chart(fig3, use_container_width=True)
 
        # Actionable Recommendations
        st.header("Actionable Recommendations")
        if negative_count > positive_count:
            st.write("- **Increase Positive Sentiment**: Focus on creating content that promotes positive sentiment among customers.")
        if 'Complaint' in df['Intent'].values:
            st.write("- **Address Complaints**: Prioritize resolving complaints that have a high impact on brand perception.")
        if 'Frustration' in df['Emotion Detection'].values:
            st.write("- **Reduce Frustration**: Take steps to address sources of customer frustration to improve overall sentiment.")
 
        # Customer Service Prioritization for High-Sensitivity Issues
        st.subheader("Customer Service Prioritization")
        st.write("High-sensitivity, high-impact, negative feedback requiring immediate attention:")
        urgent_issues = df[(df['Sentiment'] == 'Negative') &
                         (df['Brand Impact'] == 'High') &
                         (df['Sensitivity'] == 'High') &
                         (df['Flag Critical Issue'] == 'Yes')]
        if not urgent_issues.empty:
            st.write(urgent_issues[['Question', 'Sentiment', 'Emotion Detection', 'Intent', 'Brand Impact', 'Sensitivity']])
        else:
            st.write("No urgent issues requiring immediate attention were found.")

    else:
        st.info("Please provide input for analysis.")
