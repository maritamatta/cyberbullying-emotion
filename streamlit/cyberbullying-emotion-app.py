import streamlit as st
import hydralit_components as hc
from PIL import Image
import base64
import joblib

# Data manipulation libraries
import pandas as pd
import numpy as np

# Visualization libraries
import scipy
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from wordcloud import WordCloud, ImageColorGenerator
from wordcloud import STOPWORDS
from collections import Counter
import plotly.io as pio
from plotly.subplots import make_subplots

# Text analysis and NLP libraries
import re
import time
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
import gensim
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Machine learning libraries
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from nltk.corpus import stopwords
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.sequence import pad_sequences
import joblib
from textblob import TextBlob  # Import TextBlob for sentiment analysis

# Utility Functions
import plotly.express as px
from collections import Counter
from nltk.corpus import stopwords
import pandas as pd
from joblib import load

    # Show the Plotly figure
def word_bar_graph(df, column, title, num_words=15, plot_height=600, plot_width=800):
    # Tokenize the text and count word occurrences
    topic_words = [word.lower() for text in df[column] if isinstance(text, str) for word in text.split()]
    word_count_dict = dict(Counter(topic_words))
    
    # Remove stopwords
    popular_words_nonstop = [w for w in word_count_dict if w not in STOPWORDS]
    
    # Select top words and their counts
    selected_words = popular_words_nonstop[:num_words]
    selected_counts = [word_count_dict[word] for word in selected_words]
    
    # Create a DataFrame for Plotly
    data = pd.DataFrame({'Word': reversed(selected_words), 'Count': reversed(selected_counts)})
    
    # Create a horizontal bar chart using Plotly
    fig = px.bar(data, x='Count', y='Word', orientation='h', title=title, color_discrete_sequence=['mediumslateblue'])
    fig.update_layout(showlegend=False, height=plot_height, width=plot_width)

    # Show the Plotly figure
    return fig

#read the dataset
kaggle_df = pd.read_csv('data/tweet_emotions.csv').drop('tweet_id', axis=1)
kaggle_df = kaggle_df.rename(columns={'content': 'text', 'sentiment': 'label'})
huggingface_df = pd.read_parquet('data/huggingface_emotions.parquet')
# Replace numerical labels with corresponding emotions gotten from the metadata on hugging face's website
label_to_emotion = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

# Mapping the enocded labels to corresponding emotions 
huggingface_df['label'] = huggingface_df['label'].map(label_to_emotion)

# Merge the datasets based on the common columns
emotional_detection = pd.concat([huggingface_df, kaggle_df])

# Display the first few rows of the merged dataset
emotional_detection.head()

# The data are sources from twitter, kaggle, wikepedia talk page, youtube
aggression = pd.read_csv('data/aggression_parsed_dataset.csv')
attack = pd.read_csv('data/attack_parsed_dataset.csv')
toxicity = pd.read_csv('data/toxicity_parsed_dataset.csv')
racism = pd.read_csv('data/twitter_racism_parsed_dataset.csv')
sexism = pd.read_csv('data/twitter_sexism_parsed_dataset.csv')
kaggle = pd.read_csv('data/kaggle_parsed_dataset.csv')
twitter = pd.read_csv('data/twitter_parsed_dataset.csv')
youtube = pd.read_csv('data/youtube_parsed_dataset.csv')

# Add a new column to each DataFrame indicating the source dataset
aggression['source'] = 'aggression'
attack['source'] = 'attack'
toxicity['source'] = 'toxicity'
racism['source'] = 'racism'
sexism['source'] = 'sexism'
kaggle['source'] = 'kaggle'
twitter['source'] = 'twitter'
youtube['source'] = 'youtube'

# Concatenate all DataFrames along with the newly added 'source' column
cyberbullying_data = pd.concat([aggression, attack, toxicity, racism, sexism, kaggle, twitter, youtube], ignore_index=True)
cyberbullying_data = cyberbullying_data.rename(columns={'ed_label_0': 'Non-Cyberbully'})
cyberbullying_data = cyberbullying_data.rename(columns={'ed_label_1': 'Cyberbully'})
# Rename column oh_label to cyberbullying
cyberbullying_data = cyberbullying_data.rename(columns={'oh_label': 'cyberbullying'})
# Now you have a single DataFrame containing all the data from different sources
# with an additional 'source' column to indicate the dataset origin

# Make it look nice from the start
st.set_page_config(page_title='MSBA Thesis', layout='wide', initial_sidebar_state='auto')

primaryColor="#8A419C"
backgroundColor="#E6E6FA"
secondaryBackgroundColor="#262730"
textColor="#000000"

# Specify the primary menu definition with custom CSS
menu_data = [
    {'icon': "", 'label': "     "},
    {'icon': "", 'label': "     "},
    {'icon': "📚", 'label': "Literature Review"},
    {'icon': "", 'label': "     "},
    {'icon': "", 'label': "     "},
    #{'icon': "📊", 'label': "Statistics"},
    {'icon': "📈", 'label': "Exploratory Data Analysis"},
    {'icon': "", 'label': "     "},
    {'icon': "", 'label': "     "},
    {'icon': "🚫", 'label': "Enhanced Cyberbullying Model"},
    {'icon': "", 'label': "     "},
    {'icon': "", 'label': "     "},
    #{'icon': "😀", 'label': "Cyberbullying Detection"},
    #{'icon': "📈", 'label': "Dashboard", 'ttip': "My Model"},
    {'icon': "🔍", 'label': "Comparative Analysis"}
]

# Create the navigation bar
over_theme = {
    'txc_inactive': 'white',
    'menu_background': 'purple',
    'txc_active': 'grey',
    'option_active': 'white',
    "font": 'serif',
    'menu_height': '300px',
    'menu_width': '80px'  # Adjust the width of the menu bar
}

# Create the navigation menu
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    home_name='Home',
    hide_streamlit_markers=True,
    sticky_nav=True,
    sticky_mode='pinned'
)

# Check if the Home button is clicked


# Function to encode the image to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    return encoded_image

# Set the background image path
background_image_path = 'streamlit/utility-images/background_image.png' 

# Encode the image to base64
encoded_background_image = get_base64_of_bin_file(background_image_path)

# Set the opacity level (from 0 to 1)
opacity_level = 0.6

# Set the background image path
lit_image_path = 'streamlit/utility-images/Timeline.png'

# Your Streamlit content goes here
if menu_id == 'Home':
    # Use HTML and CSS to set the background image and opacity
    background_image_code = f"""
        <style>
            body {{
                background-image: url('data:image/png;base64,{encoded_background_image}');
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
                opacity: {opacity_level};
            }}
        </style>
    """
    
    # Display the background image with Streamlit markdown
    st.markdown(background_image_code, unsafe_allow_html=True)

    # Your content for the Home page
    st.header('Leveraging a BiLSTM-based Emotion Recognition Transfer Learning Model for complex phrasal analysis in cyberbullying detection')
    st.write("This research aims to investigate the role of emotions, particularly those triggered by images or text, in the context of cyberbullying on social media platforms. The study seeks to understand how users harness emotional cues as triggers for cyberbullying incidents and to develop targeted strategies for the early detection and effective mitigation of cyberbullying. The research will shed light on the specific emotional elements and nuances within social media platforms that serve as catalysts for cyberbullying behavior. By gaining a deep understanding of these emotional triggers, the research will empower both scholars and practitioners to explore viable constraints and interventions aimed at discouraging the exploitation of these triggers in the perpetration of cyberbullying. The study will contribute to the development of robust countermeasures and preventive measures that can be seamlessly integrated into social media platforms. Ultimately, the overarching goal is to cultivate a safer and more nurturing online environment where the presence of emotional triggers leading to cyberbullying can be promptly identified and effectively mitigated. The research aspires to create a digital sphere where users can engage in positive and empathetic interactions, free from the perils of cyberbullying.")

# Inside the conditional block for 'Literature Review'
if menu_id == 'Literature Review':
    # Encode the image to base64
    encoded_lit_image = get_base64_of_bin_file(lit_image_path)

    # Create HTML code with embedded image
    lit_image_code = f'<img src="data:image/png;base64,{encoded_lit_image}" style="width:100%">'

    # Display the HTML code with Streamlit markdown
    st.markdown(lit_image_code, unsafe_allow_html=True)

# Check if the Exploratory Data Analysis button is clicked
if menu_id == 'Exploratory Data Analysis':
    # Create two columns for the buttons
    col1, col2, col3, col4 = st.columns(4)

    if col2.button("Emotions Analysis"):
        # Create a 2-row, 2-column layout
        col1, col2 = st.columns(2)
    
        # Assuming you have a DataFrame called emotional_detection
        fig = px.bar(
            x=emotional_detection['label'].value_counts(),
            y=emotional_detection['label'].unique(),
            orientation='h',  # Horizontal bar chart
            title='Available Emotions',
            labels={'count': 'Count', 'y': 'Emotion'},
            color_discrete_sequence=['mediumslateblue'],  # Set the color to mediumslateblue
            opacity=0.8  # Set the opacity to 0.8 for a bit of transparency
        )

        fig.update_layout(xaxis_title='Count', yaxis_title='Emotion')

        # Show the bar chart in the first column (row 1)
        col1.plotly_chart(fig)
    
        # Number of total rows
        total_rows = emotional_detection['text'].shape[0]

        # Number of duplicated rows
        duplicated_rows = emotional_detection['text'].duplicated().sum()

        # Calculate the number of unique rows
        unique_rows = total_rows - duplicated_rows

        # Create a DataFrame for the pie chart
        data = {'Category': ['Unique Rows', 'Duplicated Rows'], 'Count': [unique_rows, duplicated_rows]}
        df = pd.DataFrame(data)

        # Create the pie chart using Plotly
        pie_fig = px.pie(df, names='Category', values='Count', title='Proportion of Unique and Duplicated Rows',  color_discrete_sequence=['orchid', 'mediumslateblue'], opacity=0.8)

        # Show the pie chart in the second column (row 1)
        col2.plotly_chart(pie_fig)
    
        # Load your custom brain-shaped mask image
        mask_image = np.array(Image.open('streamlit/utility-images/human thought.jpg'))

        # Create a WordCloud object with your mask and custom colors
        wordcloud = WordCloud(width=150, height=300, background_color=None, mode='RGBA', colormap='magma', max_words=200, mask=mask_image)

        # Generate the WordCloud
        wordcloud.generate(' '.join(emotional_detection['text']))

        # Display the WordCloud with the brain shape and custom colors (row 2)
        col1.image(wordcloud.to_image())

        # Assuming 'emotional_detection' is your DataFrame
        #word_bar_graph(emotional_detection, 'text', "Most Occurred Words")  # Use 'emotional_detection' instead of 'emotional_detection'
        # Assuming 'emotional_detection' is your DataFrame
        fig = word_bar_graph(emotional_detection, 'text', "Most Occurred Words")
        col2.plotly_chart(fig)

        # Show the word bar graph in the second column (row 2)
        #col2.plotly_chart(px.bar(data, x='Count', y='Category', orientation='h', title="Most Occurred Words"))

        emotional_detection['length'] = emotional_detection['text'].apply(lambda x: len(x))

       # Create a KDE plot using seaborn with a transparent background
        plt.figure(figsize=(10, 6))

        # Assuming 'emotional_detection' is your DataFrame
        agg_df = emotional_detection.groupby(['length', 'label']).size().reset_index(name='count')

        # Set the color palette to purple
        sns.set_palette("magma")

        # Plot the KDE plot (row 2)
        ax = sns.kdeplot(data=agg_df, x="length", hue="label", fill=True, common_norm=False, linewidth=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        # Remove unnecessary text
        plt.xlabel('')
        plt.ylabel('')
        plt.title('')

        # Remove legend title
        plt.gca().legend().set_title('')

        # Display the plot without the background
        col1.pyplot(plt, transparent=True)


        # Assuming 'emotional_detection' is your DataFrame
        emotional_detection_filtered = emotional_detection[emotional_detection['length'] < 30]

        # Get the value counts of 'length' and sort by index in descending order
        value_counts_sorted = emotional_detection_filtered['length'].value_counts().sort_index(ascending=False)
        
        countplot_fig = plt.figure(figsize=(14, 8))
        ax = sns.countplot(x='length', data=emotional_detection_filtered, order=value_counts_sorted.index, palette='magma')  # Use 'magma' color palette
        plt.title('Count of tweets with less than 30 words', fontsize=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.yticks([])

        # Add count labels on top of the bars
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points')

        plt.ylabel('Count')
        plt.xlabel('Word Length')

        # Remove background
        countplot_fig.patch.set_alpha(0)

        # Display the plot without the background in col2
        col2.pyplot(countplot_fig, transparent=True)

    if col4.button("Cyberbullying Analysis"):
        # Create a 2-row, 2-column layout
        col1, col2 = st.columns(2)

        # Calculate the counts of unique values in the 'source' column
        value_counts = cyberbullying_data['source'].value_counts()
        source_names = value_counts.index

        # Filter the data for the first pie chart (categories: 'Agression', 'attack', 'toxicity', 'sexism', 'racism')
        first_chart_data = cyberbullying_data[cyberbullying_data['source'].isin(['Agression', 'attack', 'toxicity', 'sexism', 'racism'])]

        # Count the occurrences of each category
        first_value_counts = first_chart_data['source'].value_counts()
        first_source_names = first_value_counts.index.tolist()

        # Create the first pie chart
        first_chart_fig = px.pie(
            values=first_value_counts,
            names=first_source_names,
            color_discrete_sequence=['purple', 'thistle'],
            hole=0.3,
        )
        first_chart_fig.update_traces(textinfo='percent+label')
        first_chart_fig.update_layout(title="Toxic Behaviors")

        # Display the first chart
        col1.plotly_chart(first_chart_fig)

        # Filter the data for the second pie chart (categories: 'kaggle', 'youtube', 'twitter')
        second_chart_data = cyberbullying_data[cyberbullying_data['source'].isin(['kaggle', 'youtube', 'twitter'])]

        # Count the occurrences of each category
        second_value_counts = second_chart_data['source'].value_counts()
        second_source_names = second_value_counts.index.tolist()

        # Create the second pie chart
        second_chart_fig = px.pie(
            values=second_value_counts,
            names=second_source_names,
            color_discrete_sequence=['purple', 'thistle'],
            hole=0.3,
        )
        second_chart_fig.update_traces(textinfo='percent+label')
        second_chart_fig.update_layout(title="Sources")

        # Display the second chart
        col1.plotly_chart(second_chart_fig)

        corr_matrix = cyberbullying_data[['Non-Cyberbully', 'Cyberbully', 'cyberbullying']].corr()
        corr_matrix
        # Create a heatmap using Plotly graph objects
        heatmap = go.Figure(data=go.Heatmap(
                   z=corr_matrix,
                   x=['Non-Cyberbully', 'Cyberbully', 'cyberbullying'],
                   y=['Non-Cyberbully', 'Cyberbully', 'cyberbullying'],
                   colorscale='purples'))

        # Customize the layout
        heatmap.update_layout(
            title="Correlation Matrix Heatmap",
            xaxis_title="Features",
            yaxis_title="Features"
        )

        # Show the heatmap in the first column (col1)
        col1.plotly_chart(heatmap)

        clusters_image_path = 'streamlit/utility-images/clusters.png'
        # Encode the image to base64
        encoded_clusters_image = get_base64_of_bin_file(clusters_image_path)

        # Create HTML code with embedded image
        clusters_image_code = f'<img src="data:image/png;base64,{encoded_clusters_image}" style="width:60%; height:60%">'

        # Display the HTML code with Streamlit markdown
        col2.markdown(clusters_image_code, unsafe_allow_html=True)

# Load the pre-trained model
model = load_model("models/emotioncyberbullying/NN_cyberbullying_emotion.h5")

# Load the MultiLabelBinarizer used during training
mlb_path = "models/emotioncyberbullying/mlb.pkl"
mlb = joblib.load(mlb_path)

# Load the Tokenizer
tokenizer_path = "models/emotioncyberbullying/tokenizer.pkl"
tokenizer = joblib.load(tokenizer_path)

# Max sequence length and number of classes
max_len = 100
num_classes = len(mlb.classes_)

# Inside the conditional block for 'Enhanced Cyberbullying Model'
if menu_id == 'Enhanced Cyberbullying Model':
    # Create a text input box for the user to enter a sentence
    user_input = st.text_input("Enter a sentence:")

    # Add custom styling to the text input
    st.markdown(
        """
        <style>
        div[data-baseweb="input"] input {
            color: white !important;
            background-color: black !important;
            width: 80%;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Create a button to trigger the model prediction
    predict_button = st.button("Predict")

    if predict_button:
        if user_input:
            # Tokenize and pad the input sentence
            input_seq = tokenizer.texts_to_sequences([user_input])
            input_padded = pad_sequences(input_seq, maxlen=max_len, padding='post')

            # Make the prediction
            prediction = model.predict(input_padded)
            prediction_binary = (prediction > 0.5).astype(int)

            # Convert the prediction back to original labels
            predicted_labels = mlb.inverse_transform(prediction_binary)

            # Assuming predicted_labels is a tuple with format: [('1.0', 'hate')]
            predicted_cyberbully_label, predicted_emotion_label = predicted_labels[0]

            # Convert Cyberbully Label to a more readable format
            cyberbully_label = "Cyberbully" if predicted_cyberbully_label == '1.0' else "Non-Cyberbully"

            # Analyze sentiment and polarity using TextBlob
            blob = TextBlob(user_input)
            sentiment = blob.sentiment
            sentiment_polarity = sentiment.polarity
            # Determine sentiment label
            sentiment_label = "Positive" if sentiment_polarity > 0 else "Negative" if sentiment_polarity < 0 else "Neutral"

            # Display results
            st.markdown(f'<div style="background-color: #9E9AC8; padding: 10px; border-radius: 5px;">'
                   f'<b>Cyberbully Label:</b> {cyberbully_label}<br>'
                   f'<b>Emotion Label:</b> {predicted_emotion_label}<br>'
                   f'<b>Sentiment Label:</b> {sentiment_label}<br>'
                   f'<b>Sentiment Polarity:</b> {sentiment_polarity}'
                   '</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter a sentence for prediction.")

if menu_id == 'Comparative Analysis':
    # Create two columns for the buttons
    col1, col2 = st.columns(2)
    # Paired T-Test Results
    t_statistic = 4.36
    p_value = 1.3054754339706072e-05

    # Display paired t-test results
    col1.subheader("Paired T-Test Results:")
    col1.text(f"t-statistic: {t_statistic}")
    col1.text(f"p-value: {p_value}")
    col1.text("Reject the null hypothesis: There is a significant difference in performance.")
    # McNemar's Results
    chi_statistic = 453.56
    p_value = 2.2956891179504675e-18

    # Display paired t-test results
    col2.subheader("McNemar's Results:")
    col2.text(f"t-statistic: {chi_statistic}")
    col2.text(f"p-value: {p_value}")
    col2.text("Reject the null hypothesis: There is a significant difference in performance.")

    # Your accuracy scores
    your_accuracy = 66
    ieee_accuracy = 48

    # Create a bar chart for comparison using Plotly Express
    fig = px.bar(x=['My Model', 'IEEE Research Model'], y=[your_accuracy, ieee_accuracy],
                 text=[f"{your_accuracy}%", f"{ieee_accuracy}%"],
                 labels={'y': 'Accuracy (%)', 'x':'Models'},
                 color=['My Model', 'IEEE Research Model'],
                 color_discrete_map={'My Model': '#800080', 'IEEE Research Model': '#9E9AC8'})

    # Customize layout
    fig.update_traces(textposition='outside')
    fig.update_layout(height=400, width=600, showlegend=False, title="Model Accuracy Comparison")

    # Display the chart
    col1.plotly_chart(fig)
