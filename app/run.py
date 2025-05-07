import os
import re
import json
import string
import pandas as pd
import joblib
import plotly
import logging
import plotly.express as px
import seaborn as sns
from sqlalchemy import create_engine
from flask import Flask, render_template, request, redirect , flash
from flask_sqlalchemy import SQLAlchemy
from plotly.graph_objs import Bar
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from functools import lru_cache
from models.train_classifier import *

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///feedback.db'
app.secret_key = os.urandom(24)
db = SQLAlchemy(app)


# ================== NEW: Feedback Model ==================


class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    message = db.Column(db.String(500))
    prediction = db.Column(db.String(500))
    is_correct = db.Column(db.Boolean)


# ================== NEW: Recommendations ==================
RECOMMENDATIONS = {
    'water': 'Contact Water.org and local authorities for water assistance.',
    'medical_help': 'Alert Red Cross and local hospitals for medical support.',
    'food': 'Reach out to World Food Programme and local food banks.'
}

# Load data and model
db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'DisasterResponse.db')
engine = create_engine(f'sqlite:///{os.path.abspath(db_path)}')
df = pd.read_sql_table('DisasterResponse', engine)
@lru_cache(maxsize=1)
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'classifier.pkl')
    return joblib.load(os.path.abspath(model_path))


model = load_model()


# ================== NEW: Enhanced Visualizations ==================


def create_visualizations():
    """Generate all visualizations for the dashboard."""
    visuals = {}

    # Category distribution
    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    visuals['category_dist'] = px.bar(
        x=category_counts.index.str.replace('_', ' ').str.title(),
        y=category_counts.values,
        labels={'x': 'Category', 'y': 'Count'},
        title='Message Category Distribution'
    )

    # Time series analysis
    df['date'] = pd.to_datetime(df['message'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0], errors='coerce')
    time_series = df.groupby([df['date'].dt.to_period('M'), 'genre']).size().unstack()
    visuals['time_series'] = px.line(
        time_series,
        title='Messages Over Time by Genre'
    )

    # Correlation heatmap
    corr_matrix = df.iloc[:, 4:-1].corr()
    visuals['correlation'] = px.imshow(
        corr_matrix,
        title='Category Correlation Heatmap'
    )

    # New: Message Length Distribution
    df['msg_len'] = df['message'].apply(lambda x: len(x))
    visuals['msg_len_dist'] = px.histogram(
        df, x='msg_len', nbins=50,
        title='Distribution of Message Lengths',
        labels={'msg_len': 'Message Length (chars)'}
    )

    # New: Top Words
    from collections import Counter
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))

    all_words = ' '.join(df['message'].dropna()).lower()
    tokens = word_tokenize(all_words)
    words = [word for word in tokens if word.isalpha() and word not in stop_words]
    common_words = Counter(words).most_common(10)
    word_df = pd.DataFrame(common_words, columns=['Word', 'Count'])

    visuals['top_words'] = px.bar(
        word_df, x='Word', y='Count', title='Top 10 Most Frequent Words'
    )

    return visuals


# ================== UPDATED: Enhanced Index Route ==================


@app.route('/')
def index():
    visuals = create_visualizations()
    graphJSON = json.dumps([fig.to_dict() for fig in visuals.values()])
    ids = [f"graph-{i}" for i in range(len(visuals))]
    return render_template('master.html', graphJSON=graphJSON, ids=json.dumps(ids))


# ================== UPDATED: Enhanced Classification Route ==================


@app.route('/go', methods=['GET', 'POST'])
def go():
    if request.method == 'POST':
        # Handle feedback
        feedback = request.form.get('feedback')
        if feedback:
            new_feedback = Feedback(
                message=request.form['message'],
                prediction=request.form['prediction'],
                is_correct=(feedback == 'accurate')
            )
            db.session.add(new_feedback)
            db.session.commit()
            flash("Thank you! Your feedback has been submitted.")
            return redirect('/')


    query = request.args.get('query', '')
    if not query.strip():
        return render_template('go.html', error="Please enter a message.")

    try:
        pred = model.predict([query])[0]
        results = dict(zip(df.columns[4:-1], pred))
        recommendations = [RECOMMENDATIONS[cat] for cat in results if results[cat] and cat in RECOMMENDATIONS]

        return render_template('go.html',
                       query=query,
                       classification_result=results,
                       recommendations=recommendations)

    except Exception as e:
        logging.exception("Error during classification")
        return render_template('go.html', error=f"Error processing request: {str(e)}")



if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=3001, debug=True)
