import pandas as pd
from typing import Dict
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from textblob import TextBlob
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from datetime import datetime
import os


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

class AdvancedSentimentAnalyzer:
    def __init__(self, model_path: str = None):
       
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.model = None
        self.ensemble_model = None
        
        if model_path:
            self.load_model(model_path)
    
    def preprocess_text(self, text: str) -> str:
        
        if not isinstance(text, str):
            return ""
            
        
        text = text.lower()
        
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
       
        text = re.sub(r'\@\w+|\#', '', text)
        
        text = re.sub(r'[^\w\s]', '', text)
        
        words = nltk.word_tokenize(text)
        
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        return ' '.join(words)
    
    def combine_text_features(self, row: pd.Series) -> str:
       
        title = str(row['Title']) if 'Title' in row and pd.notna(row['Title']) else ""
        description = str(row['Description']) if 'Description' in row and pd.notna(row['Description']) else ""
        return f"{title} {description}".strip()
    
    def train_model(self, dataset_path: str, 
                   test_size: float = 0.2, random_state: int = 42):
        
        try:
           
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
            
           
            print(f"Loading dataset from: {dataset_path}")
            df = pd.read_csv(dataset_path)
            
           
            required_columns = ['Title', 'Description', 'Sentiment']
            available_columns = df.columns.tolist()
            
            
            column_mapping = {}
            for col in required_columns:
                match = next((c for c in available_columns if c.lower() == col.lower()), None)
                if not match:
                    raise ValueError(f"Required column not found: {col}")
                column_mapping[col] = match
            
            print("Detected columns:")
            for standard_col, actual_col in column_mapping.items():
                print(f" - {standard_col}: {actual_col}")
            
          
            df = df.rename(columns={v: k for k, v in column_mapping.items()})
            
           
            df['combined_text'] = df.apply(self.combine_text_features, axis=1)
            df['cleaned_text'] = df['combined_text'].apply(self.preprocess_text)
            
            
            sentiment_map = {'negative': -1, 'neutral': 0, 'positive': 1}
            
            if df['Sentiment'].dtype in [np.int64, np.float64]:
                df['sentiment_num'] = df['Sentiment']
            else:
                df['sentiment_num'] = df['Sentiment'].str.lower().map(sentiment_map)
            
            
            initial_count = len(df)
            df = df.dropna(subset=['sentiment_num'])
            final_count = len(df)
            
            if initial_count != final_count:
                print(f"Removed {initial_count - final_count} rows with null/invalid sentiment values")
            
           
            X_train, X_test, y_train, y_test = train_test_split(
                df['cleaned_text'], df['sentiment_num'], 
                test_size=test_size, random_state=random_state
            )
            
            
            X_train_vec = self.vectorizer.fit_transform(X_train)
            X_test_vec = self.vectorizer.transform(X_test)
            
            
            lr = LogisticRegression(max_iter=1000, random_state=random_state)
            nb = MultinomialNB()
            svm = SVC(kernel='linear', probability=True, random_state=random_state)
            
            
            self.ensemble_model = VotingClassifier(
                estimators=[('lr', lr), ('nb', nb), ('svm', svm)],
                voting='soft'
            )
            
           
            print("Training ensemble model...")
            self.ensemble_model.fit(X_train_vec, y_train)
            
           
            y_pred = self.ensemble_model.predict(X_test_vec)
            print("\nEnsemble Model Evaluation:")
            print(classification_report(y_test, y_pred, target_names=sentiment_map.keys()))
            print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
            
           
            self.model = lr.fit(X_train_vec, y_train)
            y_pred_lr = self.model.predict(X_test_vec)
            print("\nSingle Model (Logistic Regression) Evaluation:")
            print(classification_report(y_test, y_pred_lr, target_names=sentiment_map.keys()))
            
            return True
            
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            return False
    
    def save_model(self, path: str):
        
        joblib.dump({
            'model': self.model,
            'ensemble_model': self.ensemble_model,
            'vectorizer': self.vectorizer
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a pre-trained model and vectorizer"""
        try:
            data = joblib.load(path)
            self.model = data['model']
            self.ensemble_model = data['ensemble_model']
            self.vectorizer = data['vectorizer']
            print(f"Model loaded from {path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def analyze_article(self, article_data: Dict) -> Dict:
        
        try:
           
            title = str(article_data.get('Title', article_data.get('title', '')))
            description = str(article_data.get('Description', article_data.get('description', '')))
            combined_text = f"{title} {description}".strip()
            
            
            source = article_data.get('Source', article_data.get('source', 'unknown'))
            author = article_data.get('Author', article_data.get('author', 'unknown'))
            
            
            published_at = article_data.get('Published At', article_data.get('published_at'))
            if published_at:
                try:
                    pub_date = datetime.strptime(published_at, '%Y-%m-%d %H:%M:%S')
                    date_str = pub_date.strftime('%B %d, %Y')
                except:
                    date_str = published_at
            else:
                date_str = "Unknown date"
            
            
            analysis_result = self.analyze_sentiment(combined_text)
            
            return {
                "source": source,
                "author": author,
                "title": title,
                "published_at": date_str,
                "sentiment": analysis_result['sentiment'],
                "sentiment_score": analysis_result['combined_score'],
                "emoji": analysis_result['emoji'],
                "analysis_method": analysis_result['analysis_method'],
                "confidence": analysis_result['model_confidence'],
                "url": article_data.get('URL', article_data.get('url', '')),
                "type": article_data.get('Type', article_data.get('type', 'unknown'))
            }
            
        except Exception as e:
            print(f"Error analyzing article: {str(e)}")
            return {
                "error": str(e),
                "sentiment": "Unknown",
                "sentiment_score": 0,
                "emoji": "❓"
            }
    
    def analyze_sentiment(self, text: str, use_ensemble: bool = True) -> Dict:
       
        try:
            #
            cleaned_text = self.preprocess_text(text)
            
           
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            
            model_pred = None
            model_confidence = None
            if self.ensemble_model or self.model:
                try:
                    vec_text = self.vectorizer.transform([cleaned_text])
                    if use_ensemble and self.ensemble_model:
                        pred_proba = self.ensemble_model.predict_proba(vec_text)[0]
                        model_pred = self.ensemble_model.predict(vec_text)[0]
                        model_confidence = max(pred_proba)
                    elif self.model:
                        pred_proba = self.model.predict_proba(vec_text)[0]
                        model_pred = self.model.predict(vec_text)[0]
                        model_confidence = max(pred_proba)
                except Exception as e:
                    print(f"Model prediction warning: {e}")
            
            
            combined_score = polarity
            if model_pred is not None:
                combined_score = 0.7 * model_pred + 0.3 * polarity
            
            # Determine sentiment
            if combined_score > 0.2:
                sentiment = "Positive"
                emoji_icon = "😊"
            elif combined_score < -0.2:
                sentiment = "Negative"
                emoji_icon = "😞"
            else:
                sentiment = "Neutral"
                emoji_icon = "😐"
            
            return {
                "original_text": text,
                "cleaned_text": cleaned_text,
                "polarity": round(polarity, 3),
                "subjectivity": round(subjectivity, 3),
                "model_prediction": model_pred,
                "model_confidence": round(model_confidence, 3) if model_confidence else None,
                "combined_score": round(combined_score, 3),
                "sentiment": sentiment,
                "emoji": emoji_icon,
                "analysis_method": "ensemble" if (use_ensemble and self.ensemble_model) else "single" if self.model else "textblob"
            }
            
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return {
                "sentiment": "Error",
                "combined_score": 0,
                "emoji": "❌",
                "error": str(e)
            }


if __name__ == "__main__":
    
    analyzer = AdvancedSentimentAnalyzer()
    
    
    dataset_path = r"C:\Users\harsh\OneDrive\Desktop\Tavilynews\utils\news_sentiment_analysis.csv"
    
   
    try:
        print(f"Training model with {dataset_path}...")
        success = analyzer.train_model(dataset_path)
        
        if success:
            model_path = "news_sentiment_model.pkl"
            analyzer.save_model(model_path)
            print(f"Model trained and saved to {model_path}")
        else:
            print("Training failed - using TextBlob fallback only")
    except Exception as e:
        print(f"Training failed: {str(e)}")
    
    
    sample_article = {
        "Title": "Fawcett accused of fronting scheme for hostile takeover",
        "Description": "Business Reporter A POTENTIALLY bruising fight for Gwanda-based gold producer Vubachikwe is brewing after mine owners Forbes & Thompson accused one of the creditors, Fawcett Security Operations, of trying to place the mine under corporate rescue as part of a plot to engineer a hostile takeover by London Stock Exchange-listed Kavango Resources. Fawcett made an [&#8230;]",

        "URL": "https://example.com/article",
        "Published At": "2024-07-12 22:05:43",
        "Type": "Business",
        "Source": "The Chronicle",
        "Author": "Peter"
    }
    
    print("\nRunning sample analysis...")
    result = analyzer.analyze_article(sample_article)
    
    print("\nAnalysis Results:")
    for key, value in result.items():
        print(f"{key.title()}: {value}")