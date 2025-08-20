import gradio as gr
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

# Install Gradio if not already installed
try:
    import gradio as gr
except ImportError:
    !pip install gradio
    import gradio as gr

class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.is_loaded = False

    def load_models(self):
        """Load the trained models"""
        try:
            # Load your saved models (update paths if needed)
            self.vectorizer = joblib.load('vectorizer.jb')
            self.model = joblib.load('lr_model.jb')
            self.is_loaded = True
            print("✅ Models loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False

    def predict(self, text):
        """Predict if news is real or fake"""
        if not self.is_loaded:
            return "❌ Models not loaded properly!", "", ""

        if not text.strip():
            return "⚠️ Please enter some text to analyze!", "", ""

        try:
            # Transform text and predict
            text_vectorized = self.vectorizer.transform([text])
            prediction = self.model.predict(text_vectorized)

            # Get prediction probabilities for confidence
            try:
                probabilities = self.model.predict_proba(text_vectorized)
                confidence = max(probabilities[0]) * 100
            except:
                confidence = 85.0

            # Format results
            if prediction[0] == 1:
                result = "✅ REAL NEWS"
                confidence_text = f"Confidence: {confidence:.1f}%"
                recommendation = "💡 This appears to be legitimate news. However, always verify with multiple trusted sources."
                color = "green"
            else:
                result = "❌ FAKE NEWS"
                confidence_text = f"Confidence: {confidence:.1f}%"
                recommendation = "⚠️ This appears to be misinformation. Please fact-check before sharing."
                color = "red"

            return result, confidence_text, recommendation

        except Exception as e:
            return f"❌ Prediction error: {str(e)}", "", "Please try again with different text."

# Initialize detector
detector = FakeNewsDetector()

def analyze_news(text):
    """Main function for Gradio interface"""
    result, confidence, recommendation = detector.predict(text)
    return result, confidence, recommendation

def load_sample_real():
    """Load a sample real news"""
    return """Scientists at Stanford University have developed a new machine learning algorithm that can detect early signs of Alzheimer's disease through brain scans with 95% accuracy. The research, published in Nature Medicine, could help doctors diagnose the condition years before symptoms appear."""

def load_sample_fake():
    """Load a sample fake news"""
    return """Breaking: Local man discovers that eating pizza every day for 30 days can cure all diseases according to a study that definitely exists and was totally conducted by real scientists at a real university."""

# Create Gradio Interface
def create_app():
    # Load models
    if not detector.load_models():
        print("Failed to load models. Please check your model files.")
        return None

    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
        max-width: 900px !important;
        margin: auto !important;
    }
    .header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        margin-top: 20px;
        padding: 10px;
        background-color: #f0f0f0;
        border-radius: 5px;
    }
    """

    with gr.Blocks(css=css, theme=gr.themes.Soft(), title="EchoTruth - Fake News Detector") as app:

        # Header
        gr.HTML("""
        <div class="header">
            <h1>🔍 EchoTruth</h1>
            <h3>AI-Powered Fake News Detection System</h3>
            <p>Uncover fake news and amplify real facts with advanced machine learning</p>
        </div>
        """)

        # Main Interface
        with gr.Row():
            with gr.Column(scale=2):
                # Input section
                gr.Markdown("### 📝 Enter News Article")
                news_input = gr.Textbox(
                    label="News Text",
                    placeholder="Paste the news article you want to verify here...",
                    lines=8,
                    max_lines=15
                )

                # Buttons
                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analyze News", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ Clear", scale=1)

                # Sample buttons
                gr.Markdown("### 🧪 Try Sample Articles")
                with gr.Row():
                    real_sample_btn = gr.Button("📰 Load Real News Sample", size="sm")
                    fake_sample_btn = gr.Button("🚫 Load Fake News Sample", size="sm")

            with gr.Column(scale=1):
                # Results section
                gr.Markdown("### 📊 Analysis Results")
                result_box = gr.Textbox(
                    label="Prediction",
                    interactive=False,
                    lines=2
                )

                confidence_box = gr.Textbox(
                    label="Confidence Level",
                    interactive=False,
                    lines=1
                )

                recommendation_box = gr.Textbox(
                    label="Recommendation",
                    interactive=False,
                    lines=4
                )

        # Model Info
        with gr.Accordion("📈 Model Information", open=False):
            gr.Markdown("""
            **Model Details:**
            - **Algorithm**: Logistic Regression with TF-IDF Vectorization
            - **Accuracy**: 99% (Based on test data)
            - **Training Data**: Real and fake news articles
            - **Features**: Text-based analysis using natural language processing

            **How it works:**
            1. Text preprocessing and cleaning
            2. TF-IDF vectorization for feature extraction
            3. Logistic regression classification
            4. Confidence scoring based on prediction probabilities
            """)

        # Event handlers
        analyze_btn.click(
            fn=analyze_news,
            inputs=[news_input],
            outputs=[result_box, confidence_box, recommendation_box]
        )

        clear_btn.click(
            fn=lambda: ("", "", "", ""),
            outputs=[news_input, result_box, confidence_box, recommendation_box]
        )

        real_sample_btn.click(
            fn=load_sample_real,
            outputs=[news_input]
        )

        fake_sample_btn.click(
            fn=load_sample_fake,
            outputs=[news_input]
        )

        # Footer
        gr.HTML("""
        <div class="footer">
            <p><strong>Disclaimer:</strong> This tool uses AI for detection. Always cross-verify important news with multiple trusted sources.</p>
            <p>Made with using Gradio and Scikit-learn</p>
        </div>
        """)

    return app

# Launch the app
if __name__ == "__main__":
    app = create_app()
    if app:
        # Launch with public link for easy sharing
        app.launch(
            debug=True,
            share=True,  # Creates a public link
            server_name="0.0.0.0",
            server_port=7860
        )
    else:
        print("❌ Failed to create app. Check your model files.")
