from flask import Flask, render_template, request, jsonify
import joblib
from scraper import extract_reviews
import time

app = Flask(__name__)

# Load model + vectorizer
try:
    model_data = joblib.load("model/fake_review_model.pkl")
    vectorizer_data = joblib.load("model/tfidf_vectorizer.pkl")

    model = model_data["model"]
    vectorizer = vectorizer_data["vectorizer"]

except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    vectorizer = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    url = request.form.get("url", "").strip()
    
    if not url:
        return render_template("result.html",
                             error="Please provide a valid URL!")
    
    # Validate URL
    # if not any(domain in url.lower() for domain in ["amazon","amzn", "flipkart"]):
    #     return render_template("result.html",
    #                          error="Please provide an Amazon or Flipkart product URL!")

    # Check if models are loaded
    if model is None or vectorizer is None:
        return render_template("result.html",
                             error="ML models not loaded. Please check model files!")

    # Step 1: Extract Reviews
    start_time = time.time()
    reviews = extract_reviews(url, limit=50)
    scrape_time = round(time.time() - start_time, 2)

    if len(reviews) == 0:
        return render_template("result.html",
                             error="No reviews found! The page might be blocking scraping or has no reviews.")

    fake_reviews = []
    genuine_reviews = []
    fake_confidence = []
    genuine_confidence = []

    # Step 2: Predict each review
    for review in reviews:
        if not review or len(review.strip()) < 10:  # Skip very short reviews
            continue
            
        try:
            vec = vectorizer.transform([review])
            pred = model.predict(vec)[0]
            
            # Get prediction probability if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(vec)[0]
                confidence = round(max(proba) * 100, 1)
            else:
                confidence = None

            if pred == 1:
                fake_reviews.append(review)
                if confidence:
                    fake_confidence.append(confidence)
            else:
                genuine_reviews.append(review)
                if confidence:
                    genuine_confidence.append(confidence)
        except Exception as e:
            print(f"Error predicting review: {e}")
            continue

    total_analyzed = len(fake_reviews) + len(genuine_reviews)
    fake_percentage = round((len(fake_reviews) / total_analyzed * 100), 1) if total_analyzed > 0 else 0
    genuine_percentage = round((len(genuine_reviews) / total_analyzed * 100), 1) if total_analyzed > 0 else 0

    return render_template("result.html",
                         fake_reviews=list(zip(fake_reviews, fake_confidence)) if fake_confidence else [(r, None) for r in fake_reviews],
                         genuine_reviews=list(zip(genuine_reviews, genuine_confidence)) if genuine_confidence else [(r, None) for r in genuine_reviews],
                         total=total_analyzed,
                         fake_percentage=fake_percentage,
                         genuine_percentage=genuine_percentage,
                         scrape_time=scrape_time)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """API endpoint for AJAX requests"""
    data = request.get_json()
    url = data.get("url", "").strip()
    
    if not url:
        return jsonify({"error": "URL is required"}), 400
    
    try:
        reviews = extract_reviews(url, limit=50)
        
        if len(reviews) == 0:
            return jsonify({"error": "No reviews found"}), 404
        
        results = {
            "total": len(reviews),
            "fake_count": 0,
            "genuine_count": 0,
            "reviews": []
        }
        
        for review in reviews:
            if len(review.strip()) < 10:
                continue
                
            vec = vectorizer.transform([review])
            pred = model.predict(vec)[0]
            
            review_data = {
                "text": review,
                "is_fake": bool(pred == 1)
            }
            
            if pred == 1:
                results["fake_count"] += 1
            else:
                results["genuine_count"] += 1
                
            results["reviews"].append(review_data)
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)