# 🛍️ Product Review Sentiment Analysis

A deep learning-based web application that classifies product reviews as **positive** or **negative** using a **1D Convolutional Neural Network (1D CNN)**. The model is deployed using **Flask** with a **Bootstrap 5 + JavaScript** frontend for real-time sentiment prediction.

---

## 📌 Features

- 1D CNN model trained on labeled review data  
- NLTK for text preprocessing (stopword removal, punctuation cleanup)  
- Flask-based backend with live prediction  
- Bootstrap 5 & JavaScript-powered responsive UI  
- Model and tokenizer loaded using `pickle`  

---

## 🧠 Technologies Used

- Python  
- Keras / TensorFlow  
- NLTK  
- Flask  
- HTML, Bootstrap 5, JavaScript  
- Google Colab (for model training)

---

## ⚙️ How It Works

1. **Input Review**: User enters product review.  
2. **Preprocessing**: Review is cleaned using NLTK.  
3. **Tokenization & Padding**: Text is tokenized using the saved tokenizer.  
4. **Prediction**: Pre-trained 1D CNN model predicts sentiment.  
5. **Output**: Sentiment is displayed as *Positive* or *Negative* on the frontend.

---

## 📂 Project Structure

```
.
├── backend.py              # Flask backend with prediction route
├── index.html              # Frontend (Bootstrap 5 + JS)
├── templates/
│   └── index.html          # Flask template (if using Jinja)
└── README.md
```

---

## 🚀 How to Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/pooja-joglekar/Product-Review-Sentiment-Analysis-Using-1D-CNN.git
   cd Product-Review-Sentiment-Analysis-Using-1D-CNN
   ```

2. Install dependencies:
   ```bash
   pip install flask nltk tensorflow keras
   ```

3. Run the app:
   ```bash
   python backend.py
   ```
   

## 📎 Link

🔗 GitHub Repository: [https://github.com/pooja-joglekar/Product-Review-Sentiment-Analysis-Using-1D-CNN/]

---

## 👩‍💻 Author

**Pooja Laxman Joglekar**  
MSc.IT | IT Student    
LinkedIn: (www.linkedin.com/in/pooja-joglekar)

## 📝 License

This project is licensed under the [MIT License](LICENSE).  
