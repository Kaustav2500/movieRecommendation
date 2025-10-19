# 🎬 Movie Recommendation System

This project is a **movie recommendation system** that suggests films based on both **content similarity** and **machine learning–predicted quality**.  
It analyzes features such as genre, language, budget, popularity, and ratings to recommend movies similar to a user-selected title.  
A **logistic regression model** predicts the likelihood of a movie being high-quality, and this score is combined with similarity metrics to produce ranked, meaningful recommendations.

---

## 🧠 Features
- Content-based filtering using movie metadata  
- Machine learning model (Logistic Regression) for predicting movie quality  
- Combined scoring system (70% similarity + 30% quality)  
- Automatic explanation of why each movie is similar  
- Optional movie overview previews for quick insights  

---

## 🧩 Technologies Used
- **Python 3**
- **Pandas** and **NumPy** for data processing  
- **Scikit-learn** for modeling and similarity computation  
- **Matplotlib** and **Seaborn** for data visualisation  

---

## ⚙️ How It Works
1. The system loads and preprocesses the movie dataset.  
2. A similarity matrix identifies movies related by genre, language, and other attributes.  
3. A logistic regression model predicts each movie’s quality score.  
4. The two scores are combined into a **final recommendation ranking**.  
5. The app displays the top recommended movies with reasons for similarity and a short overview.



## 📁 Project Structure
