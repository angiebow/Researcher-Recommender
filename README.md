# Researcher-Recommender
# 📚 Researcher Recommendation System  

A web-based platform that recommends **relevant researchers** based on their publication fingerprints and topic similarity.  
This system utilizes **machine learning (fingerprint vectors + transformer embeddings)** in a **web interface** for querying and exploring academic expertise.  

---

## ✨ Features
- 🔍 Search for researchers by topic or keyword  
- 🧑‍🔬 Recommendation engine using **distance similarity metrics on researcher-topic fingerprints**  
- 🤖 Semantic topic matching using **transformer-based embeddings** 
- 📊 Evaluation metrics: Precision@K, Recall@K, MAP, nDCG  
- 🌐 Web app for interactive exploration  

---

## 🏗️ Tech Stack
**Backend**
- [FastAPI](https://fastapi.tiangolo.com/) — high-performance API  
- [scikit-learn](https://scikit-learn.org/) / [NumPy](https://numpy.org/) — similarity calculations  
- [sentence-transformers](https://www.sbert.net/) — transformer embeddings 
- [PostgreSQL](https://www.postgresql.org/) — database for researchers & topics  

**Frontend**
- [Next.js](https://nextjs.org/) (React) — modern frontend framework  
- [Tailwind CSS](https://tailwindcss.com/) — styling  

**Deployment**
- [Docker](https://www.docker.com/) — containerized services  
- [Railway](https://railway.app/) / [Render](https://render.com/) — hosting options  