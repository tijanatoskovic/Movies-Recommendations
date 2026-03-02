import sys
import os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QListWidget, QTextEdit, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal
from sklearn.metrics.pairwise import cosine_similarity


class RecommendWorker(QThread):
    finished = pyqtSignal(object)

    def __init__(self, movies, df, embeddings):
        super().__init__()
        self.movies = movies
        self.df = df
        self.embeddings = embeddings

    def run(self):
        try:
            liked_ids = []
            liked_vecs = []
            matched = []
            for title in self.movies:
                matches = self.df[self.df['title'].str.lower().str.contains(title.lower(), regex=False)]
                if matches.empty:
                    print(f'  Warning: "{title}" not found => skipping')
                    continue
                row = matches.iloc[0]
                liked_ids.append(row['movieId'])
                liked_vecs.append(self.embeddings[row.name])
                matched.append(f"{row['title']} ({int(row['year'])})")

            if not liked_vecs:
                self.finished.emit(None)
                return

            query = np.mean(liked_vecs, axis=0, keepdims=True)
            query /= (np.linalg.norm(query) + 1e-8)

            sims = cosine_similarity(query, self.embeddings)[0]

            result = self.df[['movieId', 'title', 'genres', 'year']].copy()
            result['similarity'] = sims
            result = result[~result['movieId'].isin(liked_ids)]
            result = result.sort_values('similarity', ascending=False).head(10)
            # -------------
            result['year'] = result['year'].astype(int)
            result['similarity'] = result['similarity'].round(4)
            # -----------------------

            self.finished.emit((result, matched))
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            self.finished.emit(None)


class MovieGUI(QMainWindow):

    def __init__(self):
        super().__init__()
        self.movies = []
        self.df = None
        self.embeddings = None
        self.load_data()
        self.init_ui()

    def load_data(self):
        try:
            self.embeddings = np.load('models/embeddings_ga.npy')
            self.df = pd.read_csv('data/movie_lookup.csv')
            return
        except Exception as e:
            print(f"Could not load from models/: {e}")

        self.df = None
        self.embeddings = None

    def init_ui(self):
        self.setWindowTitle('Movie Recommender')
        self.setGeometry(100, 100, 600, 450)

        w = QWidget()
        self.setCentralWidget(w)
        v = QVBoxLayout()

        if self.embeddings is None:
            error = QLabel('ERROR: Could not load trained model!')
            error.setStyleSheet('color: red; font-weight: bold;')
            v.addWidget(error)

            instructions = QLabel(
                'Please run the genetic_algorithm.ipynb notebook first,\n'
                'then run the save cell to create models/embeddings_ga.npy'
            )
            v.addWidget(instructions)

        v.addWidget(QLabel('Enter movie:'))

        h = QHBoxLayout()
        self.input = QLineEdit()
        self.input.returnPressed.connect(self.add)
        h.addWidget(self.input)

        add = QPushButton('Add')
        add.clicked.connect(self.add)
        h.addWidget(add)
        v.addLayout(h)

        v.addWidget(QLabel('Your movies:'))
        self.list = QListWidget()
        self.list.setMaximumHeight(100)
        v.addWidget(self.list)

        rm = QPushButton('Remove')
        rm.clicked.connect(self.remove)
        v.addWidget(rm)

        self.btn = QPushButton('Get Recommendations')
        self.btn.clicked.connect(self.recommend)
        self.btn.setEnabled(False)
        v.addWidget(self.btn)

        v.addWidget(QLabel('Results:'))
        self.out = QTextEdit()
        self.out.setReadOnly(True)
        v.addWidget(self.out)

        w.setLayout(v)

    def add(self):
        m = self.input.text().strip()
        if m and m not in self.movies:
            self.movies.append(m)
            self.list.addItem(m)
            self.input.clear()
            self.btn.setEnabled(True)

    def remove(self):
        self.movies.clear()
        self.list.clear()
        self.out.clear()
        self.btn.setEnabled(False)

    def recommend(self):
        if self.df is None or self.embeddings is None:
            QMessageBox.warning(self, 'Error', 'Model not loaded. Please run genetic_algorithm.ipynb first.')
            return

        self.out.setText('Processing...')
        self.btn.setEnabled(False)

        self.w = RecommendWorker(self.movies, self.df, self.embeddings)
        self.w.finished.connect(self.show_results)
        self.w.start()

    def show_results(self, data):
        self.btn.setEnabled(True)

        if data is None:
            self.out.setText('No results found')
            return

        result, matched = data

        txt = 'Matched:\n'
        for m in matched:
            txt += f'  {m}\n'
        txt += f'\nTop {len(result)} recommendations:\n\n'

        for i, (_, r) in enumerate(result.iterrows(), 1):
            txt += f"{i}. {r['title']}\n"
            txt += f"   {r['genres']}\n"
            txt += f"   Year: {int(r['year'])}\n"
            txt += f"   Similarity: {r['similarity']:.4f}\n\n"

        self.out.setText(txt)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MovieGUI()
    w.show()
    sys.exit(app.exec_())