from pathlib import Path

from flask import Flask, render_template, request
import pickle
import numpy as np


def load_pickle(*candidates):
    base_dir = Path(__file__).resolve().parent

    for filename in candidates:
        file_path = base_dir / filename
        if not file_path.exists() or file_path.stat().st_size == 0:
            continue
        with file_path.open("rb") as file:
            return pickle.load(file)

    raise FileNotFoundError(
        f"No non-empty pickle file found. Checked: {', '.join(candidates)}"
    )


popular_df = load_pickle(
    "popular_df.pkl",
    "popular_df (1).pkl",
    "popular_df(1).pkl",
)
books = load_pickle("books.pkl")
pt = load_pickle("pt.pkl")
similarity_scores = load_pickle("similarity_scores.pkl")
app = Flask(__name__)


def build_recommendations(book_title):
    if book_title not in pt.index:
        return []

    book_idx = np.where(pt.index == book_title)[0][0]
    similar_items = sorted(
        list(enumerate(similarity_scores[book_idx])),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    data = []
    for item_index, _ in similar_items:
        temp_df = books[books['Book-Title'] == pt.index[item_index]]
        temp_df = temp_df.drop_duplicates('Book-Title')
        if temp_df.empty:
            continue

        row = temp_df.iloc[0]
        data.append(
            {
                "title": row["Book-Title"],
                "author": row["Book-Author"],
                "image": row["Image-URL-M"],
            }
        )

    return data


@app.route('/')
def index():
    return render_template(
        'index.html',
        book_name=popular_df['Book-Title'].tolist(),
        author=popular_df['Book-Author'].tolist(),
        image=popular_df['Image-URL-M'].tolist(),
        votes=popular_df['num_ratings'].tolist(),
        rating=popular_df['avg_rating'].tolist()
    )

@app.route('/recommend')
def recommend_ui():
    return render_template(
        'recommend.html',
        data=[],
        error=None,
        user_input='',
        book_titles=pt.index.tolist()
    )


@app.route('/recommend_book', methods=['POST'])
def recommend():
    user_input = request.form.get('user_input', '').strip()
    data = build_recommendations(user_input) if user_input else []
    error = None

    if not user_input:
        error = 'Enter a book title to get recommendations.'
    elif not data:
        error = 'Book not found in the recommendation model. Choose a title from the list.'

    return render_template(
        'recommend.html',
        data=data,
        error=error,
        user_input=user_input,
        book_titles=pt.index.tolist()
    )


if __name__ == '__main__':
    app.run(debug=True)
