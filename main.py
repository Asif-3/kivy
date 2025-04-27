import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process, fuzz
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import AsyncImage
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.graphics import Color, RoundedRectangle
from kivy.core.window import Window
from kivy.uix.textinput import TextInput
from kivy.uix.spinner import Spinner
from kivy.uix.progressbar import ProgressBar
from kivy.animation import Animation
import os

# Set futuristic background color
Window.clearcolor = (0.05, 0.05, 0.1, 1)

class GlowBox(BoxLayout):
    """Custom widget with glowing rounded background"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            Color(0.2, 0.6, 1, 0.6)
            self.rect = RoundedRectangle(radius=[30], pos=self.pos, size=self.size)
        self.bind(pos=self.update_rect, size=self.update_rect)

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

class MovieRecommenderApp(App):
    def build(self):
        self.movies = self.load_data()
        if self.movies.empty:
            return Label(text="Error loading dataset.", font_size=24, color=(1, 0, 0, 1))

        self.similarity = self.compute_similarity(self.movies)

        main_layout = BoxLayout(orientation='vertical', spacing=30, padding=30)

        self.search_input = TextInput(
            hint_text="Type a movie name...",
            size_hint=(1, None),
            height=50,
            font_size=20,
            background_color=(0.1, 0.3, 0.5, 1),
            foreground_color=(1, 1, 1, 1),
            multiline=False,
            padding=(10, 10),
            hint_text_color=(0.7, 0.7, 0.7, 1)
        )
        self.search_input.bind(on_text_validate=self.show_recommendations)

        recommend_button = Button(
            text="Get Recommendations",
            size_hint=(1, None),
            height=60,
            font_size=22,
            background_color=(0.1, 0.7, 0.2, 1),
            color=(1, 1, 1, 1),
            on_press=self.show_recommendations,
            background_normal='',  # Remove default button background
            background_down=''  # Removed pressed_button.png because it can crash if not found
        )

        self.results_layout = GridLayout(cols=1, spacing=20, size_hint_y=None, padding=10)
        self.results_layout.bind(minimum_height=self.results_layout.setter('height'))

        scroll_view = ScrollView(size_hint=(1, 1))
        scroll_view.add_widget(self.results_layout)

        main_layout.add_widget(self.search_input)
        main_layout.add_widget(recommend_button)
        main_layout.add_widget(scroll_view)

        return main_layout

    def load_data(self):
        try:
            df = pd.read_csv("dataset.csv")
            df.rename(columns={'Title': 'title', 'Overview': 'overview', 'Poster_Url': 'poster_url'}, inplace=True)
            df = df.dropna(subset=['title', 'overview']).reset_index(drop=True)
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return pd.DataFrame()

    def compute_similarity(self, df):
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['overview'].fillna(''))
        return cosine_similarity(tfidf_matrix)

    def recommend_movies(self, selected_title):
        try:
            movie_choices = self.movies['title'].tolist()
            matched_title, score = process.extractOne(selected_title, movie_choices, scorer=fuzz.token_sort_ratio)

            if score > 80:
                idx = self.movies[self.movies['title'] == matched_title].index[0]
                distances = list(enumerate(self.similarity[idx]))
                sorted_movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
                recommendations = [(self.movies.iloc[i].title, self.movies.iloc[i].overview, self.movies.iloc[i].poster_url) for i, _ in sorted_movies]
                return recommendations
            else:
                return []
        except IndexError:
            return []

    def create_movie_card(self, title, overview, poster_url):
        card = GlowBox(orientation='vertical', size_hint_y=None, height=500, spacing=10, padding=15)

        # ✅ Safer way to load poster image
        safe_poster_url = poster_url if pd.notna(poster_url) and poster_url else "https://via.placeholder.com/300x450?text=No+Image"

        image = AsyncImage(
            source=safe_poster_url,
            size_hint=(1, None),
            height=250,
            allow_stretch=True,
            keep_ratio=True
        )
        card.add_widget(image)

        card.add_widget(Label(
            text=f"{title}",
            bold=True,
            size_hint_y=None,
            height=40,
            font_size=28,
            color=(0.9, 0.9, 1, 1)
        ))

        card.add_widget(Label(
            text=(overview[:180] + "...") if len(overview) > 180 else overview,
            size_hint_y=None,
            height=90,
            font_size=18,
            color=(0.7, 0.7, 0.9, 1)
        ))

        return card


    def show_recommendations(self, instance=None):
        self.results_layout.clear_widgets()
        selected_movie = self.search_input.text.strip()

        if not selected_movie:
            self.results_layout.add_widget(Label(text="Please enter a movie name to search.", font_size=20, color=(1, 0, 0, 1)))
            return

        recommendations = self.recommend_movies(selected_movie)

        if recommendations:
            for title, overview, poster_url in recommendations:
                movie_card = self.create_movie_card(title, overview, poster_url)
                self.results_layout.add_widget(movie_card)
        else:
            self.results_layout.add_widget(Label(text="No recommendations found.", font_size=20, color=(1, 0, 0, 1)))

if __name__ == '__main__':
    MovieRecommenderApp().run()
