from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import bcrypt

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users_post.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    name = db.Column(db.String(80))
    phone = db.Column(db.String(20))

    def __repr__(self):
        return f'<User {self.username}>'

class BlogPost(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    image_filename = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<BlogPost {self.title}>'

def view_all_data():
    with app.app_context():
        print("Users:")
        users = User.query.all()
        for user in users:
            print(f"ID: {user.id}, Username: {user.username}, Name: {user.name}, Phone: {user.phone}, password:{user.password}")
        
        print("\nBlog Posts:")
        posts = BlogPost.query.all()
        for post in posts:
            print(f"ID: {post.id}, Title: {post.title}, Content: {post.content[:50]}..., Image: {post.image_filename}, Created At: {post.created_at}")

if __name__ == "__main__":
    view_all_data()
