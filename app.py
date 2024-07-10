from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from apscheduler.schedulers.background import BackgroundScheduler
import bcrypt
import numpy as np
import datetime as dt
import ta 
import pandas as pd
import yfinance as yf
import seaborn as sns
import pickle
import io,os,re
import base64
import json
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
plt.style.use('ggplot')
import locale
locale.setlocale(locale.LC_ALL, 'en_IN.utf8')
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.secret_key = "your_secret_key"

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
    
df=pd.read_csv("nifty_data.csv")
df.set_index('index',inplace=True)

# User Authentication
@app.route("/")
def index():
    if "user" in session:
        return render_template("index.html", user=session["user"])
    return redirect(url_for("login"))
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"].encode('utf-8')

        user = User.query.filter_by(username=username).first()
        if user and bcrypt.checkpw(password, user.password.encode('utf-8')):
            session["user"] = username
            return redirect(url_for("index"))
        
        return render_template("login.html", error="Invalid username or password")
    return render_template("login.html")

def validate_password(password):
    # At least 8 characters long, 1 digit, 1 uppercase, 1 lowercase, 1 special character
    regex = r"^(?=.*\d)(?=.*[a-z])(?=.*[A-Z])(?=.*[!@#$%^&*()_+\-=\[\]{};:'\",.<>?]).{8,}$"
    return re.match(regex, password) is not None

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        name = request.form["name"]
        phone = request.form["phone"]

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return render_template("register.html", error="Username already taken")

        if not validate_password(password):
            return render_template("register.html", error="Password does not meet requirements")

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        new_user = User(username=username, password=hashed_password.decode('utf-8'), name=name, phone=phone)
        db.session.add(new_user)
        db.session.commit()

        session["user"] = username
        return redirect(url_for("index"))

    return render_template("register.html")
@app.route("/logout")
def logout():
    # Remove the user from the session
    session.pop("user", None)
    # Redirect to the login page
    return redirect(url_for("login"))
# Stock Screening
@app.route("/visualization", methods=["GET", "POST"])
def visualization():

    # Generate plots
    plot_url_20 = generate_plot(pd.read_csv(os.path.join('data',"data20True.csv")), 'Top Performers of the Past Month')
    plot_url_240 = generate_plot(pd.read_csv(os.path.join('data',"data240True.csv")), 'Top Performers of the Year')
    plot_url_20n = generate_plot(pd.read_csv(os.path.join('data',"data20False.csv")), 'Top Loser of the Past Month')
    plot_url_240n = generate_plot(pd.read_csv(os.path.join('data',"data240False.csv")), 'Top Loser of the Past Year')
    plot_url_1200 = generate_plot(pd.read_csv(os.path.join('data',"data2400True.csv")), 'Top Performers of the 5 year')  
    

    # Render template with plot URLs
    return render_template('visualization.html', plot_url_20=plot_url_20, plot_url_240=plot_url_240, plot_url_20N=plot_url_20n, plot_url_240N=plot_url_240n,plot_url_1200=plot_url_1200)

def generate_plot(dataframe, title,no_of_stock=6):
    fig, ax = plt.subplots()
    for i in range(1, no_of_stock+1):
        ax.plot(dataframe.iloc[:, i], label=df.loc[dataframe.columns[i]]['Company Name'])
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1))
    ax.legend()
    fig.set_size_inches(8, 6)
    ax.set_title(title)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')
def compare_stock(num, order=True,day="1d",num_stock=5):
    with open(os.path.join('data',f'my_dict{day}.pickle'), 'rb') as handle:
        ohlc_data = pickle.load(handle)

    tickers=ohlc_data.keys()
    compare = {}
    for ticker in tickers:
      close_prices = ohlc_data[ticker]['Price']
      daily_returns = (close_prices.pct_change() + 1).iloc[-num:]
      cumulative_returns = (daily_returns).cumprod()
      compare[ticker] = cumulative_returns
    df = pd.DataFrame(compare)

   # Sort the tickers based on the last cumulative return value in descending order
    sorted_tickers = df.iloc[-1].sort_values(ascending=not order).index

   # Create a new DataFrame with the sorted tickers
    sorted_df = df[sorted_tickers]
    sorted_df=(sorted_df-1)
    sorted_df.to_csv(os.path.join('data',f"data{num}{order}.csv"))

    return pd.DataFrame(sorted_df.iloc[:,1:num_stock])
def make_clickable(val):
    return f'<a href="https://www.google.com/finance/quote/{df[df["Company Name"]==val]["Symbol"].str.replace(".NS","").to_list()[0]}:NSE" target="_blank">{val}</a>'
# Stock Recommendations
@app.route("/screener", methods=["GET", "POST"])
def screener():
    df=pd.read_csv(os.path.join('data',"Nifty_Result1d.csv"))
    df['Company Name'] = df['Company Name'].apply(make_clickable)

    html_table = df.to_html(classes="sortable-table",escape=False, index=False)
    return render_template("screener.html", table=html_table)
def indicater(df):
        df['MACD hist']=ta.trend.macd_diff(df['Close'])
        df['ADX']=ta.trend.adx(df["High"], df["Low"], df["Close"], window=14)
        df['ATR']=ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)
        df["EMA 200"] = ta.trend.EMAIndicator(df['Close'], window=200, fillna=False).ema_indicator()
        df['RSI']= ta.momentum.rsi(df["Close"], window=14)
        df['Stochastic Oscillator']=ta.momentum.stoch(df["High"], df["Low"], df["Close"], window=14, smooth_window=3)
        df['MACD hist sloap*']=df['MACD hist'].diff()
        df["EMA 200 Sloap"]=df["EMA 200"].diff()/df['Close']
        return df
def run(interval='1d'):
    df=pd.read_csv("nifty_data.csv")
    tickers=df['Symbol']

    ohlc_data={}
    dic={'1d':2400,"5m":30,"10m":30,"15m":30}
    start = dt.datetime.today()-dt.timedelta(dic[interval])
    end = dt.datetime.today()

    for ticker in tickers:
        ohlc_data[ticker]=yf.download(ticker, interval=interval,start=start, end=end)

        ohlc_data[ticker]=indicater(ohlc_data[ticker])
        ohlc_data[ticker].drop(["Open","High","Low","Adj Close"],axis=1, inplace=True)
        ohlc_data[ticker].rename(columns = {'Close':'Price'}, inplace = True)
        ohlc_data[ticker]["Company Name"]=df[df['Symbol']==ticker]['Company Name'].iloc[0]
    
    with open(os.path.join('data',f'my_dict{interval}.pickle'), 'wb') as handle:
        pickle.dump(ohlc_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    final_data = [ohlc_data[ticker].iloc[-1] for ticker in tickers]

    df=pd.DataFrame(final_data).set_index("Company Name")
    df.to_csv(os.path.join('data',f"Nifty_Result{interval}.csv"))
    if(interval=='1d'):
        compare_stock(20)
        compare_stock(240)
        compare_stock(20, order=False)
        compare_stock(240, order=False)
        compare_stock(1200,)
        compare_stock(2400,)
 
    if(interval=="15m"):
        compare_stock(24,day="15m")
        compare_stock(144,day="15m")
        compare_stock(24, order=False,day="15m")
        compare_stock(144, order=False,day="15m")
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')

@app.route('/blog')
def blog():
    posts = BlogPost.query.order_by(BlogPost.created_at.desc()).all()
    return render_template('blog.html', posts=posts)

@app.route('/admin', methods=['GET', 'POST'])
def new_post():
    if 'user' not in session or session['user'] != 'Admin':
        return "<h1>Admin only can access this page</h1>"

    posts = BlogPost.query.order_by(BlogPost.created_at.desc()).all()

    if request.method == 'POST':
        if 'new_post' in request.form:
            title = request.form['title']
            content = request.form['content']
            image_file = request.files.get('image_file', None)
            
            if image_file and image_file.filename != '':
                image_filename = secure_filename(image_file.filename)
                image_file.save(os.path.join(app.config['UPLOAD_FOLDER'], image_filename))
            else:
                image_filename = ''

            new_post = BlogPost(title=title, content=content, image_filename=image_filename)
            db.session.add(new_post)
            db.session.commit()
           
        
        elif 'delete_post' in request.form:
            post_id = request.form['delete_post']
            post = BlogPost.query.get_or_404(post_id)
            
            if post.image_filename:
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], post.image_filename)
                if os.path.exists(image_path):
                    os.remove(image_path)
            
            db.session.delete(post)
            db.session.commit()
            

        return redirect(url_for('new_post'))

    return render_template('new_post.html', posts=posts)
# Load the model
filename = 'portfolio.pkl'
with open(filename, 'rb') as file:
    model = pickle.load(file)

avg_returns = {
    'Equity': 0.18,  # Equity mutual funds have provided annual returns of around 15%[^1^][3].
    'Mutual Funds': 0.15,  # This remains the same as the average returns for mutual funds.
    'Debt Funds': 0.07,  # Debt funds have average returns of approximately 9%[^1^][3].
    'Sovereign Gold Bonds': 0.09,  # This remains the same as the average returns for sovereign gold bonds.
    'Government Bonds': 0.07,  # This remains the same as the average returns for government bonds.
    'Public Provident Fund': 0.08,  # This remains the same as the average returns for PPF.
    'Fixed Deposits': 0.075  # This remains the same as the average returns for fixed deposits.
}

def portfolio_allocation(Age, Investor_Type):
    Investor_Type = Investor_Type.replace("Aggressive Investor ", "3")
    Investor_Type = Investor_Type.replace("Moderate Investor ", "2")
    Investor_Type = Investor_Type.replace("Conservative Investor ", "1")
    df = pd.DataFrame({"Age": [Age], 'Investor Type': [Investor_Type]})
    predict_value = model.predict(df)[0]
    df_result = predict_value / predict_value.sum() * 100
    df_round = np.round(df_result)
    columns = ['Equity', 'Mutual Funds', 'Debt Funds', 'Sovereign Gold Bonds', 'Government Bonds', 'Public Provident Fund', 'Fixed Deposits']
    df_allocation = pd.DataFrame([df_round], columns=columns)

    # Calculate CAGR
    cagr = 0
    for col, weight in df_allocation.iloc[0].items():
        cagr += weight * avg_returns[col] / 100

    return df_allocation.to_dict('records')[0], cagr

@app.route('/portfolio', methods=['GET', 'POST'])
def portfolio():
    try:
        allocation = None
        cagr = None
        if request.method == 'POST':
            age = int(request.form['age'])
            investor_type = request.form['investor_type']
            allocation, cagr = portfolio_allocation(age, investor_type)
        return render_template('portfolio.html', allocation=allocation, cagr=cagr)
    except Exception as e:
        print(e)

def format_amount(amount):
    return locale.format_string("%.*f", (2, amount), True)

@app.route('/compound-interest', methods=['GET', 'POST'])
def compound_interest():
    if request.method == 'POST':
        monthly_sip = float(request.form['monthly_sip'])
        time_period = int(request.form['time_period'])
        interest_rate = float(request.form['interest_rate']) / 100

        chart_data = []
        contributions_data = []
        current_value = 0
        for year in range(1, time_period + 1):
            for month in range(12):
                current_value += monthly_sip
                current_value *= (1 + interest_rate / 12)
            chart_data.append(current_value)
            contributions_data.append(monthly_sip * year * 12)

        return render_template('compound_interest.html', monthly_sip=format_amount(int(monthly_sip)), total_investment=format_amount(int(contributions_data[-1])),
                               total_value=format_amount(int(chart_data[-1])), chart_data=chart_data, contributions_data=contributions_data, time_period=time_period)

    return render_template('compound_interest.html')

scheduler = BackgroundScheduler()
scheduler.add_job(run, 'interval', minutes=280)
scheduler.add_job(run, 'interval', minutes=15, args=["15m"])
scheduler.start()
#run("15m")
#run()

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', debug=True)