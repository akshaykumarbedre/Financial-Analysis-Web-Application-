# Stock Market Analysis and Portfolio Management

This Flask-based web application provides tools for stock market analysis, portfolio management, and financial planning. It includes features such as stock screening, visualization, portfolio allocation recommendations, and a compound interest calculator.

## Features

- User authentication and registration
- Stock screening and visualization
- Blog functionality for sharing market insights
- Portfolio allocation recommendations based on age and investor type
- Compound interest calculator
- Real-time stock data updates

## Technologies Used

- Python
- Flask
- SQLAlchemy
- Pandas
- Scikit-learn
- yfinance
- Matplotlib
- Seaborn
- APScheduler

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/stock-market-analysis.git
   cd stock-market-analysis
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up the database:
   ```
   flask db init
   flask db migrate
   flask db upgrade
   ```

5. Run the application:
   ```
   python app.py
   ```

## Usage

1. Register a new account or log in with existing credentials.
2. Explore the stock screening tool to analyze market trends.
3. Use the portfolio allocation feature to get personalized investment recommendations.
4. Calculate potential returns using the compound interest calculator.
5. Read and contribute to the blog for market insights.

## Deployment on AWS

To deploy this application on AWS:

1. Set up an EC2 instance with Ubuntu.
2. Install required software (Python, pip, nginx, etc.) on the instance.
3. Clone the repository to the EC2 instance.
4. Set up a virtual environment and install dependencies.
5. Configure nginx as a reverse proxy for the Flask application.
6. Use Gunicorn to serve the Flask application.
7. Set up environment variables for sensitive information.
8. Configure security groups to allow necessary inbound traffic.

Detailed AWS deployment instructions can be found in the `DEPLOYMENT.md` file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
