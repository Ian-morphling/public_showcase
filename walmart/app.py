from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/forecast')
def forecast():
    return render_template('forecast_plot.html')

@app.route('/holiday-by-date')
def holiday_by_date():
    return render_template('holiday_effect_by_date.html')

@app.route('/holiday-by-name')
def holiday_by_name():
    return render_template('holiday_effects_by_holiday.html')

if __name__ == '__main__':
    app.run(debug=True)