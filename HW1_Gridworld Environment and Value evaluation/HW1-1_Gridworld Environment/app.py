from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    # Default n = 5
    n = 5

    if request.method == 'POST':
        try:
            user_n = int(request.form.get('n', 5))
        except ValueError:
            user_n = 5

        # Clamp n to range [5, 9]
        if user_n < 5:
            n = 5
        elif user_n > 9:
            n = 9
        else:
            n = user_n

    return render_template('index.html', n=n, show_grid=True)

if __name__ == '__main__':
    app.run(debug=True)
